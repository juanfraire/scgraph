"""E3, second revision (R1.5): what local repair costs at the node.

The best multi-hop results execute with hop-by-hop repair (routing/reactive.py):
a node whose planned contact is absent or closes too early recomputes a CGR
route over its nominal plan from where it is. The paper's onboard cost (under
a microsecond per decision) is the table lookup of a forwarding plane that
does not repair. This experiment times what repair adds, on the instances of
Fig. 9, for the three repaired policies:

  cgr_react   reactive CGR on the nominal plan (the source route is a search too)
  scg_repair  the SLA-aware static route, repaired with CGR
  scg_policy  the route chosen by its simulated success under repair, repaired

Per bundle: the number of repairs, and the wall time of each repair search
(one `cgr_route` over the instance's nominal plan from the repairing node: the
30-minute window of the 3606-contact multi-hop plan that the instance routes
over, 351 to 587 contacts, recorded as `n_contacts`).
The source's own first search (cgr_react only) is timed apart. Offline, per
request: the time of the static SLA-aware selection and of the repair-aware
selection, which runs on the planning plane.

Timing runs in one process (SCGRAPH_WORKERS=1), like Fig. 11. Instances
replay `run_uc4.one_instance` up to the candidate set (same seeds and order).

Usage: SCGRAPH_WORKERS=1 python experiments/run_r1_repair.py [--quick]
Outputs: results/r1_repair.csv (per point and policy),
         results/r1_repair_instances.csv
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _parallel import pmap

from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, pick_route
from scgraph.routing.reactive import execute_reactive, pick_route_under_repair
from scgraph.scenarios.uc4_continuum import make_instance

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013          # run_uc4.SEED
M_SEL = 2000
M_POL = 300
M_REAL = 1000
POLICIES = ("cgr_react", "scg_repair", "scg_policy")


def one_instance(task, rng):
    sigma, margin, quick = task["sigma"], task["margin"], task["quick"]
    m_real = 200 if quick else M_REAL
    rng_e = np.random.default_rng(rng.integers(2**63))   # held-out stream (as run_uc4)
    g, p = make_instance(rng, sigma, margin=margin)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    mean = g.mean_plan()
    cands = candidate_routes(g, src, dst, size, t0, rng, 40, risk_classes=True)

    t = time.perf_counter()
    r_scg, _ = pick_route(g, src, dst, size, t0, D, rng, m=M_SEL, cands=cands)
    t_sel = time.perf_counter() - t
    t = time.perf_counter()
    r_pol, _ = pick_route_under_repair(g, cands, src, dst, size, t0, D, rng,
                                       m=M_POL if not quick else 120, nominal=mean)
    t_pol = time.perf_counter() - t

    first = {"cgr_react": None,
             "scg_repair": [c.cid for c in r_scg] if r_scg else None,
             "scg_policy": [c.cid for c in r_pol] if r_pol else None}
    reals = g.realize_batch(rng_e, m_real)
    out = {"sigma": sigma, "margin": margin, "n_cands": len(cands), "n_contacts": len(g.contacts),
           "t_select_s": t_sel, "t_select_repair_s": t_pol}
    for k in POLICIES:
        rep_times, src_times, n_rep, viol = [], [], [], 0
        for real in reals:
            calls = []

            def replan(node, a, excluded, _calls=calls):
                t1 = time.perf_counter()
                r, _arr = cgr_route(mean, node, dst, size, a, excluded_cids=excluded)
                _calls.append((node == src and a == t0 and not excluded,
                               time.perf_counter() - t1))
                return [c.cid for c in r] if r is not None else None

            arr, reps = execute_reactive(real, mean, src, dst, size, t0,
                                         first_route=first[k], replan=replan)
            viol += not (arr <= D)
            n_rep.append(reps)
            for is_src, dt in calls:
                (src_times if is_src else rep_times).append(dt)
        n_rep = np.array(n_rep)
        rep_ms = np.array(rep_times) * 1e3
        out[f"{k}_viol"] = viol / m_real
        out[f"{k}_rep_mean"] = float(n_rep.mean())
        out[f"{k}_rep_p95"] = float(np.percentile(n_rep, 95))
        out[f"{k}_rep_max"] = int(n_rep.max())
        out[f"{k}_rep_share"] = float(np.mean(n_rep > 0))
        out[f"{k}_n_searches"] = len(rep_ms)
        out[f"{k}_search_ms_med"] = float(np.median(rep_ms)) if rep_ms.size else math.nan
        out[f"{k}_search_ms_p95"] = float(np.percentile(rep_ms, 95)) if rep_ms.size else math.nan
        out[f"{k}_search_ms_max"] = float(rep_ms.max()) if rep_ms.size else math.nan
        out[f"{k}_onboard_ms_per_bundle"] = float(rep_ms.sum() / m_real)
        out[f"{k}_src_ms_med"] = (float(np.median(src_times) * 1e3) if src_times else math.nan)
    return out


def aggregate(res, key):
    rows = []
    for x in sorted({r[key] for r in res}):
        sub = [r for r in res if r[key] == x]
        base = {key: x, "R": len(sub),
                "t_select_s_med": float(np.median([r["t_select_s"] for r in sub])),
                "t_select_repair_s_med": float(np.median([r["t_select_repair_s"] for r in sub]))}
        for k in POLICIES:
            row = dict(base, policy=k)
            for f in ("viol", "rep_mean", "rep_share", "onboard_ms_per_bundle"):
                row[f] = float(np.mean([r[f"{k}_{f}"] for r in sub]))
            row["rep_p95"] = float(np.median([r[f"{k}_rep_p95"] for r in sub]))
            row["rep_max"] = int(max(r[f"{k}_rep_max"] for r in sub))
            med = [r[f"{k}_search_ms_med"] for r in sub if math.isfinite(r[f"{k}_search_ms_med"])]
            p95 = [r[f"{k}_search_ms_p95"] for r in sub if math.isfinite(r[f"{k}_search_ms_p95"])]
            row["search_ms_med"] = float(np.median(med)) if med else math.nan
            row["search_ms_p95"] = float(np.median(p95)) if p95 else math.nan
            row["search_ms_max"] = float(max((r[f"{k}_search_ms_max"] for r in sub
                                              if math.isfinite(r[f"{k}_search_ms_max"])),
                                             default=math.nan))
            src = [r[f"{k}_src_ms_med"] for r in sub if math.isfinite(r[f"{k}_src_ms_med"])]
            row["src_ms_med"] = float(np.median(src)) if src else math.nan
            rows.append(row)
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    quick = "--quick" in sys.argv
    t = time.time()
    R = 3 if quick else 30
    sigmas = [0.0, 40.0] if quick else [0.0, 10.0, 20.0, 40.0, 60.0]
    margins = [0.0, 240.0] if quick else [-60.0, 0.0, 60.0, 120.0, 240.0]
    tasks = [{"sigma": s, "margin": 60.0, "quick": quick} for s in sigmas for _ in range(R)]
    res_s = pmap(one_instance, tasks, seed=SEED)
    tasks = [{"sigma": 40.0, "margin": mg, "quick": quick} for mg in margins for _ in range(R)]
    res_m = pmap(one_instance, tasks, seed=SEED + 1)
    rows = ([{"sweep": "sigma", "margin": 60.0, **r} for r in aggregate(res_s, "sigma")]
            + [{"sweep": "margin", "sigma": 40.0, **r} for r in aggregate(res_m, "margin")])
    (ROOT / "results").mkdir(exist_ok=True)
    sfx = "_quick" if quick else ""
    write(ROOT / "results" / f"r1_repair{sfx}.csv", rows)
    write(ROOT / "results" / f"r1_repair_instances{sfx}.csv",
          [{"sweep": "sigma", **r} for r in res_s] + [{"sweep": "margin", **r} for r in res_m])
    for r in rows:
        print(f"{r['sweep']:6s} sigma={r['sigma']:4.0f} margin={r['margin']:5.0f} {r['policy']:10s}"
              f" viol {r['viol']:.3f} | repairs/bundle {r['rep_mean']:.2f} (share {r['rep_share']:.2f},"
              f" p95 {r['rep_p95']:.0f}, max {r['rep_max']}) | search {r['search_ms_med']:.2f} ms"
              f" (p95 {r['search_ms_p95']:.2f}, max {r['search_ms_max']:.2f})"
              f" | onboard {r['onboard_ms_per_bundle']:.2f} ms/bundle"
              f" | offline sel {r['t_select_s_med']:.2f}s repair-aware {r['t_select_repair_s_med']:.2f}s")
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
