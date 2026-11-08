"""E1, second revision (R1.2): SLA admission on the multi-hop continuum.

Fig. 9 of the paper plots the max-probability router (eq. obj-maxprob), which
has no violation budget. This experiment runs the chance-constrained router
with lower-bound admission (eqs. obj-chance and admit) on the same instances
and candidate sets, at budgets eps in EPSILONS, and reports admitted requests
separately from best-effort ones:

  admitted     some candidate's Clopper-Pearson lower bound (level delta/|Pi|)
               clears 1 - eps; the router sends the fastest such route
  best effort  no candidate clears it; the router sends the max-probability
               route and reports the request as not admitted

Three disjoint samples per request, as everywhere in the paper: the candidate
set (nominal route, suppression alternatives, the CGR routes of 40 sampled
realisations, risk-class routes), M fresh draws per candidate for scoring and
admission, and M_EVAL held-out draws from a separately seeded stream for the
reported violation.

The first three random calls of `one_instance` replay `run_uc4.one_instance`
(held-out seed, instance, candidates) under the same seeds and task order, so
instances and candidate sets are those of Fig. 9.

Usage: python experiments/run_r1_admission.py [--quick]
Outputs: results/r1_admission.csv (per point and budget),
         results/r1_admission_instances.csv (per request and budget)
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

from scgraph.routing.quantile import candidate_routes, route_stats
from scgraph.scenarios.uc4_continuum import make_instance
from scgraph.stats import admission_draws, cp_lower
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013          # run_uc4.SEED: same instances as Fig. 9
EPSILONS = (0.05, 0.10, 0.20)
DELTA = 0.05
M_EVAL = 10_000


def _key(route):
    return tuple(c.cid for c in route) if route else None


def one_instance(task, rng):
    sigma, margin, quick = task["sigma"], task["margin"], task["quick"]
    m_eval = 2000 if quick else M_EVAL
    rng_e = np.random.default_rng(rng.integers(2**63))   # held-out stream (as run_uc4)
    g, p = make_instance(rng, sigma, margin=margin)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    cands = candidate_routes(g, src, dst, size, t0, rng, 40, risk_classes=True)

    # scoring and admission: fresh draws per candidate, enough for the tightest budget
    m = admission_draws(min(EPSILONS), len(cands), DELTA)
    scored = [(r, route_stats(r, size, t0, D, m, rng, latent=g.latent)) for r in cands]
    best_effort = max(scored, key=lambda rs: (rs[1]["p_success"], -rs[1]["mean_delivery"]))

    evals: dict = {}

    def held_out(route):
        k = _key(route)
        if k not in evals:
            evals[k] = evaluate_route(route, size, t0, D, m_eval, rng_e, latent=g.latent)
        return evals[k]

    out = {"sigma": sigma, "margin": margin, "n_cands": len(cands), "m_score": m}
    ev = held_out(best_effort[0])
    out["maxp_viol"], out["maxp_lat"] = ev["viol"], ev["mean_delivery"] - t0
    out["maxp_pred"] = best_effort[1]["p_success"]
    d = DELTA / len(scored)
    for eps in EPSILONS:
        ok = [(r, s) for r, s in scored if cp_lower(s["k"], s["n"], d) >= 1.0 - eps]
        tag = f"e{int(round(eps * 100)):02d}"
        if ok:
            route, st = min(ok, key=lambda rs: rs[1]["mean_delivery"])
        else:
            route, st = best_effort
        ev = held_out(route)
        out[f"{tag}_admitted"] = bool(ok)
        out[f"{tag}_viol"] = ev["viol"]
        out[f"{tag}_viol_ub"] = ev["viol_ub"]
        out[f"{tag}_lat"] = ev["mean_delivery"] - t0
        out[f"{tag}_lcb"] = cp_lower(st["k"], st["n"], d)
        out[f"{tag}_over"] = bool(ok and ev["viol"] > eps)
    return out


def aggregate(res, key):
    rows = []
    for x in sorted({r[key] for r in res}):
        sub = [r for r in res if r[key] == x]
        for eps in EPSILONS:
            tag = f"e{int(round(eps * 100)):02d}"
            adm = [r for r in sub if r[f"{tag}_admitted"]]
            bef = [r for r in sub if not r[f"{tag}_admitted"]]

            def mean(rs, f):
                v = [r[f] for r in rs if math.isfinite(r[f])]
                return float(np.mean(v)) if v else math.nan

            rows.append({
                key: x, "eps": eps, "R": len(sub),
                "admitted": len(adm), "admitted_share": len(adm) / len(sub),
                "adm_viol_mean": mean(adm, f"{tag}_viol"),
                "adm_viol_max": max((r[f"{tag}_viol"] for r in adm), default=math.nan),
                "adm_over": sum(r[f"{tag}_over"] for r in adm),
                "adm_lat_mean": mean(adm, f"{tag}_lat"),
                "be_viol_mean": mean(bef, f"{tag}_viol"),
                "be_lat_mean": mean(bef, f"{tag}_lat"),
                "all_viol_mean": mean(sub, f"{tag}_viol"),
                "maxp_viol_mean": mean(sub, "maxp_viol"),
                "maxp_lat_mean": mean(sub, "maxp_lat"),
            })
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    quick = "--quick" in sys.argv
    t = time.time()
    R = 4 if quick else 30
    sigmas = [0.0, 40.0] if quick else [0.0, 10.0, 20.0, 40.0, 60.0]
    margins = [0.0, 120.0] if quick else [-60.0, 0.0, 60.0, 120.0, 240.0]
    tasks = [{"sigma": s, "margin": 60.0, "quick": quick} for s in sigmas for _ in range(R)]
    res_s = pmap(one_instance, tasks, seed=SEED)
    tasks = [{"sigma": 40.0, "margin": mg, "quick": quick} for mg in margins for _ in range(R)]
    res_m = pmap(one_instance, tasks, seed=SEED + 1)
    rows = ([{"sweep": "sigma", **r} for r in aggregate(res_s, "sigma")]
            + [{"sweep": "margin", **r} for r in aggregate(res_m, "margin")])
    for r in rows:
        r.setdefault("sigma", 40.0)
        r.setdefault("margin", 60.0)
    rows = [{k: r[k] for k in ["sweep", "sigma", "margin"] + [k for k in r if k not in
             ("sweep", "sigma", "margin")]} for r in rows]
    (ROOT / "results").mkdir(exist_ok=True)
    sfx = "_quick" if quick else ""
    write(ROOT / "results" / f"r1_admission{sfx}.csv", rows)
    write(ROOT / "results" / f"r1_admission_instances{sfx}.csv",
          [{"sweep": "sigma", **r} for r in res_s] + [{"sweep": "margin", **r} for r in res_m])
    for r in rows:
        print(f"{r['sweep']:6s} sigma={r['sigma']:4.0f} margin={r['margin']:5.0f} eps={r['eps']:.2f}"
              f" | admitted {r['admitted']:2d}/{r['R']}"
              f" viol {r['adm_viol_mean']:.4f} (max {r['adm_viol_max']:.4f}, over {r['adm_over']})"
              f" lat {r['adm_lat_mean']:.0f}s"
              f" | best effort viol {r['be_viol_mean']:.4f} lat {r['be_lat_mean']:.0f}s"
              f" | max-P viol {r['maxp_viol_mean']:.4f}")
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
