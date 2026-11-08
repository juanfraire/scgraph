"""UC4: the multi-hop continuum, every facet at once (revision).

Answers the reviewers' depth requests (R1: "paths typically span 4 to 8 hops",
"jitters compound nonlinearly"; R2: "a combined scenario in which a contact can
be late, faded, and absent in the same window", "a deeper multi-hop
scenario"). Instances come from `scgraph/scenarios/uc4_continuum.py`: the far
terminal (Dakar) to the Paris gateway over a 60-satellite ISL mesh, with
jittered, coupled, possibly absent UAV hand-offs, cross-plane ISL acquisition
delays, and rain-faded feeder passes sharing a weather factor.

Two execution models, because on a deep route they differ:

  static      the source commits to a route and the route either delivers or
              not (the submitted paper's evaluation model)
                det        CGR on the nominal plan
                bern       Bernoulli-TEG static path (Delta = 10 s)
                scg        SLA-aware, argmax P[T <= D] on the joint laws
                scg_indep  SLA-aware on the independent twin (same marginals,
                           within-contact coupling removed): what the joint
                           law buys, scored on the joint truth
  contingent  every node repairs locally with CGR when a contact fails, as a
              deployed CGR implementation does (routing/reactive.py)
                cgr_react  reactive CGR on the nominal plan
                scg_repair the static SLA-aware route, then CGR repair
                scg_policy the first route chosen by its simulated success
                           UNDER CGR repair (the planning plane scoring
                           candidates against the forwarding plane's actual
                           behaviour), then CGR repair

Candidates include the risk-class routes (`candidate_routes(...,
risk_classes=True)`): on this meshed backbone the safe route shares contacts
with the risky ones, and the gap study (run_gap.py) shows the default
generator misses it in most instances.

Static policies are scored on M_EVAL held-out draws per route; contingent ones
on M_REAL held-out realizations shared by all of them (paired). Nothing is
scored on draws used to choose it.

Calibration (what the joint law buys). Route choice barely depends on the
within-contact coupling here, but the planner's probability statement does:
for the joint planner and the marginals-only planner, the CSVs record the
success probability each predicts for its chosen route (fresh draws of its
own model) against the held-out truth, and, at budget EPS with lower-bound
admission, whether the admitted route's true violation exceeds the budget.

Parts:  sigma sweep at margin 60 s; margin sweep at sigma 40 s. The
compounding of jitter along a route is isolated in run_compound.py.

Usage: python experiments/run_uc4.py [--quick] [--replot]
Outputs: results/uc4_sigma.csv, results/uc4_margin.csv, figures/uc4.{pdf,png}
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap
from _style import COL_BASE1, COL_BASE2, COL_BOUND, COL_SCG, STYLE

from scgraph.arrival import sample_route_deliveries
from scgraph.routing.bernoulli import bernoulli_route
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, pick_route, pick_route_chance
from scgraph.routing.reactive import execute_reactive, pick_route_under_repair
from scgraph.scenarios.uc4_continuum import make_instance
from scgraph.stats import cp_upper
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
M_SEL = 2000          # draws per candidate for static selection
M_POL = 300           # planning realizations for repair-aware selection
M_EVAL = 10_000       # held-out draws per static route
M_REAL = 1000         # held-out realizations for contingent policies
BERN_SLOT = 10.0      # the fine end of the ablation's sweep; deep routes need many slots
EPS = 0.20            # budget for the calibration / admission check
UNCERTAIN = {"uav", "isl_x", "feeder", "haps_up"}

STATIC = ("det", "bern", "scg", "scg_indep")
CONTINGENT = ("cgr_react", "scg_repair", "scg_policy")


def one_instance(task, rng):
    sigma, margin, quick = task["sigma"], task["margin"], task["quick"]
    m_sel = 600 if quick else M_SEL
    m_eval = 2000 if quick else M_EVAL
    m_real = 300 if quick else M_REAL
    rng_e = np.random.default_rng(rng.integers(2**63))   # held-out stream

    g, p = make_instance(rng, sigma, margin=margin)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    mean = g.mean_plan()
    cands = candidate_routes(g, src, dst, size, t0, rng, 40, risk_classes=True)

    routes = {}
    det, _ = cgr_route(mean, src, dst, size, t0)
    routes["det"] = [g.by_id(c.cid) for c in det] if det else None
    routes["bern"], _ = bernoulli_route(g, src, dst, size, t0, D, BERN_SLOT, rng, m=2000)
    routes["scg"], _ = pick_route(g, src, dst, size, t0, D, rng, m=m_sel, cands=cands)
    twin = g.independent_twin()
    r_ind, _ = pick_route(twin, src, dst, size, t0, D, rng, m=m_sel,
                          cands=[[twin.by_id(c.cid) for c in r] for r in cands])
    routes["scg_indep"] = [g.by_id(c.cid) for c in r_ind] if r_ind else None
    r_pol, _ = pick_route_under_repair(g, cands, src, dst, size, t0, D, rng, m=M_POL if not quick else 120,
                                       nominal=mean)

    # calibration: each planner's own prediction for its chosen route, on fresh
    # draws of ITS model, against the held-out truth (below)
    pred = {}
    pred["scg"] = _p_model(routes["scg"], size, t0, D, rng)
    pred["scg_indep"] = _p_model([twin.by_id(c.cid) for c in routes["scg_indep"]]
                                 if routes["scg_indep"] else None, size, t0, D, rng)
    # admission at budget EPS on each model
    adm = {}
    for key, model in (("joint", g), ("indep", twin)):
        mc = [[model.by_id(c.cid) for c in r] for r in cands]
        r_a, st_a = pick_route_chance(model, src, dst, size, t0, D, rng, m=m_sel, eps=EPS,
                                      admission="lcb", cands=mc)
        adm[key] = ([g.by_id(c.cid) for c in r_a] if r_a else None, bool(st_a.get("feasible")))

    out = {"sigma": sigma, "margin": margin, "D_rel": D - t0,
           "nominal_hops": p["nominal_hops"], "reliable_hops": p["reliable_hops"],
           "n_cands": len(cands)}
    for k in STATIC:
        ev = evaluate_route(routes[k], size, t0, D, m_eval, rng_e)
        out[f"{k}_fails"], out[f"{k}_n"] = ev["fails"], ev["n"]
        out[f"{k}_viol"] = ev["viol"]
        out[f"{k}_lat"] = ev["mean_delivery"] - t0
        r = routes[k] or []
        out[f"{k}_hops"] = len(r)
        out[f"{k}_unc"] = sum(c.kind in UNCERTAIN for c in r)

    for k in ("scg", "scg_indep"):
        out[f"{k}_pred"] = pred[k]
    for key, (r_a, feas) in adm.items():
        ev = evaluate_route(r_a, size, t0, D, m_eval, rng_e)
        out[f"adm_{key}_feasible"] = feas
        out[f"adm_{key}_viol"] = ev["viol"]
        out[f"adm_{key}_over"] = bool(feas and ev["viol"] > EPS)

    reals = g.realize_batch(rng_e, m_real)
    first = {"cgr_react": None,
             "scg_repair": [c.cid for c in routes["scg"]] if routes["scg"] else None,
             "scg_policy": [c.cid for c in r_pol] if r_pol else None}
    for k in CONTINGENT:
        arr = np.array([execute_reactive(real, mean, src, dst, size, t0,
                                         first_route=first[k])[0] for real in reals])
        fails = int(np.count_nonzero(~(arr <= D)))
        out[f"{k}_fails"], out[f"{k}_n"] = fails, m_real
        out[f"{k}_viol"] = fails / m_real
        fin = arr[np.isfinite(arr)]
        out[f"{k}_lat"] = float(np.mean(fin)) - t0 if fin.size else math.nan

    return out


def _p_model(route, size, t0, D, rng, m: int = 4000) -> float:
    if not route:
        return 0.0
    t = sample_route_deliveries(route, size, t0, m, rng)
    return float(np.mean(t <= D))


def ci(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)])
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def aggregate(results, key):
    rows = []
    for val in sorted({r[key] for r in results}):
        sub = [r for r in results if r[key] == val]
        row = {key: val, "R": len(sub)}
        for k in STATIC + CONTINGENT:
            v = [r[f"{k}_viol"] for r in sub]
            row[f"{k}_viol"] = float(np.mean(v))
            row[f"{k}_ci"] = ci(v)
            f = sum(r[f"{k}_fails"] for r in sub)
            n = sum(r[f"{k}_n"] for r in sub)
            row[f"{k}_ub"] = cp_upper(f, n, 0.05)
            lat = [r[f"{k}_lat"] for r in sub if np.isfinite(r[f"{k}_lat"])]
            row[f"{k}_lat"] = float(np.mean(lat)) if lat else math.nan
            if k in STATIC:
                row[f"{k}_hops"] = float(np.mean([r[f"{k}_hops"] for r in sub]))
                row[f"{k}_unc"] = float(np.mean([r[f"{k}_unc"] for r in sub]))
        for k in ("scg", "scg_indep"):
            row[f"{k}_pred"] = float(np.mean([r[f"{k}_pred"] for r in sub]))
            # mean over instances of (predicted success - true success)
            row[f"{k}_calib_gap"] = float(np.mean(
                [r[f"{k}_pred"] - (1.0 - r[f"{k}_viol"]) for r in sub]))
        for mk in ("joint", "indep"):
            feas = [r for r in sub if r[f"adm_{mk}_feasible"]]
            row[f"adm_{mk}_feasible"] = len(feas) / len(sub)
            row[f"adm_{mk}_viol"] = float(np.mean([r[f"adm_{mk}_viol"] for r in feas])) if feas else math.nan
            row[f"adm_{mk}_over"] = float(np.mean([r[f"adm_{mk}_over"] for r in feas])) if feas else math.nan
        row["nominal_hops"] = float(np.mean([r["nominal_hops"] for r in sub]))
        row["n_cands"] = float(np.mean([r["n_cands"] for r in sub]))
        rows.append(row)
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def read(path):
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    if "--replot" in sys.argv:
        rows_s = read(ROOT / "results" / "uc4_sigma.csv")
        rows_m = read(ROOT / "results" / "uc4_margin.csv")
    else:
        R = 6 if quick else 30
        sigmas = [0.0, 20.0, 40.0] if quick else [0.0, 10.0, 20.0, 40.0, 60.0]
        margins = [0.0, 120.0] if quick else [-60.0, 0.0, 60.0, 120.0, 240.0]
        tasks = [{"sigma": s, "margin": 60.0, "quick": quick}
                 for s in sigmas for _ in range(R)]
        res_s = pmap(one_instance, tasks, seed=SEED)
        tasks = [{"sigma": 40.0, "margin": m, "quick": quick} for m in margins for _ in range(R)]
        res_m = pmap(one_instance, tasks, seed=SEED + 1)
        (ROOT / "results").mkdir(exist_ok=True)
        rows_s = aggregate(res_s, "sigma")
        rows_m = aggregate(res_m, "margin")
        write(ROOT / "results" / "uc4_sigma.csv", rows_s)
        write(ROOT / "results" / "uc4_margin.csv", rows_m)

    for rows, key in ((rows_s, "sigma"), (rows_m, "margin")):
        for r in rows:
            print(f"{key}={r[key]:6.1f} | " + "  ".join(
                f"{k}={r[f'{k}_viol']:.3f}" for k in STATIC + CONTINGENT)
                + f" | hops det/scg {r['det_hops']:.1f}/{r['scg_hops']:.1f}"
                + f" | calib joint {r['scg_calib_gap']:+.3f} indep {r['scg_indep_calib_gap']:+.3f}"
                + f" | admitted joint {r['adm_joint_feasible']:.2f} indep {r['adm_indep_feasible']:.2f}")

    import matplotlib.pyplot as plt

    dur = ROOT / "results" / "r1_bernoulli_instances.csv"
    if dur.exists():
        # four series in (a): a wider figure (placed at \textwidth, same printed
        # font size) gives three equal panels room for (a)'s legend
        fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.6))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    sty = {
        "det": dict(STYLE["base1"], label="CGR"),
        "bern": dict(color="#7b4fa8", linestyle="-.", marker="^",
                     label=r"Bernoulli TEG ($\Delta$=10 s)"),
        "scg": dict(STYLE["scg"], label="SLA-aware"),
        "cgr_react": dict(color=COL_BASE2, linestyle="--", marker="s", label="CGR + repair"),
        "scg_policy": dict(color="#1b9e77", linestyle="-", marker="D",
                           label="SLA-aware + repair"),
    }
    ax = axes[0]
    x = [r["sigma"] for r in rows_s]
    for k in ("bern", "det", "scg"):
        ax.errorbar(x, [r[f"{k}_viol"] for r in rows_s], yerr=[r[f"{k}_ci"] for r in rows_s],
                    capsize=2, **sty[k])
    # second revision (R1.3): the Bernoulli corner with duration-aware slot bookkeeping,
    # same instances (run_r1_bernoulli.py), when its results are present
    if dur.exists():
        with open(dur) as f:
            inst = [r for r in csv.DictReader(f) if r["sweep"] == "sigma"]
        xs = sorted({float(r["sigma"]) for r in inst})
        v = [[float(r["dur@10_viol"]) for r in inst if float(r["sigma"]) == s] for s in xs]
        ax.errorbar(xs, [float(np.mean(a)) for a in v], yerr=[ci(a) for a in v], capsize=2,
                    color="#7b4fa8", linestyle=":", marker="^", mfc="white",
                    label=r"duration-aware TEG")
    ax.set_xlabel(r"UAV timing jitter $\sigma$ [s]")
    ax.set_ylabel("SLA-violation rate")
    ax.set_title("(a) committed source routes", fontsize=7)
    if dur.exists():
        # four series: the legend sits in a band above the curves
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.legend(loc="upper right", fontsize=5.2, ncol=2, handlelength=1.5,
                  handletextpad=0.4, columnspacing=0.6, borderaxespad=0.2)
    else:
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper left", fontsize=5.5, bbox_to_anchor=(0.0, 0.57), handlelength=2.2)
    ax = axes[1]
    for k in ("cgr_react", "scg_policy"):
        ax.errorbar(x, [r[f"{k}_viol"] for r in rows_s], yerr=[r[f"{k}_ci"] for r in rows_s],
                    capsize=2, **sty[k])
    ax.set_xlabel(r"UAV timing jitter $\sigma$ [s]")
    ax.set_title("(b) with hop-by-hop repair", fontsize=7)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=6)
    ax = axes[2]
    xm = [r["margin"] for r in rows_m]
    for k in ("det", "scg", "cgr_react", "scg_policy"):
        ax.errorbar(xm, [r[f"{k}_viol"] for r in rows_m], yerr=[r[f"{k}_ci"] for r in rows_m],
                    capsize=2, **sty[k])
    ax.set_xlabel("deadline margin [s]")
    ax.set_title(r"(c) $\sigma=40$ s, deadline swept", fontsize=7)
    ax.set_ylim(bottom=0)
    _style.save(fig, str(ROOT / "figures" / "uc4"))
    print(f"done in {time.time() - t_start:.1f}s -> results/uc4_*.csv, figures/uc4.pdf")


if __name__ == "__main__":
    main()
