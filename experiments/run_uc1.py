"""H1 experiment: SLA-violation rate vs timing jitter (UC1).

For each jitter level sigma, R random instances are generated. On each:
  det  - deterministic CGR on the nominal (mean) plan, evaluated under the laws
  bern - the Bernoulli-TEG corner of G at a tractable slot width (ablation)
  scg  - SLA-aware route: argmax P[T <= D] over the candidate set (eq. 6)
All routes are then scored with M Monte-Carlo draws under the FULL laws; the
per-instance violation probability is averaged over instances with a 95% CI.

Revision (held-out evaluation). The submitted version reported the SLA-aware
router's violation as 1 - P_hat on the same draws that selected its route,
which is optimistic by construction. Selection below is unchanged draw for
draw, so the instances and the chosen routes are the submitted ones, and every
policy is now ALSO scored on an independent stream (SEED + 2) with M_EVAL
draws. The *_ho columns are the held-out numbers the paper reports; the old
in-sample SLA-aware column is kept as viol_scg_insample for the record. Pooled
failure counts give exact one-sided 95% upper bounds (viol_ub_*), so a
measured zero is reported together with what it actually supports.

The Bernoulli ablation draws from a dedicated RNG stream so that adding it
leaves the det/scg numbers bit-identical to the runs without it.

Usage: python experiments/run_uc1.py [--quick] [--skeleton DIR --tag NAME]
Outputs: results/uc1.csv, figures/uc1_h1.{pdf,png}
         (with --tag: results/uc1_<tag>.csv, figures/uc1_h1_<tag>.{pdf,png})

--skeleton runs the identical experiment on another CPD plan. The revision
uses it with scenarios/uc1_raw, the same constellation and sites WITHOUT fair
contact-plan design, so each pass serves the event site and the gateway at the
same time (R2: the slow satellite path is partly an artefact of fair design).
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
from _style import STYLE

from scgraph.arrival import sample_route_deliveries
from scgraph.routing.bernoulli import bernoulli_route
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import pick_route
from scgraph.scenarios.uc1_event import make_instance_cpd as make_instance
from scgraph.stats import cp_upper
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260709
# Slot width for the Bernoulli-TEG ablation. ~12 slots over the UC1 horizon,
# the order of magnitude at which the slotted MDPs of the uncertainty-aware DTN
# line stay tractable. figures/subsumption.pdf sweeps this.
BERN_SLOT = 30.0
M_EVAL = 10_000          # held-out draws per route and instance


def run(quick: bool = False, skeleton=None):
    sigmas = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0]
    R = 20 if quick else 60          # instances per sigma
    M = 800 if quick else 2000       # MC draws per route evaluation
    rng = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED + 1)   # ablation-only stream
    rng_e = np.random.default_rng(SEED + 2)   # held-out evaluation stream
    m_eval = 2000 if quick else M_EVAL

    rows = []
    for sigma in sigmas:
        viol_det, viol_scg, lat_det, lat_scg = [], [], [], []
        viol_bern = []
        ho = {k: {"viol": [], "fails": 0, "n": 0, "lat": []} for k in ("det", "bern", "scg")}

        def holdout(key, route, g, size, t0, D):
            ev = evaluate_route(route, size, t0, D, m_eval, rng_e)
            ho[key]["viol"].append(ev["viol"])
            ho[key]["fails"] += ev["fails"]
            ho[key]["n"] += ev["n"]
            if math.isfinite(ev["mean_delivery"]):
                ho[key]["lat"].append(ev["mean_delivery"] - t0)

        for _ in range(R):
            g, p = make_instance(rng, sigma, skeleton=skeleton)
            src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]

            # deterministic baseline: route on the nominal plan, live with the laws
            det_route, _ = cgr_route(g.mean_plan(), src, dst, size, t0)
            det_sc = [g.by_id(c.cid) for c in det_route] if det_route else None
            if det_sc is None:
                viol_det.append(1.0)
            else:
                t = sample_route_deliveries(det_sc, size, t0, M, rng)
                viol_det.append(float(np.mean(~(t <= D))))
                if np.isfinite(t).any():
                    # end-to-end latency, not absolute clock time: with a CPD
                    # skeleton the release time sits deep inside the plan
                    lat_det.append(float(np.mean(t[np.isfinite(t)])) - t0)

            # SLA-aware router
            route, stats = pick_route(g, src, dst, size, t0, D, rng, m=M)
            viol_scg.append(1.0 - stats["p_success"] if route else 1.0)
            if route and math.isfinite(stats["mean_delivery"]):
                lat_scg.append(stats["mean_delivery"] - t0)

            # ablation: route on the Bernoulli-TEG projection, live with the laws
            b_route, _ = bernoulli_route(
                g, src, dst, size, t0, D, BERN_SLOT, rng_b, m=M
            )
            if b_route is None:
                viol_bern.append(1.0)
            else:
                tb = sample_route_deliveries(b_route, size, t0, M, rng_b)
                viol_bern.append(float(np.mean(~(tb <= D))))

            # held-out scoring of all three committed routes (revision)
            holdout("det", det_sc, g, size, t0, D)
            holdout("bern", b_route, g, size, t0, D)
            holdout("scg", route, g, size, t0, D)

        def ci(xs):
            xs = np.asarray(xs)
            return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0

        row = {
            "sigma": sigma,
            # held-out (the reported numbers)
            "viol_det": float(np.mean(ho["det"]["viol"])), "ci_det": ci(ho["det"]["viol"]),
            "viol_bern": float(np.mean(ho["bern"]["viol"])), "ci_bern": ci(ho["bern"]["viol"]),
            "viol_scg": float(np.mean(ho["scg"]["viol"])), "ci_scg": ci(ho["scg"]["viol"]),
            "lat_det": float(np.mean(ho["det"]["lat"])) if ho["det"]["lat"] else math.nan,
            "lat_scg": float(np.mean(ho["scg"]["lat"])) if ho["scg"]["lat"] else math.nan,
        }
        for k in ("det", "bern", "scg"):
            row[f"fails_{k}"] = ho[k]["fails"]
            row[f"n_{k}"] = ho[k]["n"]
            row[f"viol_ub_{k}"] = cp_upper(ho[k]["fails"], ho[k]["n"], 0.05)
            row[f"inst_max_{k}"] = float(np.max(ho[k]["viol"]))
        # the submitted, selection-stream numbers, kept for the record
        row.update({
            "viol_det_v1": float(np.mean(viol_det)),
            "viol_bern_v1": float(np.mean(viol_bern)),
            "viol_scg_insample": float(np.mean(viol_scg)),
            "lat_det_v1": float(np.mean(lat_det)) if lat_det else math.nan,
            "lat_scg_v1": float(np.mean(lat_scg)) if lat_scg else math.nan,
            "R": R, "M": M, "M_eval": m_eval, "bern_slot": BERN_SLOT,
        })
        rows.append(row)
        print(
            f"sigma={sigma:5.1f}  viol(det)={row['viol_det']:.4f}"
            f"  viol(bern)={row['viol_bern']:.4f}"
            f"  viol(scg)={row['viol_scg']:.5f} [ub {row['viol_ub_scg']:.2e},"
            f" in-sample {row['viol_scg_insample']:.4f}]"
            f"  lat det/scg={row['lat_det']:.0f}/{row['lat_scg']:.0f}"
        )
    return rows


def main():
    quick = "--quick" in sys.argv
    skeleton, tag = None, ""
    if "--skeleton" in sys.argv:
        from scgraph.io import load_cpd_plan

        skeleton = load_cpd_plan(ROOT / sys.argv[sys.argv.index("--skeleton") + 1], layer="space")
    if "--tag" in sys.argv:
        tag = "_" + sys.argv[sys.argv.index("--tag") + 1]
    t_start = time.time()
    _style.setup()
    rows = run(quick, skeleton=skeleton)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"uc1{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5))

    s = [r["sigma"] for r in rows]
    ax1.errorbar(s, [r["viol_det"] for r in rows], yerr=[r["ci_det"] for r in rows],
                 label="deterministic CGR (mean plan)", capsize=2, **STYLE["base1"])
    ax1.errorbar(s, [r["viol_bern"] for r in rows], yerr=[r["ci_bern"] for r in rows],
                 label=rf"Bernoulli TEG ($\Delta={BERN_SLOT:.0f}$ s)", capsize=2,
                 color="#7b4fa8", linestyle="-.", marker="^")
    ax1.errorbar(s, [r["viol_scg"] for r in rows], yerr=[r["ci_scg"] for r in rows],
                 label="SLA-aware (stochastic contact graph)", capsize=2, **STYLE["scg"])
    ax1.set_xlabel(r"timing jitter $\sigma(T_s, T_e)$ [s]")
    ax1.set_ylabel("SLA-violation rate")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left")

    ax2.plot(s, [r["lat_det"] for r in rows], label="deterministic CGR", **STYLE["base1"])
    ax2.plot(s, [r["lat_scg"] for r in rows], label="SLA-aware", **STYLE["scg"])
    ax2.set_xlabel(r"timing jitter $\sigma(T_s, T_e)$ [s]")
    ax2.set_ylabel("mean delivery time [s]\n(successful deliveries)")
    ax2.legend(loc="upper left")

    _style.save(fig, str(ROOT / "figures" / f"uc1_h1{tag}"))
    print(f"done in {time.time() - t_start:.1f}s -> results/uc1{tag}.csv, figures/uc1_h1{tag}.pdf")


if __name__ == "__main__":
    main()
