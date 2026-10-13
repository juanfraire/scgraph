"""Subsumption ablation: the Bernoulli-TEG corner of G, as a function of slot width.

The paper claims the discrete-time, on/off probabilistic TEG of the
uncertainty-aware DTN line (BRUF / RUCoP / LSS) is a coarse marginal of eq. (1).
This experiment builds that marginal explicitly (routing/bernoulli.py), routes on
it, and scores the resulting route under the FULL laws, on the same UC1 instances
that produce H1.

Sweeping the slot width interpolates between the two corners: as the grid
coarsens the projection loses the sub-slot timing information and the policy
decays toward deterministic CGR; as it refines, the policy converges to the
SLA-aware one. Both reference levels are measured on the same instances.

Revision: selection is unchanged draw for draw, and every policy is also
scored on an independent stream (SEED + 2, M_EVAL draws). The SLA-aware
reference is now drawn at its measured held-out level instead of at zero.

Usage: python experiments/run_ablation.py [--quick]
Outputs: results/ablation_slots.csv, figures/subsumption.{pdf,png}
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
from _style import STYLE, COL_BASE1, COL_SCG

from scgraph.arrival import sample_route_deliveries
from scgraph.routing.bernoulli import bernoulli_route
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import pick_route
from scgraph.scenarios.uc1_event import make_instance_cpd as make_instance
from scgraph.stats import cp_upper
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260727
M_EVAL = 10_000


def run(quick: bool = False):
    slots = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0]
    sigmas = [30.0, 60.0]
    R = 15 if quick else 40
    M = 800 if quick else 2000
    rng = np.random.default_rng(SEED)
    rng_e = np.random.default_rng(SEED + 2)   # held-out evaluation stream
    m_eval = 2000 if quick else M_EVAL

    def ho(route, size, t0, D):
        return evaluate_route(route, size, t0, D, m_eval, rng_e)

    rows = []
    for sigma in sigmas:
        # one instance set per sigma, shared by every policy so the comparison
        # is paired (same topologies, same deadlines)
        instances = [make_instance(rng, sigma) for _ in range(R)]

        viol_det, viol_scg = [], []
        ho_det, ho_scg = [], []
        f_scg, n_scg = 0, 0
        for g, p in instances:
            src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
            det, _ = cgr_route(g.mean_plan(), src, dst, size, t0)
            det_sc = [g.by_id(c.cid) for c in det] if det is not None else None
            if det is None:
                viol_det.append(1.0)
            else:
                t = sample_route_deliveries(det_sc, size, t0, M, rng)
                viol_det.append(float(np.mean(~(t <= D))))
            _r, st = pick_route(g, src, dst, size, t0, D, rng, m=M)
            viol_scg.append(1.0 - st["p_success"] if _r else 1.0)
            ho_det.append(ho(det_sc, size, t0, D)["viol"])
            ev = ho(_r, size, t0, D)
            ho_scg.append(ev["viol"])
            f_scg += ev["fails"]
            n_scg += ev["n"]

        for dw in slots:
            viol_b, ho_b = [], []
            for g, p in instances:
                src, dst, size, t0, D = (
                    p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
                )
                r, _pred = bernoulli_route(g, src, dst, size, t0, D, dw, rng, m=M)
                ho_b.append(ho(r, size, t0, D)["viol"])
                if r is None:
                    viol_b.append(1.0)
                    continue
                t = sample_route_deliveries(r, size, t0, M, rng)
                viol_b.append(float(np.mean(~(t <= D))))

            def ci(xs):
                xs = np.asarray(xs)
                return (
                    1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
                )

            rows.append(
                {
                    "sigma": sigma,
                    "slot": dw,
                    # held-out (reported)
                    "viol_bern": float(np.mean(ho_b)),
                    "ci_bern": ci(ho_b),
                    "viol_det": float(np.mean(ho_det)),
                    "viol_scg": float(np.mean(ho_scg)),
                    "viol_ub_scg": cp_upper(f_scg, n_scg, 0.05),
                    # submitted selection-stream numbers, for the record
                    "viol_bern_v1": float(np.mean(viol_b)),
                    "viol_det_v1": float(np.mean(viol_det)),
                    "viol_scg_insample": float(np.mean(viol_scg)),
                    "R": R,
                    "M": M,
                    "M_eval": m_eval,
                }
            )
            print(
                f"sigma={sigma:4.0f}  slot={dw:5.1f}s  "
                f"bern={rows[-1]['viol_bern']:.3f}  "
                f"(det={rows[-1]['viol_det']:.3f}, scg={rows[-1]['viol_scg']:.3f})"
            )
    return rows


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    rows = run(quick)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "ablation_slots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    sigmas = sorted({r["sigma"] for r in rows})
    marks = ["o", "^"]
    for i, sigma in enumerate(sigmas):
        sub = [r for r in rows if r["sigma"] == sigma]
        x = [r["slot"] for r in sub]
        ax.errorbar(
            x, [r["viol_bern"] for r in sub], yerr=[r["ci_bern"] for r in sub],
            label=rf"Bernoulli TEG, $\sigma={sigma:.0f}$ s", capsize=2,
            color="#7b4fa8", linestyle="-", marker=marks[i],
        )
        lvl = sub[0]["viol_det"]
        ax.axhline(lvl, color=COL_BASE1, linestyle="--", linewidth=1.0)
        ax.annotate(
            rf"deterministic CGR, $\sigma={sigma:.0f}$ s",
            xy=(x[0], lvl), xytext=(0, 2), textcoords="offset points",
            ha="left", va="bottom", fontsize=6, color=COL_BASE1,
        )
    # the SLA-aware level is measured (held-out), one value per sigma; on this
    # sweep it is at or within 1e-4 of zero for both, so one line represents both
    scg_lvl = max(r["viol_scg"] for r in rows)
    ax.axhline(scg_lvl, color=COL_SCG, linestyle="-", linewidth=1.2,
               label="SLA-aware routing on the full laws (held-out)")
    ax.set_xscale("log")
    ax.set_xlabel(r"slot width $\Delta$ [s]  (coarser $\rightarrow$)")
    ax.set_ylabel("SLA-violation rate")
    ax.set_ylim(bottom=-0.012)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.88))

    _style.save(fig, str(ROOT / "figures" / "subsumption"))
    print(
        f"done in {time.time() - t_start:.1f}s -> "
        "results/ablation_slots.csv, figures/subsumption.pdf"
    )


if __name__ == "__main__":
    main()
