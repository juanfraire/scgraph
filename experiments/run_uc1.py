"""H1 experiment: SLA-violation rate vs timing jitter (UC1).

For each jitter level sigma, R random instances are generated. On each:
  det  - deterministic CGR on the nominal (mean) plan, evaluated under the laws
  scg  - SLA-aware route: argmax P[T <= D] over the candidate set (eq. 6)
Both routes are then scored with M Monte-Carlo draws; the per-instance
violation probability is averaged over instances with a 95% CI.

Usage: python experiments/run_uc1.py [--quick]
Outputs: results/uc1.csv, figures/uc1_h1.{pdf,png}
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
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import pick_route
from scgraph.scenarios.uc1_event import make_instance

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260709


def run(quick: bool = False):
    sigmas = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0]
    R = 20 if quick else 60          # instances per sigma
    M = 800 if quick else 2000       # MC draws per route evaluation
    rng = np.random.default_rng(SEED)

    rows = []
    for sigma in sigmas:
        viol_det, viol_scg, lat_det, lat_scg = [], [], [], []
        for _ in range(R):
            g, p = make_instance(rng, sigma)
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
                    lat_det.append(float(np.mean(t[np.isfinite(t)])))

            # SLA-aware router
            route, stats = pick_route(g, src, dst, size, t0, D, rng, m=M)
            viol_scg.append(1.0 - stats["p_success"] if route else 1.0)
            if route and math.isfinite(stats["mean_delivery"]):
                lat_scg.append(stats["mean_delivery"])

        def ci(xs):
            xs = np.asarray(xs)
            return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0

        rows.append(
            {
                "sigma": sigma,
                "viol_det": float(np.mean(viol_det)), "ci_det": ci(viol_det),
                "viol_scg": float(np.mean(viol_scg)), "ci_scg": ci(viol_scg),
                "lat_det": float(np.mean(lat_det)) if lat_det else math.nan,
                "lat_scg": float(np.mean(lat_scg)) if lat_scg else math.nan,
                "R": R, "M": M,
            }
        )
        print(
            f"sigma={sigma:5.1f}  viol(det)={rows[-1]['viol_det']:.3f}"
            f"  viol(scg)={rows[-1]['viol_scg']:.3f}"
        )
    return rows


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    rows = run(quick)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "uc1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5))

    s = [r["sigma"] for r in rows]
    ax1.errorbar(s, [r["viol_det"] for r in rows], yerr=[r["ci_det"] for r in rows],
                 label="deterministic CGR (mean plan)", capsize=2, **STYLE["base1"])
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

    _style.save(fig, str(ROOT / "figures" / "uc1_h1"))
    print(f"done in {time.time() - t_start:.1f}s -> results/uc1.csv, figures/uc1_h1.pdf")


if __name__ == "__main__":
    main()
