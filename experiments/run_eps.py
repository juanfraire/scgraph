"""The epsilon knob (Sec. V-C, eq. 7): reliability bought with delivery time.

The chance-constrained router minimizes latency subject to
P[deliver by deadline] >= 1 - eps, so eps is the paper's tuning parameter.
Sweeping it on UC1 does two things at once:

  left   calibration. The achieved violation rate should sit at or below the
         budget eps, which is what makes the constraint a *certificate* rather
         than a preference.
  right  the price. The delivery time the router pays to honour a tighter
         budget, which is the reliability-latency trade eq. (7) makes explicit.

Run at a deadline margin where the timing-certain backbone is feasible, so a
tight budget always has a safe route to retreat to and the trade is visible
across the whole sweep.

Usage: python experiments/run_eps.py [--quick]
Outputs: results/eps.csv, figures/eps.{pdf,png}
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
from _style import COL_BASE1, COL_BOUND, COL_SCG

from scgraph.routing.quantile import pick_route_chance
from scgraph.scenarios.uc1_event import make_instance_cpd as make_instance

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260728
SIGMA = 40.0
MARGIN = 20.0


def ci(xs):
    xs = np.asarray(xs)
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def run(quick: bool = False):
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    R = 15 if quick else 50
    M = 800 if quick else 2000
    rng = np.random.default_rng(SEED)

    # one instance set, reused at every eps so the sweep is paired
    instances = [make_instance(rng, SIGMA, margin=MARGIN) for _ in range(R)]

    rows = []
    for eps in epsilons:
        viol, lat, feas = [], [], []
        for g, p in instances:
            src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
            route, st = pick_route_chance(
                g, src, dst, size, t0, D, rng, m=M, eps=eps
            )
            if route is None:
                viol.append(1.0)
                feas.append(0.0)
                continue
            viol.append(1.0 - st["p_success"])
            feas.append(1.0 if st.get("feasible") else 0.0)
            if math.isfinite(st["mean_delivery"]):
                lat.append(st["mean_delivery"] - t0)   # end-to-end latency
        rows.append(
            {
                "eps": eps,
                "viol": float(np.mean(viol)), "ci_viol": ci(viol),
                "lat": float(np.mean(lat)) if lat else math.nan,
                "ci_lat": ci(lat) if len(lat) > 1 else 0.0,
                "feasible_frac": float(np.mean(feas)),
                "R": R, "M": M, "sigma": SIGMA, "margin": MARGIN,
            }
        )
        print(
            f"eps={eps:5.3f}  achieved viol={rows[-1]['viol']:.4f}"
            f"  mean delivery={rows[-1]['lat']:7.1f} s"
            f"  feasible={rows[-1]['feasible_frac']:.2f}"
        )
    return rows


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    rows = run(quick)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "eps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    e = [r["eps"] for r in rows]

    ax1.plot(e, e, label=r"budget $\varepsilon$ (constraint)", **{
        "color": COL_BOUND, "linestyle": ":", "marker": ""})
    ax1.errorbar(e, [r["viol"] for r in rows], yerr=[r["ci_viol"] for r in rows],
                 label="achieved violation rate", capsize=2,
                 color=COL_SCG, linestyle="-", marker="o")
    # linear in y: several achieved rates are exactly zero, which a log axis
    # would drop silently
    ax1.set_xscale("log")
    ax1.set_xlabel(r"violation budget $\varepsilon$")
    ax1.set_ylabel("SLA-violation rate")
    ax1.set_ylim(bottom=-0.015)
    ax1.legend(loc="upper left")

    ax2.errorbar(e, [r["lat"] for r in rows], yerr=[r["ci_lat"] for r in rows],
                 capsize=2, color=COL_SCG, linestyle="-", marker="o")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"violation budget $\varepsilon$")
    ax2.set_ylabel("mean delivery time [s]\n(successful deliveries)")

    _style.save(fig, str(ROOT / "figures" / "eps"))
    print(f"done in {time.time() - t_start:.1f}s -> results/eps.csv, figures/eps.pdf")


if __name__ == "__main__":
    main()
