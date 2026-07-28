"""Monte Carlo vs analytic convolution: which evaluator is cheaper, and when.

Section V-F of the paper makes a quantitative claim that the rest of the
runtime study does not produce, because run_runtime.py sweeps *plan size* at a
fixed M=2000 while this claim is about sweeping *M*:

    "scoring one route takes 0.75 ms at M=2000 and grows linearly to 519 ms at
     M=2e6, whereas a 4000-point convolution costs about 170 ms whatever eps
     is; the two cross near M ~ 5e5."

This script is that measurement. It exists so no number in the paper is left
without an artefact behind it.

The point being made is deliberately against intuition, and the paper says so:
the analytic path is the SLOW one at every epsilon this work uses. Monte Carlo
is O(M) and starts three orders of magnitude cheaper; the convolution pays a
fixed grid cost regardless of eps. They only cross once the tail of interest is
far beyond eps=1e-3, which is the regime an operator certifying extreme tails
would be in and this paper is not. Do not "optimise" the convolution on the
assumption that it ought to be the fast path.

Wall-clock numbers are properties of the machine, so they are not expected to
reproduce bit-for-bit; the claim is the shape (MC linear in M, convolution
flat) and the location of the crossover within a factor of about two.

Usage: python experiments/run_mc_scaling.py [--quick] [--replot]
Outputs: results/mc_scaling.csv, figures/mc_scaling.{pdf,png}
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _style import COL_BASE1, COL_BOUND, COL_SCG

from scgraph.analytic import route_quantile
from scgraph.routing.quantile import candidate_routes, route_stats
from scgraph.scenarios.uc1_event import make_instance_cpd

ROOT = Path(__file__).resolve().parents[1]
# Same stream this instance gets inside run_runtime.py's 12-satellite case, so
# the route measured here is the route measured there. This script writes its
# own CSV and touches no other runner's RNG.
SEED = 20260728 + 1000 * 12
SIGMA = 40.0
GRID = 4000        # convolution grid points, matching analytic.route_quantile


def _timed(fn, repeats: int) -> float:
    """Median wall time of `fn` over `repeats` runs, in milliseconds."""
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def run(quick: bool = False):
    ms = [2_000, 20_000, 200_000] if quick else [
        2_000, 5_000, 20_000, 50_000, 200_000, 500_000, 2_000_000
    ]
    repeats = 3 if quick else 5

    rng = np.random.default_rng(SEED)
    g, p = make_instance_cpd(rng, SIGMA)
    src, dst = p["src"], p["dst"]
    size, t0, D = p["size"], p["t0"], p["deadline"]
    route = candidate_routes(g, src, dst, size, t0, rng, n_realizations=40)[0]

    # The convolution cost is independent of eps and of M, so it is measured
    # once and carried across the sweep as a horizontal reference.
    t_conv = _timed(lambda: route_quantile(route, size, t0, 0.95, t_max=D), repeats)
    print(f"convolution ({GRID}-point grid): {t_conv:.2f} ms  (independent of M and eps)")

    rows = []
    for m in ms:
        t_mc = _timed(
            lambda m=m: route_stats(route, size, t0, D, m, rng, latent=g.latent),
            repeats,
        )
        rows.append({
            "M": m,
            "t_mc_ms": t_mc,
            "t_conv_ms": t_conv,
            "grid": GRID,
            "repeats": repeats,
            "hops": len(route),
        })
        cheaper = "MC" if t_mc < t_conv else "convolution"
        print(f"  M={m:>9,}  MC {t_mc:8.2f} ms   ({cheaper} cheaper)")

    # Crossover by linear interpolation on the O(M) fit, reported rather than
    # asserted: it is the quantity Sec. V-F puts at ~5e5.
    per_draw = rows[-1]["t_mc_ms"] / rows[-1]["M"]
    print(f"crossover at M ~ {t_conv / per_draw:,.0f} draws")
    return rows


def load():
    """Re-read a finished run, so the figure can be redrawn without re-measuring."""
    with open(ROOT / "results" / "mc_scaling.csv") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    t_start = time.time()
    _style.setup()

    if replot:
        rows = load()
        print(f"replotting {len(rows)} rows from results/mc_scaling.csv")
    else:
        print("MC vs convolution, scoring one route (Sec. V-F):")
        rows = run(quick)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    if not replot:
        with open(ROOT / "results" / "mc_scaling.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = [r["M"] for r in rows]
    ax.plot(x, [r["t_mc_ms"] for r in rows], label="Monte Carlo ($O(M)$)",
            color=COL_SCG, linestyle="-", marker="o")
    ax.axhline(rows[0]["t_conv_ms"], color=COL_BASE1, linestyle="--", linewidth=1.2)
    ax.annotate(
        f"convolution, {GRID}-point grid",
        xy=(x[0], rows[0]["t_conv_ms"]), xytext=(0, 3), textcoords="offset points",
        ha="left", va="bottom", fontsize=6, color=COL_BASE1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Monte-Carlo draws $M$ per route")
    ax.set_ylabel("wall time [ms]")
    ax.legend(loc="upper left", fontsize=6)

    _style.save(fig, str(ROOT / "figures" / "mc_scaling"))
    print(f"done in {time.time() - t_start:.1f}s -> results/mc_scaling.csv, "
          "figures/mc_scaling.pdf")


if __name__ == "__main__":
    main()
