"""Runtime of SLA-aware route selection against plan size (reviewer risk R2).

The objection the paper has to answer is that quantile routing over convolutions
does not scale. This measures where the time actually goes, on real CPD plans of
increasing size (the same scenario re-run with larger constellations), and
separates the two costs that scale differently:

  candidate generation   deterministic route searches over the plan. Grows with
                         the plan, and is exactly what a classical CGR router
                         already pays.
  candidate scoring      per-route evaluation under the laws, either by Monte
                         Carlo (M draws) or by numerical convolution. Grows with
                         the number of candidates and the route length, and is
                         independent of the plan size.

Both sit in the offline planning plane. The online forwarding decision is a
table lookup and is not measured here because it does no route search at all.

Unlike every other runner, the numbers here are wall-clock measurements of the
machine rather than properties of the model, so they do not reproduce
bit-for-bit and scripts/verify_results.py exempts the timing columns. The claim
the figure makes is the *shape*: scoring flat in plan size, generation growing.
`--replot` redraws from the committed CSV without re-measuring, which is how the
figure stays consistent with the numbers the paper quotes.

Usage: python experiments/run_runtime.py [--quick] [--replot]
Outputs: results/runtime.csv, figures/runtime.{pdf,png}
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _style import COL_BASE1, COL_BOUND, COL_SCG

from scgraph.analytic import route_quantile
from scgraph.io import load_cpd_plan
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, route_stats
from scgraph.scenarios.uc1_event import EVENT_SITE, GATEWAY, make_instance_cpd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260728
SIGMA = 40.0
SIZES = [6, 12, 24, 48]        # constellation sizes with a generated skeleton
M_SCORE = 2000


def _timed(fn, repeats: int):
    """Median wall time of `fn` over `repeats` runs, in milliseconds."""
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def run(quick: bool = False):
    repeats = 3 if quick else 7
    n_inst = 3 if quick else 8

    rows = []
    for n in SIZES:
        path = ROOT / "scenarios" / f"scale_{n}"
        if not path.exists():
            print(f"  (skipping {n}: no skeleton at {path})")
            continue
        sk = load_cpd_plan(path)

        # Several release times per plan size. The candidate set varies a lot
        # with pass geometry (a rich moment yields many disjoint routes), so a
        # single instance would make the total look arbitrary.
        per = {k: [] for k in
               ("cands", "det", "gen", "mc", "conv", "total")}
        for i in range(n_inst):
            rng = np.random.default_rng(SEED + 1000 * n + i)
            g, p = make_instance_cpd(rng, SIGMA, skeleton=sk)
            src, dst = p["src"], p["dst"]
            size, t0, D = p["size"], p["t0"], p["deadline"]
            mean_plan = g.mean_plan()

            t_det = _timed(lambda: cgr_route(mean_plan, src, dst, size, t0), repeats)
            cands = candidate_routes(g, src, dst, size, t0, rng, n_realizations=40)
            t_gen = _timed(
                lambda: candidate_routes(g, src, dst, size, t0, rng,
                                         n_realizations=40),
                repeats,
            )
            route = cands[0]
            t_mc = _timed(
                lambda: route_stats(route, size, t0, D, M_SCORE, rng,
                                    latent=g.latent),
                repeats,
            )
            try:
                t_conv = _timed(
                    lambda: route_quantile(route, size, t0, 0.95, t_max=D), repeats
                )
            except Exception:                            # pragma: no cover
                t_conv = math.nan

            per["cands"].append(len(cands))
            per["det"].append(t_det)
            per["gen"].append(t_gen)
            per["mc"].append(t_mc)
            per["conv"].append(t_conv)
            per["total"].append(t_gen + len(cands) * t_mc)

        med = {k: statistics.median(v) for k, v in per.items()}
        rows.append(
            {
                "sats": n,
                "contacts": len(sk.contacts),
                "nodes": sk.num_nodes,
                "candidates": med["cands"],
                "t_det_ms": med["det"],
                "t_cand_ms": med["gen"],
                "t_mc_per_route_ms": med["mc"],
                "t_conv_per_route_ms": med["conv"],
                "t_total_ms": med["total"],
                "M": M_SCORE,
                "instances": n_inst,
                "repeats": repeats,
            }
        )
        print(
            f"  sats={n:3d}  contacts={len(sk.contacts):6d}  "
            f"cands(med)={med['cands']:5.1f}  det={med['det']:7.2f}ms  "
            f"gen={med['gen']:8.2f}ms  mc/route={med['mc']:5.2f}ms  "
            f"conv/route={med['conv']:6.2f}ms  total={med['total']:9.2f}ms"
        )
    return rows


def load():
    """Re-read a finished run, so the figure can be redrawn without re-measuring."""
    with open(ROOT / "results" / "runtime.csv") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    t_start = time.time()
    _style.setup()

    if replot:
        rows = load()
        print(f"replotting {len(rows)} rows from results/runtime.csv")
    else:
        print("Route-search runtime vs plan size (CPD skeletons):")
        rows = run(quick)
    if not rows:
        print("no skeletons found; generate them first")
        return

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    if not replot:
        with open(ROOT / "results" / "runtime.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = [r["contacts"] for r in rows]
    ax.plot(x, [r["t_total_ms"] for r in rows],
            label="SLA-aware selection (total)", **{"color": COL_SCG,
                                                    "linestyle": "-", "marker": "o"})
    ax.plot(x, [r["t_cand_ms"] for r in rows],
            label="candidate generation", **{"color": "#7b4fa8",
                                             "linestyle": "-.", "marker": "^"})
    ax.plot(x, [r["t_mc_per_route_ms"] for r in rows],
            label=f"scoring, per route ($M$={M_SCORE})",
            **{"color": COL_BOUND, "linestyle": ":", "marker": "d"})
    ax.plot(x, [r["t_det_ms"] for r in rows],
            label="deterministic CGR (one search)", **{"color": COL_BASE1,
                                                       "linestyle": "--", "marker": "s"})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("contacts in the plan")
    ax.set_ylabel("wall time [ms]")
    ax.legend(loc="upper left", fontsize=6)

    _style.save(fig, str(ROOT / "figures" / "runtime"))
    print(f"done in {time.time() - t_start:.1f}s -> results/runtime.csv, "
          "figures/runtime.pdf")


if __name__ == "__main__":
    main()
