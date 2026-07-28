"""H3 experiment: delivery ratio vs collapse severity under correlated failure (UC3).

Policies at copy budget N=2 (equal overhead), plus references:
  stale    - single nominal-best route from the pre-collapse plan
  naive-2  - the 2 nominally fastest routes (probability/correlation blind)
  bern-2   - Bernoulli-TEG ablation: knows the marginal existence probabilities,
             assumes contacts fail independently (eq. 12's belief)
  scg-2    - greedy joint-success selection on the stochastic contact graph
  epidemic - all candidate routes (unbounded overhead; upper reference)

The bern-2 / scg-2 gap separates knowing *how likely* each contact is from
knowing *which contacts fail together*: only the second is what eq. (1) adds
over the discrete on/off model. The ablation uses a dedicated RNG stream, so
the other four series are bit-identical to the runs without it.

Usage: python experiments/run_uc3.py [--quick]
Outputs: results/uc3.csv, figures/uc3_h3.{pdf,png}
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

from scgraph.arrival import delivery_time
from scgraph.routing.bernoulli import bernoulli_select
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes
from scgraph.routing.replication import greedy_select, joint_success
from scgraph.scenarios.uc3_emergency import make_instance

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260709
BERN_SLOT = 30.0   # ~13 slots over the UC3 horizon; see figures/subsumption.pdf


def nominal_arrival(route, size, t0):
    return delivery_time([c.mean_det() for c in route], size, t0)


def run(quick: bool = False):
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    R = 15 if quick else 40
    M = 2000 if quick else 4000
    N = 2
    rng = np.random.default_rng(SEED)
    rng_b = np.random.default_rng(SEED + 1)   # ablation-only stream

    rows = []
    for rho in severities:
        acc = {"stale": [], "naive": [], "bern": [], "scg": [], "epidemic": []}
        for _ in range(R):
            g, p = make_instance(rng, rho)
            src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]

            cands = candidate_routes(g, src, dst, size, t0, rng, n_realizations=30)
            if not cands:
                for k in acc:
                    acc[k].append(0.0)
                continue

            # stale central plan: nominal-best route, executed blindly
            stale_route, _ = cgr_route(g.mean_plan(include_threshold=0.0), src, dst, size, t0)
            stale = [[g.by_id(c.cid) for c in stale_route]] if stale_route else []

            # naive-N: N nominally fastest candidates
            by_nominal = sorted(cands, key=lambda r: nominal_arrival(r, size, t0))
            naive = by_nominal[:N]

            # scg-N: correlation-aware greedy (eq. 13)
            scg = greedy_select(cands, N, size, t0, D, M // 2, rng, latent=g.latent)

            for key, routes in (
                ("stale", stale), ("naive", naive), ("scg", scg), ("epidemic", cands)
            ):
                acc[key].append(
                    joint_success(routes, size, t0, D, M, rng, latent=g.latent)
                )

            # ablation: plan under the independent-Bernoulli belief, then score
            # under the true (correlated) laws. Scored on rng_b so the four
            # series above are untouched.
            bern = bernoulli_select(
                g, src, dst, size, t0, D, BERN_SLOT, N, rng_b, m=M
            )
            acc["bern"].append(
                joint_success(bern, size, t0, D, M, rng_b, latent=g.latent)
            )

        def ci(xs):
            xs = np.asarray(xs)
            return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0

        row = {"severity": rho, "R": R, "M": M, "N": N}
        for k in acc:
            row[k] = float(np.mean(acc[k]))
            row[f"ci_{k}"] = ci(acc[k])
        rows.append(row)
        print(
            f"rho={rho:.1f}  stale={row['stale']:.3f}  naive={row['naive']:.3f}"
            f"  bern={row['bern']:.3f}  scg={row['scg']:.3f}"
            f"  epidemic={row['epidemic']:.3f}"
        )
    return rows


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    rows = run(quick)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "uc3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = [r["severity"] for r in rows]
    # Labels carry only what distinguishes the policies; the copy budget they
    # share (N=2 for every bounded policy) is stated once in the caption. Long
    # labels used to push the legend across the curves.
    ax.plot(x, [r["epidemic"] for r in rows], label="epidemic (unbounded)",
            **STYLE["bound"])
    ax.errorbar(x, [r["scg"] for r in rows], yerr=[r["ci_scg"] for r in rows],
                label="SCG, correlation-aware", capsize=2,
                **STYLE["scg"])
    ax.errorbar(x, [r["bern"] for r in rows], yerr=[r["ci_bern"] for r in rows],
                label="Bernoulli TEG, independent",
                capsize=2,
                color="#7b4fa8", linestyle="-.", marker="^")
    ax.errorbar(x, [r["naive"] for r in rows], yerr=[r["ci_naive"] for r in rows],
                label="naive, nominal-fastest", capsize=2,
                **STYLE["base2"])
    ax.errorbar(x, [r["stale"] for r in rows], yerr=[r["ci_stale"] for r in rows],
                label="stale plan (1 copy)", capsize=2, **STYLE["base1"])
    ax.set_xlabel(r"collapse severity $\rho$ (per-region collapse probability)")
    ax.set_ylabel("delivery ratio by deadline")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", handlelength=1.8, borderaxespad=0.3,
              labelspacing=0.35)

    _style.save(fig, str(ROOT / "figures" / "uc3_h3"))
    print(f"done in {time.time() - t_start:.1f}s -> results/uc3.csv, figures/uc3_h3.pdf")


if __name__ == "__main__":
    main()
