"""How timing jitter compounds along a route, isolated (revision, R1).

Reviewer 1: "In deep multi-hop topologies, intermediate hop timing jitters
compound nonlinearly (max(A_{i-1}, T_s) operations over random variables)."
This experiment isolates exactly that mechanism on a chain of k hops and
compares the end-to-end recursion the paper evaluates by Monte Carlo against
the two per-hop shortcuts a router could take instead:

  hop-wise product  P[on time] ~ prod_i P[hop i succeeds when the bundle
                    arrives at its NOMINAL time]. This is what any model that
                    scores contacts one at a time does (a per-contact
                    probability, a slotted Bernoulli flag): it assumes the hops
                    fail independently and that lateness does not propagate.
  quantile sum      Q_95(delivery) ~ nominal + sum_i (per-hop 95% lateness).
                    Quantiles do not compose along the max-plus recursion, so
                    this is the composition the paper declines to make.

Chain: hop i opens at a nominal time with start jitter sigma (TruncNormal,
+-3.5 sigma, the window shifting rigidly as in UC1), lasts 60 s, and carries
the 100-unit bundle in 10 s; consecutive windows leave `slack` seconds of
nominal margin. The deadline is the nominal arrival plus `margin`. Lateness
is absorbed by the next window when it lands inside the slack, and propagates
when it does not: that switch is the nonlinearity.

Usage: python experiments/run_compound.py [--quick]
Outputs: results/compound.csv, figures/compound.{pdf,png}
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
from _style import COL_BASE1, COL_BASE2, COL_SCG

from scgraph.arrival import sample_route_deliveries
from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac, TruncNormal

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261014
SIZE, RATE, DUR = 100.0, 10.0, 60.0
SLACK, MARGIN = 10.0, 30.0


def chain(k: int, sigma: float) -> tuple[list[StochasticContact], float]:
    """k jittered hops; returns (route, nominal arrival)."""
    route, s = [], 0.0
    for i in range(k):
        law = Dirac(s) if sigma <= 0 else TruncNormal(s, sigma, s - 3.5 * sigma, s + 3.5 * sigma)
        route.append(StochasticContact(i, i, i + 1, law, Dirac(DUR), Dirac(RATE)))
        s += SIZE / RATE + SLACK
    nominal = s - SLACK            # last hop's nominal completion
    return route, nominal


def hopwise(route, nominal, deadline, m, rng):
    """Per-hop estimates with the bundle assumed on time at every hop.

    Each hop i is scored on its own, with the bundle ready at its nominal
    arrival: success_i = the transfer fits the (jittered) window, lateness_i =
    completion minus nominal completion. The product of the success
    probabilities (times the last hop's chance of meeting the deadline) is the
    hop-wise P[on time]; the sum of the lateness 95% quantiles, the composed
    quantile. A hop failing more than 5% of the time makes that sum infinite.
    """
    p, q_sum, a_nom = 1.0, 0.0, 0.0
    for i, c in enumerate(route):
        ts = np.asarray(c.start.sample(rng, m), dtype=float)
        tau = np.maximum(a_nom, ts)
        e = tau + SIZE / RATE
        ok = e <= ts + DUR
        e_nom = max(a_nom, c.start.mean()) + SIZE / RATE
        if i == len(route) - 1:
            p *= float(np.mean(ok & (e <= deadline)))
        else:
            p *= float(np.mean(ok))
        late = np.where(ok, e - e_nom, np.inf)
        q_sum += float(np.quantile(late, 0.95)) if np.mean(ok) >= 0.95 else math.inf
        a_nom = e_nom
    return p, nominal + max(0.0, q_sum)


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    rng = np.random.default_rng(SEED)
    m = 20_000 if quick else 200_000
    ks = list(range(1, 11))
    sigmas = [5.0, 10.0, 20.0, 30.0]
    rows = []
    for sigma in sigmas:
        for k in ks:
            route, nominal = chain(k, sigma)
            deadline = nominal + MARGIN
            t = sample_route_deliveries(route, SIZE, 0.0, m, rng)
            p_true = float(np.mean(t <= deadline))
            fin = t[np.isfinite(t)]
            q_true = (float(np.quantile(t[np.isfinite(t)], 0.95 / np.isfinite(t).mean()))
                      if np.isfinite(t).mean() >= 0.95 else math.inf)
            p_hop, q_hop = hopwise(route, nominal, deadline, m, rng)
            rows.append({
                "sigma": sigma, "hops": k, "nominal": nominal, "deadline": deadline,
                "p_true": p_true, "p_hopwise": p_hop, "p_nominal": 1.0,
                "q95_true": q_true, "q95_quantile_sum": q_hop,
                "mean_true": float(np.mean(fin)) if fin.size else math.nan, "M": m,
            })
            print(f"sigma={sigma:4.0f} k={k:2d}  P true={p_true:.4f} hop-wise={p_hop:.4f}"
                  f"  Q95 true={q_true - nominal:7.1f} sum={q_hop - nominal:7.1f} (late, s)")

    with open(ROOT / "results" / "compound.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5))
    marks = {5.0: "o", 10.0: "s", 20.0: "^", 30.0: "D"}
    for sigma in (20.0, 30.0):
        sub = [r for r in rows if r["sigma"] == sigma]
        k = [r["hops"] for r in sub]
        ax1.plot(k, [1 - r["p_true"] for r in sub], color=COL_SCG, marker=marks[sigma],
                 label=rf"end-to-end MC, $\sigma$={sigma:.0f} s")
        ax1.plot(k, [1 - r["p_hopwise"] for r in sub], color=COL_BASE2, linestyle="--",
                 marker=marks[sigma], label=rf"hop-wise product, $\sigma$={sigma:.0f} s")
    ax1.axhline(0.0, color="#6e6e6e", linestyle=":", label="nominal plan (CGR)")
    ax1.set_xlabel("hops with timing jitter")
    ax1.set_ylabel(r"P[miss deadline]")
    ax1.set_title("(a) risk compounds with depth", fontsize=7)
    ax1.legend(fontsize=5.5, loc="upper left")
    for sigma in (5.0, 10.0):
        sub = [r for r in rows if r["sigma"] == sigma]
        k = [r["hops"] for r in sub]
        ax2.plot(k, [r["q95_true"] - r["nominal"] for r in sub], color=COL_SCG,
                 marker=marks[sigma], label=rf"end-to-end MC, $\sigma$={sigma:.0f} s")
        ax2.plot(k, [r["q95_quantile_sum"] - r["nominal"] for r in sub], color=COL_BASE1,
                 linestyle="--", marker=marks[sigma],
                 label=rf"sum of per-hop quantiles, $\sigma$={sigma:.0f} s")
    ax2.set_xlabel("hops with timing jitter")
    ax2.set_ylabel("95% lateness over nominal [s]")
    ax2.set_title("(b) lateness absorbed by slack", fontsize=7)
    ax2.legend(fontsize=5.5, loc="upper left")
    _style.save(fig, str(ROOT / "figures" / "compound"))
    print(f"done in {time.time() - t_start:.1f}s -> results/compound.csv, figures/compound.pdf")


if __name__ == "__main__":
    main()
