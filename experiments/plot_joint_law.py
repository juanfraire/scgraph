"""A coupled stochastic contact, drawn from its joint law (revision, R2 minor).

Reviewer 2: "Figure 1 plots three separate densities and therefore weakens the
joint-law message. Please show a coupled example." This draws one UAV hand-off
window of the UC4 annotation (`uc4_continuum.uav_law`, sigma = 40 s) twice:
from its joint law (late openings come with shorter, weaker windows) and from
its independent twin (identical marginals, no coupling). Each point is one
realization (T_s, T_e), coloured by the deliverable volume R * (T_e - T_s).
Each title gives P[the window carries a 100-unit bundle offered 30 s after
the nominal start], the event a router actually cares about: the marginals are
identical across the two panels, and the probability of that event is not.

Usage: python experiments/plot_joint_law.py
Outputs: figures/joint_law.{pdf,png}, results/joint_law.csv (summary numbers)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style

from scgraph.contact import StochasticContact
from scgraph.scenarios.uc4_continuum import uav_law

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
NOMINAL_START, NOMINAL_DUR, SIGMA = 0.0, 70.0, 40.0
OFFER, SIZE = 30.0, 100.0          # bundle offered 30 s after the nominal start


def draws(couple: bool, m: int, rng):
    st, du, ra, co = uav_law(NOMINAL_START, NOMINAL_DUR, SIGMA, couple=couple)
    c = StochasticContact(0, 0, 1, st, du, ra, corr=co)
    _, ts, te, rate = c.sample_attrs(rng, m)
    return ts, te, rate


def carries(ts, te, rate):
    tau = np.maximum(OFFER, ts)
    return tau + SIZE / rate <= te


def main():
    _style.setup()
    rng = np.random.default_rng(SEED)
    m_plot, m_stat = 1500, 400_000
    import matplotlib.pyplot as plt

    # one column of the paper at the common print scale (about 0.85): 4 in wide
    fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.5), sharex=True, sharey=True,
                             layout="constrained")
    rows = []
    vmax = None
    for ax, couple, title in ((axes[0], True, "joint law"),
                              (axes[1], False, "independent twin")):
        ts, te, rate = draws(couple, m_stat, rng)
        ok = carries(ts, te, rate)
        rows.append({
            "law": "joint" if couple else "independent",
            "p_carries": float(ok.mean()),
            "corr_start_dur": float(np.corrcoef(ts, te - ts)[0, 1]),
            "corr_start_rate": float(np.corrcoef(ts, rate)[0, 1]),
            "sd_end": float(np.std(te)),
            "mean_volume": float(np.mean(rate * (te - ts))),
        })
        ts, te, rate = ts[:m_plot], te[:m_plot], rate[:m_plot]
        vol = rate * (te - ts)
        vmax = vmax or float(np.quantile(vol, 0.98))
        sc = ax.scatter(ts, te, c=vol, s=3, cmap="viridis", vmin=0, vmax=vmax,
                        linewidths=0, alpha=0.8)
        ax.plot([-150, 200], [-150, 200], color="#999999", lw=0.6, ls=":")
        ax.set_title(f"{title}\nP[carries] = {rows[-1]['p_carries']:.3f}", fontsize=7)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-80, 230)
        ax.set_xticks([-100, 0, 100])
    axes[0].set_ylabel(r"window end $T_e$ [s]")
    fig.supxlabel(r"window start $T_s$ [s from nominal]", fontsize=8)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02)
    cb.set_ticks([0, 500, 1000, 1500])
    cb.set_label(r"deliverable volume $R\,(T_e-T_s)$", fontsize=7)
    fig.savefig(str(ROOT / "figures" / "joint_law.pdf"), bbox_inches="tight")
    fig.savefig(str(ROOT / "figures" / "joint_law.png"), bbox_inches="tight", dpi=300)

    with open(ROOT / "results" / "joint_law.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
