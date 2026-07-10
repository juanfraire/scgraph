"""H2 experiment: provisioned overhead vs fade severity (UC2).

Usage: python experiments/run_uc2.py
Outputs: results/uc2.csv, figures/uc2_h2.{pdf,png}
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _style import STYLE

from scgraph.scenarios.uc2_logistics import provisioning

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260709


def main():
    t_start = time.time()
    _style.setup()
    rng = np.random.default_rng(SEED)

    demand, mean_vol, eps = 4000.0, 1000.0, 0.05
    cvs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for cv in cvs:
        r = provisioning(rng, demand, mean_vol, cv, eps=eps)
        r["cv"] = cv
        rows.append(r)
        print(
            f"cv={cv:.1f}  k_mean={r['k_mean']} (SLA {r['sla_mean']:.2f})"
            f"  k_cc={r['k_cc']} (SLA {r['sla_cc']:.2f})  k_wc={r['k_wc']}"
        )

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "uc2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cv = [r["cv"] for r in rows]
    ax.plot(cv, [r["overhead_wc"] for r in rows],
            label="worst-case plan (meets SLA)", **STYLE["base2"])
    ax.plot(cv, [r["overhead_cc"] for r in rows],
            label=rf"chance-constrained, $\varepsilon$={eps} (meets SLA)", **STYLE["scg"])
    ax.plot(cv, [r["overhead_mean"] for r in rows],
            label="mean-case plan (violates SLA)", **STYLE["base1"])
    ax.set_xlabel("capacity fade severity (coefficient of variation of $V$)")
    ax.set_ylabel("provisioned overhead\n(reserved capacity / demand)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")

    _style.save(fig, str(ROOT / "figures" / "uc2_h2"))
    print(f"done in {time.time() - t_start:.1f}s -> results/uc2.csv, figures/uc2_h2.pdf")


if __name__ == "__main__":
    main()
