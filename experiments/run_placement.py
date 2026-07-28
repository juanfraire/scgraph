"""Placement probe (Sec. VI-B): does where the planning plane runs change the SLA?

Three fixed, trivial placements (routing/placement.py) are executed against the
same drawn realizations, on UC1 as the deadline tightens and on UC3 as the
collapse deepens:

  ground  global view, full compute, single uplinked route, no contingency
  proxy   global view, reduced compute, ranked (node, time-bucket) table
  dist    local view only, no precomputation, greedy earliest arrival

The headline is the spread between them, not an optimizer.

Usage: python experiments/run_placement.py [--quick]
Outputs: results/placement_uc1.csv, results/placement_uc3.csv,
         figures/placement.{pdf,png}
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

from scgraph.routing.placement import PLACEMENTS, placement_delivery
from scgraph.scenarios.uc1_event import make_instance_cpd as uc1_instance
from scgraph.scenarios.uc3_emergency import make_instance as uc3_instance

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260728
UC1_SIGMA = 40.0

# entity -> style, fixed across the figure
# ground is drawn with a wider line and larger marker because it coincides
# exactly with proxy wherever contingency never fires; it would otherwise be
# hidden underneath and read as missing data
PSTYLE = {
    "ground": dict(color=COL_BASE1, linestyle="--", marker="s",
                   markersize=5.5, linewidth=2.4, zorder=1),
    "proxy": dict(color=COL_SCG, linestyle="-", marker="o", zorder=3),
    "dist": dict(color="#7b4fa8", linestyle="-.", marker="^", zorder=2),
}
PLABEL = {
    "ground": "ground-central (no contingency)",
    "proxy": "sky proxy (ranked table)",
    "dist": "fully distributed (local, greedy)",
}


def ci(xs):
    xs = np.asarray(xs)
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def sweep(make, xs, xname, R, M, rng):
    rows = []
    for x in xs:
        acc = {k: [] for k in PLACEMENTS}
        for _ in range(R):
            g, p = make(rng, x)
            d = placement_delivery(
                g, p["src"], p["dst"], p["size"], p["t0"], p["deadline"],
                rng, m=M, plan_m=max(300, M // 2),
            )
            for k, v in d.items():
                acc[k].append(v)
        row = {xname: x, "R": R, "M": M}
        for k in PLACEMENTS:
            row[k] = float(np.mean(acc[k]))
            row[f"ci_{k}"] = ci(acc[k])
        spread = max(row[k] for k in PLACEMENTS) - min(row[k] for k in PLACEMENTS)
        row["spread"] = spread
        rows.append(row)
        print(
            f"  {xname}={x:6.2f}  "
            + "  ".join(f"{k}={row[k]:.3f}" for k in PLACEMENTS)
            + f"   spread={spread:.3f}"
        )
    return rows


def run(quick: bool = False):
    R = 10 if quick else 30
    M = 400 if quick else 1200
    rng = np.random.default_rng(SEED)

    print(f"UC1 (sigma={UC1_SIGMA:.0f} s), deadline margin sweep:")
    margins = [60.0, 40.0, 20.0, 0.0, -20.0, -40.0]
    uc1 = sweep(
        lambda r, x: uc1_instance(r, UC1_SIGMA, margin=x), margins, "margin", R, M, rng
    )

    print("UC3, collapse severity sweep:")
    sev = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    uc3 = sweep(lambda r, x: uc3_instance(r, x), sev, "severity", R, M, rng)
    return uc1, uc3


def load(name):
    """Re-read a finished sweep, so the figure can be restyled without re-running."""
    with open(ROOT / "results" / f"{name}.csv") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    t_start = time.time()
    _style.setup()

    if replot:
        uc1, uc3 = load("placement_uc1"), load("placement_uc3")
    else:
        uc1, uc3 = run(quick)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    if not replot:
        for name, rows in (("placement_uc1", uc1), ("placement_uc3", uc3)):
            with open(ROOT / "results" / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))

    x1 = [r["margin"] for r in uc1]
    for k in PLACEMENTS:
        ax1.errorbar(x1, [r[k] for r in uc1], yerr=[r[f"ci_{k}"] for r in uc1],
                     label=PLABEL[k], capsize=2, **PSTYLE[k])
    ax1.set_xlabel("deadline margin over the certain backbone [s]")
    ax1.set_ylabel("delivery ratio by deadline")
    ax1.invert_xaxis()
    ax1.set_title(rf"UC1, $\sigma={UC1_SIGMA:.0f}$ s (tighter $\rightarrow$)", fontsize=8)
    ax1.legend(loc="lower left")

    x2 = [r["severity"] for r in uc3]
    for k in PLACEMENTS:
        ax2.errorbar(x2, [r[k] for r in uc3], yerr=[r[f"ci_{k}"] for r in uc3],
                     label=PLABEL[k], capsize=2, **PSTYLE[k])
    ax2.set_xlabel(r"collapse severity $\rho$")
    ax2.set_ylabel("delivery ratio by deadline")
    ax2.set_title("UC3", fontsize=8)

    _style.save(fig, str(ROOT / "figures" / "placement"))
    print(
        f"done in {time.time() - t_start:.1f}s -> results/placement_*.csv, "
        "figures/placement.pdf"
    )


if __name__ == "__main__":
    main()
