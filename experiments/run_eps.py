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

Revision. The submitted figure scored the admitted route on the draws that
admitted it, so "achieved <= budget" held by construction. Two admission rules
are now compared, and every admitted route is scored on an independent stream
(SEED + 2, M_EVAL draws):

  point  the submitted rule, P_hat >= 1 - eps on M = 2000 draws. Selection is
         bit-identical to the submitted run.
  lcb    the Clopper-Pearson lower bound, Bonferroni-corrected over the
         candidate set at delta = 0.05, must clear 1 - eps; the draw count
         per candidate is sized from eps (stats.admission_draws). Runs on its
         own stream (SEED + 3).

Per (eps, rule) the CSV reports the held-out mean violation, the fraction of
instances whose admitted route measures above eps, the fraction with a
statistically significant breach (exact 95% lower bound on its violation above
eps), the pooled exact upper bound, and the delivery time paid.

Usage: python experiments/run_eps.py [--quick] [--replot]
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

from scgraph.routing.quantile import candidate_routes, pick_route_chance
from scgraph.scenarios.uc1_event import make_instance_cpd as make_instance
from scgraph.stats import admission_draws, cp_lower, cp_upper
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260728
SIGMA = 40.0
MARGIN = 20.0
M_EVAL = 20_000
DELTA = 0.05


def ci(xs):
    xs = np.asarray(xs)
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def run(quick: bool = False):
    epsilons = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    R = 15 if quick else 50
    M = 800 if quick else 2000
    m_eval = 4000 if quick else M_EVAL
    rng = np.random.default_rng(SEED)
    rng_l = np.random.default_rng(SEED + 3)   # lcb-admission stream
    rng_e = np.random.default_rng(SEED + 2)   # held-out evaluation stream

    # one instance set, reused at every eps so the sweep is paired
    instances = [make_instance(rng, SIGMA, margin=MARGIN) for _ in range(R)]

    rows = []
    for eps in epsilons:
        acc = {rule: {"viol": [], "lat": [], "feas": [], "over": 0, "breach": 0,
                      "fails": 0, "n": 0, "insample": [], "m": []}
               for rule in ("point", "lcb")}
        for g, p in instances:
            src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]

            # the submitted rule, selection bit-identical to v1
            route_p, st_p = pick_route_chance(
                g, src, dst, size, t0, D, rng, m=M, eps=eps, admission="point"
            )
            # the revised rule: exact lower bound, Bonferroni over candidates
            cands = candidate_routes(g, src, dst, size, t0, rng_l, 40)
            m_l = M if quick else admission_draws(eps, len(cands), DELTA)
            route_l, st_l = pick_route_chance(
                g, src, dst, size, t0, D, rng_l, m=m_l, eps=eps,
                admission="lcb", delta=DELTA, cands=cands,
            )

            for rule, route, st, m_used in (("point", route_p, st_p, M),
                                            ("lcb", route_l, st_l, m_l)):
                a = acc[rule]
                a["m"].append(m_used)
                if route is None:
                    a["viol"].append(1.0)
                    a["feas"].append(0.0)
                    continue
                a["feas"].append(1.0 if st.get("feasible") else 0.0)
                a["insample"].append(1.0 - st["p_success"])
                ev = evaluate_route(route, size, t0, D, m_eval, rng_e)
                a["viol"].append(ev["viol"])
                a["fails"] += ev["fails"]
                a["n"] += ev["n"]
                a["over"] += int(ev["viol"] > eps)
                a["breach"] += int(cp_lower(ev["fails"], ev["n"], 0.025) > eps)
                if math.isfinite(ev["mean_delivery"]):
                    a["lat"].append(ev["mean_delivery"] - t0)   # end-to-end latency

        row = {"eps": eps}
        for rule, a in acc.items():
            row.update({
                f"viol_{rule}": float(np.mean(a["viol"])),
                f"ci_viol_{rule}": ci(a["viol"]),
                f"max_viol_{rule}": float(np.max(a["viol"])),
                f"frac_over_{rule}": a["over"] / R,
                f"frac_breach_{rule}": a["breach"] / R,
                f"viol_ub_{rule}": cp_upper(a["fails"], a["n"], 0.05),
                f"insample_{rule}": float(np.mean(a["insample"])) if a["insample"] else math.nan,
                f"lat_{rule}": float(np.mean(a["lat"])) if a["lat"] else math.nan,
                f"ci_lat_{rule}": ci(a["lat"]) if len(a["lat"]) > 1 else 0.0,
                f"feasible_frac_{rule}": float(np.mean(a["feas"])),
                f"m_{rule}": int(np.median(a["m"])),
            })
        row.update({"R": R, "M_eval": m_eval, "sigma": SIGMA, "margin": MARGIN, "delta": DELTA})
        rows.append(row)
        print(
            f"eps={eps:5.3f} | point: viol={row['viol_point']:.4f} "
            f"over={row['frac_over_point']:.2f} breach={row['frac_breach_point']:.2f} "
            f"lat={row['lat_point']:6.1f} | lcb: viol={row['viol_lcb']:.4f} "
            f"over={row['frac_over_lcb']:.2f} breach={row['frac_breach_lcb']:.2f} "
            f"lat={row['lat_lcb']:6.1f} m={row['m_lcb']}"
        )
    return rows


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    if "--replot" in sys.argv:
        with open(ROOT / "results" / "eps.csv") as f:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    else:
        rows = run(quick)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    if "--replot" not in sys.argv:
        with open(ROOT / "results" / "eps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    e = [r["eps"] for r in rows]

    ax1.plot(e, e, label=r"budget $\varepsilon$ (constraint)", **{
        "color": COL_BOUND, "linestyle": ":", "marker": ""})
    # held-out violation spans five decades, so both axes are logarithmic. A
    # measured zero cannot sit on a log axis: it is drawn at its exact one-sided
    # 95% upper bound (pooled over instances) with an open downward marker, which
    # is what the data actually supports.
    for rule, col, ls, mk, lab in (
        ("lcb", COL_SCG, "-", "o", r"admission on exact lower bound ($\delta{=}0.05$)"),
        ("point", COL_BASE1, "--", "s", "admission on point estimate"),
    ):
        y = [r[f"viol_{rule}"] if r[f"viol_{rule}"] > 0 else r[f"viol_ub_{rule}"] for r in rows]
        ax1.plot(e, y, color=col, linestyle=ls, label=lab)
        pos = [(x, r[f"viol_{rule}"], r[f"ci_viol_{rule}"]) for x, r in zip(e, rows) if r[f"viol_{rule}"] > 0]
        if pos:
            ax1.errorbar([a for a, _, _ in pos], [b for _, b, _ in pos],
                         yerr=[c for _, _, c in pos], color=col, linestyle="none",
                         marker=mk, capsize=2)
        zer = [(x, r[f"viol_ub_{rule}"]) for x, r in zip(e, rows) if r[f"viol_{rule}"] <= 0]
        if zer:
            ax1.plot([a for a, _ in zer], [b for _, b in zer], color=col, linestyle="none",
                     marker="v", markerfacecolor="white")
    ax1.plot([], [], color="#444444", linestyle="none", marker="v", markerfacecolor="white",
             label="zero measured: 95% upper bound")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-6, 1.0)
    ax1.set_xlabel(r"violation budget $\varepsilon$")
    ax1.set_ylabel("held-out SLA-violation rate")
    ax1.legend(loc="upper left")

    ax2.errorbar(e, [r["lat_lcb"] for r in rows], yerr=[r["ci_lat_lcb"] for r in rows],
                 capsize=2, color=COL_SCG, linestyle="-", marker="o",
                 label="lower-bound admission")
    ax2.errorbar(e, [r["lat_point"] for r in rows], yerr=[r["ci_lat_point"] for r in rows],
                 capsize=2, color=COL_BASE1, linestyle="--", marker="s",
                 label="point-estimate admission")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"violation budget $\varepsilon$")
    ax2.set_ylabel("mean delivery time [s]\n(delivered bundles)")
    ax2.legend(loc="lower left")

    _style.save(fig, str(ROOT / "figures" / "eps"))
    print(f"done in {time.time() - t_start:.1f}s -> results/eps.csv, figures/eps.pdf")


if __name__ == "__main__":
    main()
