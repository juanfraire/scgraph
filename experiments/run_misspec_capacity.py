"""Misspecified capacity law (revision T6, R2.6 / R3.2): fade variance and pass dependence (UC2).

The three plans of Fig. 6 (mean-case, chance-constrained at eps = 0.05,
worst-case) are sized exactly as in `run_uc2.py`, on the planner's belief:
lognormal fades of coefficient of variation `cv`, independent across passes.
They are then scored on held-out draws of a TRUTH that differs from the belief
in one of two ways (`provisioning_cpd(..., cv_true=, rho_true=)`):

  variance   cv_true / cv in {1, 1.25, 1.5, 2}, passes still independent
  dependence rho_true in {0, 0.1, 0.2, 0.3, 0.5}, cv_true = cv; the passes'
             log-fades share one common factor (a weather system persisting
             across passes), which an independence planner cannot see

Why dependence matters here. The chance-constrained saving over worst-case
sizing is a concentration effect: the sum of k independent fades is much
tighter than k times one fade, so the 5% quantile of the aggregate sits far
above k times the 5% quantile of a pass. A common factor removes that
averaging. At rho = 1 every pass fades identically, P[F * sum(nominal) >= W]
>= 0.95 is exactly the worst-case rule, and the saving is gone.

Dependence-aware planner (written here, `size_chance`). It sizes the reserve
on draws from the same one-factor model the truth uses (`true_fades`) at an
ASSUMED rho_plan. On the diagonal rho_plan = rho_true it shows the overhead
needed to restore the SLA at each rho, and how much of the rho = 0 saving
survives: saving = overhead_wc / overhead_dep, and the surviving fraction
(overhead_wc - overhead_dep) / (overhead_wc - overhead_cc at rho = 0). Off the
diagonal (the plan matrix) it shows what a planner that assumes a moderate
rho_plan buys when the truth is weaker or stronger.

Two instance sets:
  paper      the pass sequence of Fig. 6 (reservation starts at the first
             real pass of the CPD plan), via `provisioning_cpd`
  rotations  all 27 starting passes of the plan, so the conclusion does not
             hinge on one sequence's integer rounding; reports the fraction
             of starts whose plan meets the SLA held-out

Usage: python experiments/run_misspec_capacity.py [--quick]
Outputs: results/robust_capacity.csv, figures/robust_capacity.{pdf,png}
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
from _parallel import pmap
from _style import COL_BASE1, COL_BASE2, COL_BOUND, COL_SCG

from scgraph.distributions import LogNormal
from scgraph.scenarios.uc1_event import GATEWAY, load_skeleton
from scgraph.scenarios.uc2_logistics import cpd_pass_volumes, provisioning_cpd, true_fades
from scgraph.stats import cp_lower

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261014
N_PASSES = 4
EPS = 0.05
K_MAX = 40
CVS = [0.3, 0.5, 0.9]
RATIOS = [1.0, 1.25, 1.5, 2.0]
RHOS = [0.0, 0.1, 0.2, 0.3, 0.5]
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
COL_DEP = "#1baf7a"     # aqua (reference palette slot 3)
SEQ_BLUES = ["#8fbcef", "#2a78d6", "#14457f"]   # cv 0.3 / 0.5 / 0.9 (ordered)
PLAN_RHOS = [0.1, 0.2, 0.3, 0.5]   # assumed rho of the robust planners (rotations)
DELTA = 0.05                       # confidence of the lower-bound sizing rule


# ------------------------------------------------------------ sizing helpers

def plan_counts(seq: np.ndarray, demand: float, cv: float):
    """k_mean and k_wc exactly as `provisioning_cpd` computes them."""
    k_mean = int(np.searchsorted(np.cumsum(seq), demand) + 1)
    q05 = LogNormal.from_mean_cv(1.0, cv).quantile(0.05)
    k_wc = min(int(np.searchsorted(np.cumsum(seq) * q05, demand) + 1), len(seq))
    return k_mean, k_wc


def sla_curve(seq: np.ndarray, demand: float, fades: np.ndarray) -> np.ndarray:
    """(k_max,) array: P[sum of the first k faded passes >= demand], k = 1..k_max."""
    csum = np.cumsum(seq[None, :] * fades, axis=1)
    return (csum >= demand).mean(axis=0)


def hits_curve(seq: np.ndarray, demand: float, fades: np.ndarray) -> np.ndarray:
    csum = np.cumsum(seq[None, :] * fades, axis=1)
    return np.count_nonzero(csum >= demand, axis=0)


def size_chance(seq, demand, fades, eps, k_min=1, delta=None) -> int:
    """Smallest k >= k_min whose SLA on `fades` clears 1 - eps.

    With `delta`, the test uses the one-sided (1 - delta) Clopper-Pearson
    lower bound on the SLA, so a reserve sized on finite draws does not miss
    the target by Monte-Carlo noise alone (the same rule as the lcb route
    admission).
    """
    hits = hits_curve(seq, demand, fades)
    m = fades.shape[0]
    for k in range(k_min, len(seq) + 1):
        h = int(hits[k - 1])
        ok = (h / m >= 1.0 - eps) if delta is None else cp_lower(h, m, delta) >= 1.0 - eps
        if ok:
            return k
    return len(seq)


def overhead(seq, demand, k) -> float:
    return float(seq[: min(k, len(seq))].sum() / demand)


# --------------------------------------------------------------- the tasks

def truth_conditions():
    """(cv_ratio, rho_true) pairs of the two misspecification sweeps."""
    conds = [(r, 0.0) for r in RATIOS] + [(1.0, rho) for rho in RHOS if rho > 0.0]
    return conds


def task_paper(cv, rng, m, m_eval):
    """Fig. 6's pass sequence: the three belief plans, scored on each truth."""
    sk = load_skeleton()
    gs = sk.node(GATEWAY)
    r_plan, r_ev, r_dep, r_dev = rng.spawn(4)
    rows = []
    for ratio, rho in truth_conditions():
        out = provisioning_cpd(r_plan, sk, gs, N_PASSES, cv, eps=EPS, m=m, k_max=K_MAX,
                               rng_eval=r_ev, m_eval=m_eval, cv_true=cv * ratio,
                               rho_true=rho)
        row = {"block": "misspec", "instance": "paper", "cv": cv, "cv_ratio": ratio,
               "cv_true": cv * ratio, "rho_true": rho, "rho_plan": 0.0}
        for key in ("mean", "cc", "wc"):
            row[f"k_{key}"] = out[f"k_{key}"]
            row[f"overhead_{key}"] = out[f"overhead_{key}"]
            row[f"sla_{key}"] = out[f"sla_{key}_ho"]
            row[f"sla_{key}_lcb"] = cp_lower(out["m_eval"] - out[f"fails_{key}_ho"],
                                             out["m_eval"], 0.05)
        row["sla_cc_insample"] = out["sla_cc"]
        row["m_eval"] = out["m_eval"]
        rows.append(row)

    # the dependence-aware / variance-aware planner on the same sequence
    nominal = cpd_pass_volumes(sk, gs)
    seq = np.resize(nominal, K_MAX)
    demand = N_PASSES * float(nominal.mean())
    k_mean, k_wc0 = plan_counts(seq, demand, cv)
    oh_wc0 = overhead(seq, demand, k_wc0)

    # plans sized on (cv, rho_plan): one sizing draw per rho_plan
    plans = {}
    for rho_p in RHO_GRID:
        f = true_fades(r_dep, m, K_MAX, cv, rho_p)
        plans[rho_p] = size_chance(seq, demand, f, EPS, k_min=k_mean, delta=DELTA)
    oh_cc0 = overhead(seq, demand, plans[0.0])
    # truths at cv_true = cv over the rho grid: score every plan on the same draws
    for rho_t in RHO_GRID:
        hits = hits_curve(seq, demand, true_fades(r_dev, m_eval, K_MAX, cv, rho_t))
        for rho_p, k in plans.items():
            h = int(hits[k - 1])
            oh = overhead(seq, demand, k)
            rows.append({
                "block": "dep_matrix", "instance": "paper", "cv": cv, "cv_ratio": 1.0,
                "cv_true": cv, "rho_true": rho_t, "rho_plan": rho_p,
                "k_dep": k, "overhead_dep": oh, "sla_dep": h / m_eval,
                "sla_dep_lcb": cp_lower(h, m_eval, 0.05),
                "k_wc": k_wc0, "overhead_wc": oh_wc0, "overhead_cc": oh_cc0,
                "saving": oh_wc0 / oh,
                "saving_kept": (oh_wc0 - oh) / (oh_wc0 - oh_cc0) if oh_wc0 > oh_cc0 else math.nan,
                "m_eval": m_eval,
            })
    # variance-aware planner: sized on the true cv (independent passes)
    for ratio in RATIOS:
        cv_t = cv * ratio
        k = size_chance(seq, demand, true_fades(r_dep, m, K_MAX, cv_t, 0.0), EPS,
                        k_min=plan_counts(seq, demand, cv_t)[0], delta=DELTA)
        hits = hits_curve(seq, demand, true_fades(r_dev, m_eval, K_MAX, cv_t, 0.0))
        _, k_wc_t = plan_counts(seq, demand, cv_t)
        oh_wc_t, oh = overhead(seq, demand, k_wc_t), overhead(seq, demand, k)
        rows.append({
            "block": "var_aware", "instance": "paper", "cv": cv, "cv_ratio": ratio,
            "cv_true": cv_t, "rho_true": 0.0, "rho_plan": 0.0,
            "k_dep": k, "overhead_dep": oh, "sla_dep": int(hits[k - 1]) / m_eval,
            "sla_dep_lcb": cp_lower(int(hits[k - 1]), m_eval, 0.05),
            "k_wc": k_wc_t, "overhead_wc": oh_wc_t, "saving": oh_wc_t / oh,
            "m_eval": m_eval,
        })
    return rows


def task_rotations(cv, rng, m, m_eval):
    """All starting passes: belief plans, robust plans and the truth-sized plan.

    Planners per starting pass:
      mean, wc   as in Fig. 6 (deterministic counts)
      cc         chance-constrained on the belief (cv, independent passes)
      plan<r>    chance-constrained on (cv, assumed rho_plan = r)
      dep        chance-constrained on the truth itself (cv_true, rho_true):
                 the reserve actually needed, and the saving that survives
    All chance-constrained sizing uses the lower-bound rule at DELTA. Common
    random numbers: one sizing draw per planner and one truth draw per
    condition are shared by every starting pass, which differ only in the
    order of the passes.
    """
    sk = load_skeleton()
    nominal = cpd_pass_volumes(sk, sk.node(GATEWAY))
    demand = N_PASSES * float(nominal.mean())
    n_starts = int(nominal.size)
    seqs = [np.resize(np.roll(nominal, -s), K_MAX) for s in range(n_starts)]
    r_plan, r_ev = rng.spawn(2)

    k_mean = [plan_counts(q, demand, cv)[0] for q in seqs]
    ks = {"mean": k_mean, "wc": [plan_counts(q, demand, cv)[1] for q in seqs]}
    for rho_p in [0.0] + PLAN_RHOS:
        f = true_fades(r_plan, m, K_MAX, cv, rho_p)
        name = "cc" if rho_p == 0.0 else f"plan{rho_p:g}"
        ks[name] = [size_chance(q, demand, f, EPS, k_min=k0, delta=DELTA)
                    for q, k0 in zip(seqs, k_mean)]
        del f

    conds = [(r, 0.0) for r in RATIOS] + [(1.0, rho) for rho in RHO_GRID if rho > 0.0]
    rows = []
    for ratio, rho in conds:
        cv_t = cv * ratio
        f = true_fades(r_plan, m, K_MAX, cv_t, rho)
        ks_c = dict(ks)
        ks_c["dep"] = [size_chance(q, demand, f, EPS, k_min=plan_counts(q, demand, cv_t)[0],
                                   delta=DELTA) for q in seqs]
        del f
        f_truth = true_fades(r_ev, m_eval, K_MAX, cv_t, rho)
        sla = {key: [] for key in ks_c}
        ohs = {key: [] for key in ks_c}
        for i, q in enumerate(seqs):
            hits = hits_curve(q, demand, f_truth)
            for key, kl in ks_c.items():
                sla[key].append(int(hits[kl[i] - 1]) / m_eval)
                ohs[key].append(overhead(q, demand, kl[i]))
        del f_truth
        row = {"block": "rotations", "instance": f"all {n_starts} starts", "cv": cv,
               "cv_ratio": ratio, "cv_true": cv_t, "rho_true": rho, "rho_plan": rho,
               "m_eval": m_eval, "n_starts": n_starts}
        for key in ks_c:
            s_ = np.asarray(sla[key])
            row[f"sla_{key}"] = float(s_.mean())
            row[f"sla_{key}_min"] = float(s_.min())
            row[f"frac_meet_{key}"] = float(np.mean(s_ >= 1.0 - EPS))
            row[f"overhead_{key}"] = float(np.mean(ohs[key]))
        oh_wc, oh_dep, oh_cc = (np.asarray(ohs[k]) for k in ("wc", "dep", "cc"))
        row["saving"] = float(np.mean(oh_wc / oh_dep))
        row["saving_kept"] = float((oh_wc.mean() - oh_dep.mean())
                                   / (oh_wc.mean() - oh_cc.mean()))
        rows.append(row)
    return rows


def one_task(task, rng):
    kind, cv, quick = task
    m = 40_000 if quick else 200_000
    m_eval = 40_000 if quick else 200_000
    if kind == "paper":
        return task_paper(cv, rng, m, m_eval)
    return task_rotations(cv, rng, m // 2, m_eval // 2)


# ------------------------------------------------------------------- figure

def plot(rows, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    target = 1.0 - EPS
    plan_sty = {
        "mean": dict(color=COL_BASE1, marker="s", label="mean-case plan"),
        "cc": dict(color=COL_SCG, marker="o", label=rf"chance-constrained, $\varepsilon={EPS:g}$"),
        "wc": dict(color=COL_BASE2, marker="^", label="worst-case plan"),
    }
    mis = [r for r in rows if r["block"] == "misspec"]

    # (a) fade variance under-estimated
    ax = axes[0, 0]
    for cv, ls in ((0.5, "-"), (0.3, "--")):
        rs = sorted([r for r in mis if r["cv"] == cv and r["rho_true"] == 0.0],
                    key=lambda r: r["cv_ratio"])
        for key, sty in plan_sty.items():
            ax.plot([r["cv_ratio"] for r in rs], [r[f"sla_{key}"] for r in rs],
                    linestyle=ls, **{**sty, "label": sty["label"] if cv == 0.5 else None})
    ax.axhline(target, color=COL_BOUND, linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"fade variance under-estimate: true CV / planned CV")
    ax.set_ylabel("held-out SLA (demand delivered)")
    ax.set_title("(a) under-estimated fade variance", loc="left")
    ax.set_ylim(0.5, 1.01)
    ax.legend(loc="lower left", fontsize=6.3)
    ax.text(0.98, 0.6, "solid: CV = 0.5\ndashed: CV = 0.3", transform=ax.transAxes,
            ha="right", fontsize=6.3, color="#52514e")

    # (b) inter-pass dependence ignored
    ax = axes[0, 1]
    for cv, ls in ((0.5, "-"), (0.3, "--")):
        rs = sorted([r for r in mis if r["cv"] == cv and r["cv_ratio"] == 1.0],
                    key=lambda r: r["rho_true"])
        for key, sty in plan_sty.items():
            ax.plot([r["rho_true"] for r in rs], [r[f"sla_{key}"] for r in rs],
                    linestyle=ls, **{**sty, "label": None})
    ax.axhline(target, color=COL_BOUND, linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"true inter-pass fade correlation $\rho$ (planner assumes 0)")
    ax.set_ylabel("held-out SLA (demand delivered)")
    ax.set_title("(b) dependence ignored by the planner", loc="left")
    ax.set_ylim(0.5, 1.01)

    # (c) saving that survives when the planner sizes with the true rho
    ax = axes[1, 0]
    rot = [r for r in rows if r["block"] == "rotations" and r["cv_ratio"] == 1.0]
    dm = [r for r in rows if r["block"] == "dep_matrix" and r["rho_plan"] == r["rho_true"]]
    for cv, col in zip(CVS, SEQ_BLUES):
        rs = sorted([r for r in rot if r["cv"] == cv], key=lambda r: r["rho_true"])
        ax.plot([r["rho_true"] for r in rs], [r["saving"] for r in rs], color=col,
                marker="o", label=f"CV = {cv:g}")
        rs = sorted([r for r in dm if r["cv"] == cv], key=lambda r: r["rho_true"])
        ax.plot([r["rho_true"] for r in rs], [r["saving"] for r in rs], color=col,
                linestyle="--", linewidth=0.9, marker="")
    ax.axhline(1.0, color=COL_BOUND, linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"inter-pass fade correlation $\rho$ (planner sizes with it)")
    ax.set_ylabel("saving over worst-case\n(worst-case / dependence-aware overhead)")
    ax.set_title("(c) how much of the saving survives", loc="left")
    ax.set_ylim(bottom=0.9)
    ax.legend(loc="upper right", fontsize=6.3)
    ax.text(0.98, 0.62, "solid: mean over all starting passes\ndashed: Fig. 6 pass sequence",
            transform=ax.transAxes, ha="right", fontsize=6.0, color="#52514e")

    # (d) robust planners at CV = 0.5: assumed rho vs true rho, all starting passes
    ax = axes[1, 1]
    rs = sorted([r for r in rot if r["cv"] == 0.5], key=lambda r: r["rho_true"])
    shades = ["#14457f", "#2a78d6", "#5c9be3", "#8fbcef", "#b9d5f5"]
    for key, rho_p, col in zip(["cc"] + [f"plan{x:g}" for x in PLAN_RHOS],
                               [0.0] + PLAN_RHOS, shades):
        oh = float(np.mean([r[f"overhead_{key}"] for r in rs]))
        ax.plot([r["rho_true"] for r in rs], [r[f"frac_meet_{key}"] for r in rs],
                color=col, marker="o",
                label=rf"assumes $\rho={rho_p:g}$ (overhead {oh:.2f})")
    ax.plot([r["rho_true"] for r in rs], [r["frac_meet_wc"] for r in rs], color=COL_BASE2,
            marker="^", linestyle="-.",
            label=f"worst-case (overhead {float(np.mean([r['overhead_wc'] for r in rs])):.2f})")
    ax.set_xlabel(r"true inter-pass fade correlation $\rho$")
    ax.set_ylabel(rf"starting passes meeting SLA $\geq {target:g}$")
    ax.set_title(r"(d) CV = 0.5, robust planners (legend below)", loc="left")
    ax.set_ylim(-0.02, 1.02)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, [f"(d) {x}" for x in lab], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), fontsize=6.5)

    fig.tight_layout(pad=0.4, rect=(0, 0.075, 1, 1))
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")


# --------------------------------------------------------------------- main

def fieldnames(rows):
    """Union of the row keys, in order of first appearance."""
    out: list = []
    for r in rows:
        out.extend(k for k in r if k not in out)
    return out


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    tasks = [(kind, cv, quick) for kind in ("paper", "rotations") for cv in CVS]
    rows = [r for out in pmap(one_task, tasks, seed=SEED) for r in out]

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "robust_capacity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames(rows), restval="")
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        if r["block"] == "misspec":
            print(f"[misspec] cv={r['cv']:.1f} cv_true/cv={r['cv_ratio']:.2f} "
                  f"rho={r['rho_true']:.1f}  SLA mean/cc/wc = {r['sla_mean']:.4f} / "
                  f"{r['sla_cc']:.4f} (lcb {r['sla_cc_lcb']:.4f}) / {r['sla_wc']:.4f}"
                  f"  k={r['k_mean']}/{r['k_cc']}/{r['k_wc']}")
        elif r["block"] == "dep_matrix" and r["rho_plan"] == r["rho_true"]:
            print(f"[dep-aware] cv={r['cv']:.1f} rho={r['rho_true']:.1f}  k={r['k_dep']}"
                  f" overhead={r['overhead_dep']:.3f} (wc {r['overhead_wc']:.3f}, "
                  f"cc@0 {r['overhead_cc']:.3f})  SLA={r['sla_dep']:.4f}"
                  f"  saving={r['saving']:.2f}x kept={r['saving_kept']:.2f}")
        elif r["block"] == "var_aware":
            print(f"[var-aware] cv={r['cv']:.1f} cv_true={r['cv_true']:.2f}  k={r['k_dep']}"
                  f" overhead={r['overhead_dep']:.3f} (wc {r['overhead_wc']:.3f})"
                  f"  SLA={r['sla_dep']:.4f}")
        elif r["block"] == "rotations":
            print(f"[rotations] cv={r['cv']:.1f} cv_true/cv={r['cv_ratio']:.2f} "
                  f"rho={r['rho_true']:.1f}  meet mean/cc/wc/dep = "
                  f"{r['frac_meet_mean']:.2f}/{r['frac_meet_cc']:.2f}/"
                  f"{r['frac_meet_wc']:.2f}/{r['frac_meet_dep']:.2f}"
                  f"  SLA cc {r['sla_cc']:.4f} (min {r['sla_cc_min']:.4f})"
                  f"  overhead cc/dep/wc {r['overhead_cc']:.2f}/{r['overhead_dep']:.2f}/"
                  f"{r['overhead_wc']:.2f}  saving {r['saving']:.2f}x kept {r['saving_kept']:.2f}")

    plot(rows, str(ROOT / "figures" / "robust_capacity"))
    print(f"done in {time.time() - t_start:.1f}s -> results/robust_capacity.csv, "
          f"figures/robust_capacity.pdf")


if __name__ == "__main__":
    main()
