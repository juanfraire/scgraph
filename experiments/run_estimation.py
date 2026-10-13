"""Learning the timing law from observations (revision T7, R2.10), UC1.

The paper describes learning the laws as a drop-in extension. This experiment
measures what it costs. The truth is the UC1 instance itself: every air
contact's start jitters as TruncNormal(nominal, sigma_true) truncated at
+-3.5 sigma_true. The planner does not know sigma_true. It observes n i.i.d.
air-window start offsets (realised start minus nominal start, pooled over air
contacts, as an operator would log them from past windows), fits the mean
offset and the standard deviation, builds its belief graph from the fit
(TruncNormal(nominal + mu_hat, sigma_hat), truncated at +-3.5 sigma_hat), and
routes with the chance-constrained router (`pick_route_chance`, eps = 0.05,
lcb admission at delta = 0.05) on candidates generated from that belief. The
chosen route is scored on held-out draws of the truth.

Two fits per training set:
  plugin  mu_hat = sample mean, sigma_hat = sample standard deviation
  ucb     the same mean, sigma_hat replaced by the one-sided 90% upper
          confidence bound from the chi-square law of the sample variance,
          sigma_ucb = s * sqrt((n - 1) / chi2_{0.10, n-1})
and the oracle, which plans on the true law (sigma_true, zero offset).

Training sets are nested: one stream of 200 offsets per instance, the fit at
n uses its first n values, so the n-sweep is paired within an instance (an
operator accumulating observations), and the training sets are independent
across instances. n in {3, 5, 10, 20, 50, 200}.

Two truths. sigma_true = 40 s is the main case. There, at eps = 0.05, the
oracle never admits an air route (no air route of a UC1 instance has a true
violation clearly below 5%, cf. results/eps.csv), so every air admission is a
learning error and the breach rate measures it directly. sigma_true = 20 s is
added because there the oracle does admit air routes, and the experiment then
also shows what the conservative fit costs in delivery time.

Usage: python experiments/run_estimation.py [--quick]
Outputs: results/estimation.csv, figures/estimation.{pdf,png}
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap
from _style import COL_BASE1, COL_BOUND, COL_SCG

from scgraph.distributions import TruncNormal
from scgraph.routing.quantile import candidate_routes, pick_route_chance
from scgraph.scenarios.uc1_event import make_instance_cpd as make_instance
from scgraph.stats import admission_draws, cp_lower, cp_upper
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261016
SIGMAS_TRUE = [40.0, 20.0]
NS = [3, 5, 10, 20, 50, 200]
EPS = 0.05
DELTA = 0.05
UCB_LEVEL = 0.90
MARGIN = 40.0
TRUNC_W = 3.5
M_EVAL = 20_000
COL_UCB = "#1baf7a"    # aqua (reference palette slot 3)

FITS = ["plugin", "ucb"]
LABEL = {"plugin": r"plug-in fit ($\hat\sigma = s$)",
         "ucb": rf"upper bound on $\sigma$ ({UCB_LEVEL:.0%} $\chi^2$)",
         "oracle": r"oracle ($\sigma$ known)"}
STY = {"plugin": dict(color=COL_BASE1, linestyle="--", marker="s"),
       "ucb": dict(color=COL_UCB, linestyle="-", marker="^"),
       "oracle": dict(color=COL_SCG, linestyle="-.", linewidth=1.2)}


def fit_sigma(x: np.ndarray, kind: str) -> tuple[float, float]:
    """(mu_hat, sigma_hat) from the offsets x."""
    n = x.size
    mu, s = float(x.mean()), float(x.std(ddof=1))
    if kind == "plugin":
        return mu, s
    if kind == "ucb":
        return mu, s * math.sqrt((n - 1) / stats.chi2.ppf(1.0 - UCB_LEVEL, n - 1))
    raise ValueError(kind)


def belief_graph(g, mu_hat: float, sigma_hat: float):
    """The instance with every air start law replaced by the fitted law."""
    out = []
    for c in g.contacts:
        if c.layer == "air":
            nom = c.start.mu + mu_hat
            out.append(c.with_laws(start=TruncNormal(nom, sigma_hat, nom - TRUNC_W * sigma_hat,
                                                     nom + TRUNC_W * sigma_hat)))
        else:
            out.append(c)
    return g.replaced(out)


def one_instance(task, rng):
    sigma, idx, quick = task
    r_sel, r_train, r_ev = rng.spawn(3)
    g, p = make_instance(r_sel, sigma, margin=MARGIN)     # g is the truth
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    m_eval = 5000 if quick else M_EVAL
    offsets = TruncNormal(0.0, sigma, -TRUNC_W * sigma, TRUNC_W * sigma).sample(
        r_train, max(NS))

    beliefs = [("oracle", 0, 0.0, sigma)]
    for n in NS:
        for kind in FITS:
            mu, sh = fit_sigma(offsets[:n], kind)
            beliefs.append((kind, n, mu, sh))

    cache: dict = {}
    recs = []
    for kind, n, mu, sh in beliefs:
        gb = g if kind == "oracle" else belief_graph(g, mu, sh)
        cands = candidate_routes(gb, src, dst, size, t0, r_sel, 40)
        m = admission_draws(EPS, len(cands), DELTA)
        route, st = pick_route_chance(gb, src, dst, size, t0, D, r_sel, m=m, eps=EPS,
                                      admission="lcb", delta=DELTA, cands=cands)
        cids = tuple(c.cid for c in route) if route else ()
        if cids not in cache:
            ev = evaluate_route([g.by_id(c) for c in cids] if cids else None,
                                size, t0, D, m_eval, r_ev)
            cache[cids] = (ev["fails"], ev["n"], ev["mean_delivery"])
        fails, ne, md = cache[cids]
        recs.append({
            "sigma_true": sigma, "idx": idx, "fit": kind, "n": n,
            "sigma_ratio": sh / sigma, "mu_hat": mu, "fails": fails, "n_eval": ne,
            "lat": md - t0 if math.isfinite(md) else math.nan,
            "air": any(g.by_id(c).layer == "air" for c in cids),
            "pred": 1.0 - st["p_success"], "feasible": bool(st.get("feasible")),
        })
    return recs


def ci95(xs):
    xs = np.asarray(xs, dtype=float)
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def aggregate(recs):
    groups: dict = {}
    for r in recs:
        groups.setdefault((r["sigma_true"], r["fit"], r["n"]), []).append(r)
    rows = []
    for (sigma, fit, n), rs in groups.items():
        viol = np.array([r["fails"] / r["n_eval"] for r in rs])
        lat = [r["lat"] for r in rs if math.isfinite(r["lat"])]
        fails = sum(r["fails"] for r in rs)
        ne = sum(r["n_eval"] for r in rs)
        ratio = np.array([r["sigma_ratio"] for r in rs])
        rows.append({
            "sigma_true": sigma, "fit": fit, "n": n if fit != "oracle" else "",
            "viol": float(viol.mean()), "ci_viol": ci95(viol),
            "viol_ub": cp_upper(fails, ne, 0.05), "max_viol": float(viol.max()),
            "frac_over": float(np.mean(viol > EPS)),
            "frac_breach": float(np.mean([cp_lower(r["fails"], r["n_eval"], 0.025) > EPS
                                          for r in rs])),
            "frac_air": float(np.mean([r["air"] for r in rs])),
            "lat": float(np.mean(lat)) if lat else math.nan,
            "ci_lat": ci95(lat) if len(lat) > 1 else 0.0,
            "sigma_ratio_mean": float(ratio.mean()),
            "sigma_ratio_median": float(np.median(ratio)),
            "frac_sigma_under": float(np.mean(ratio < 1.0)),
            "mu_hat_sd": float(np.std([r["mu_hat"] for r in rs])),
            "pred_insample": float(np.mean([r["pred"] for r in rs])),
            "feasible_frac": float(np.mean([r["feasible"] for r in rs])),
            "fails": fails, "n_eval": ne, "R": len(rs), "eps": EPS,
        })
    order = {"oracle": 0, "plugin": 1, "ucb": 2}
    rows.sort(key=lambda r: (-r["sigma_true"], order[r["fit"]],
                             0 if r["n"] == "" else r["n"]))
    return rows


def plot(rows, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    metrics = [("viol", "ci_viol", "held-out SLA-violation rate"),
               ("frac_over", None, r"instances with violation $>\varepsilon$"),
               ("lat", "ci_lat", "mean delivery time [s]")]
    handles = {}
    for i, sigma in enumerate(SIGMAS_TRUE):
        orc = next(r for r in rows if r["sigma_true"] == sigma and r["fit"] == "oracle")
        for j, (key, ci, ylab) in enumerate(metrics):
            ax = axes[i, j]
            for fit in FITS:
                rs = sorted([r for r in rows if r["sigma_true"] == sigma and r["fit"] == fit],
                            key=lambda r: r["n"])
                x = [r["n"] for r in rs]
                y = [r[key] for r in rs]
                if ci:
                    h = ax.errorbar(x, y, yerr=[r[ci] for r in rs], capsize=2, **STY[fit])
                else:
                    h, = ax.plot(x, y, **STY[fit])
                handles.setdefault(fit, h)
            h = ax.axhline(orc[key], **STY["oracle"])
            handles.setdefault("oracle", h)
            if key == "viol":
                ax.axhline(EPS, color=COL_BOUND, linestyle=":", linewidth=1.0)
                ax.annotate(rf"$\varepsilon={EPS:g}$", (210, EPS), color="#52514e",
                            fontsize=6.5, ha="right", va="bottom")
            ax.set_xscale("log")
            ax.set_xticks(NS, [str(n) for n in NS])
            ax.minorticks_off()
            ax.set_xlabel("observed windows $n$")
            ax.set_ylabel(ylab)
            ax.set_ylim(bottom=0)
            if key == "frac_over":
                ax.set_ylim(0, 1.0)
            ax.set_title(rf"({'abcdef'[3 * i + j]}) $\sigma_{{true}}={sigma:g}$ s",
                         loc="left")
    fig.legend([handles[k] for k in ("plugin", "ucb", "oracle")],
               [LABEL[k] for k in ("plugin", "ucb", "oracle")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.tight_layout(pad=0.4, rect=(0, 0.05, 1, 1))
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")


def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    R = 10 if quick else 300
    tasks = [(sigma, i, quick) for sigma in SIGMAS_TRUE for i in range(R)]
    recs = [r for out in pmap(one_instance, tasks, seed=SEED) for r in out]
    rows = aggregate(recs)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "estimation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        n = "oracle" if r["fit"] == "oracle" else f"n={r['n']}"
        print(f"sigma_true={r['sigma_true']:4.0f} {r['fit']:6s} {n:7s} "
              f"viol={r['viol']:.4f} +-{r['ci_viol']:.4f}  over={r['frac_over']:.3f} "
              f"breach={r['frac_breach']:.3f}  air={r['frac_air']:.2f}  "
              f"lat={r['lat']:6.1f}  sigma_hat/sigma={r['sigma_ratio_median']:.2f} "
              f"(P[under]={r['frac_sigma_under']:.2f})")

    plot(rows, str(ROOT / "figures" / "estimation"))
    print(f"done in {time.time() - t_start:.1f}s -> results/estimation.csv, "
          f"figures/estimation.pdf")


if __name__ == "__main__":
    main()
