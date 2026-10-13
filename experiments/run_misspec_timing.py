"""Misspecified timing law (revision T6, R2.6): heavy-tailed truth, normal belief (UC1).

The planner believes the instance's TruncNormal(sigma) jitter on every air
contact's start time (truncated at +-3.5 sigma, as `make_instance_cpd` builds
it). The world draws each air start from a location-scale Student-t of the same
standard deviation and nu degrees of freedom, truncated much wider (+-12 sigma)
so the tail can actually reach the hand-off. nu = inf is the normal control.

Matching the standard deviation. `TruncStudentT.matched_std` matches the
UNTRUNCATED variance. At nu = 2.5 the variance integral converges so slowly
that cutting the law at +-12 sigma removes about a quarter of it (the
truncated std is 0.85 sigma), so the heavy-tailed truth would also be a
narrower one and the comparison would understate the tail. The scale is
therefore re-solved (`t_scale`) so that the TRUNCATED law has standard
deviation exactly sigma; the comparison then changes the tail shape and
nothing else. `std_true` in the CSV records the achieved ratio (1.000).

Policies, all selected on the planner's belief and all scored on held-out
draws of the TRUE graph (independent stream, M_EVAL draws per route):

  det        deterministic CGR on the nominal (mean) plan
  sla        SLA-aware max P[T <= D] (`pick_route`, eq. 6)
  cc         chance-constrained, lcb admission at eps (`pick_route_chance`)
  cc_t       mitigation: the same router on a heavy-tailed belief (t, nu = 3,
             same std, +-12 sigma); a planner that hedges the tail shape
  cc_infl    mitigation: the same router on an inflated belief (1.5 sigma);
             the blunt operator knob

The chance-constrained policies run at eps = 0.05 (the paper's operating
point) and eps = 0.01, because a tail misspecification bites in proportion to
how deep in the tail the SLA sits. Per (sigma, nu, policy, eps) the CSV
reports the mean held-out violation with its CI, the pooled exact upper bound,
the fraction of instances whose violation measures above eps (`frac_over`),
the fraction with a significant breach (exact 97.5% lower bound above eps,
`frac_breach`), the fraction routed on the air layer, the planner's own
predicted violation (its belief, held-out draws), and the delivery time.

Usage: python experiments/run_misspec_timing.py [--quick]
Outputs: results/robust_timing.csv, figures/robust_timing.{pdf,png}
"""

from __future__ import annotations

import csv
import functools
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import integrate, optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap
from _style import COL_BASE1, COL_BASE2, COL_BOUND, COL_SCG

from scgraph.distributions import TruncNormal, TruncStudentT
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, pick_route, pick_route_chance
from scgraph.scenarios.uc1_event import make_instance_cpd as make_instance
from scgraph.stats import admission_draws, cp_lower, cp_upper
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
SIGMAS = [20.0, 40.0]
NUS = [math.inf, 10.0, 5.0, 3.0, 2.5]
EPSS = [0.05, 0.01]
MARGIN = 40.0          # UC1 default: the backbone always meets the deadline
BELIEF_W = 3.5         # belief truncation, in sigmas (as make_instance_cpd)
TRUTH_W = 12.0         # truth truncation, in sigmas
T_BELIEF_NU = 3.0      # heavy-tailed belief of the cc_t mitigation
INFLATE = 1.5          # sigma multiplier of the cc_infl mitigation
DELTA = 0.05
M_SEL = 2000           # draws per candidate for the eps-free max-P router
M_EVAL = 40_000        # held-out truth draws per route and instance
EPS_REF = 0.05         # budget used to count breaches of the eps-free policies

COL_CC = COL_SCG
COL_SLA = "#6250d6"    # violet (reference palette slot 7)
COL_T = "#1baf7a"      # aqua (slot 3)
COL_INFL = COL_BASE2

POLICIES = ["det", "sla", "cc", "cc_t", "cc_infl"]
LABEL = {
    "det": "deterministic CGR (mean plan)",
    "sla": r"SLA-aware, max $P[T\leq D]$",
    "cc": r"chance-constrained, normal belief",
    "cc_t": r"chance-constrained, $t_3$ belief",
    "cc_infl": rf"chance-constrained, ${INFLATE:g}\sigma$ belief",
}
STY = {
    "det": dict(color=COL_BASE1, linestyle="--", marker="s"),
    "sla": dict(color=COL_SLA, linestyle=":", marker="D"),
    "cc": dict(color=COL_CC, linestyle="-", marker="o"),
    "cc_t": dict(color=COL_T, linestyle="-.", marker="^"),
    "cc_infl": dict(color=COL_INFL, linestyle=(0, (5, 1, 1, 1)), marker="v"),
}


# ---------------------------------------------------------------- laws

def _trunc_t_std(scale: float, nu: float, w: float) -> float:
    """Standard deviation of t(nu, 0, scale) truncated to [-w, w]."""
    law = TruncStudentT(0.0, scale, nu, -w, w)
    mass = law._fhi - law._flo
    var, _ = integrate.quad(lambda x: x * x * law._t.pdf(x), -w, w, limit=400)
    return math.sqrt(var / mass)


@functools.lru_cache(maxsize=None)
def t_scale(nu: float, w: float = TRUTH_W) -> float:
    """Scale (in units of sigma) giving the [-w, w]-truncated t law unit std."""
    s0 = math.sqrt((nu - 2.0) / nu)            # matched_std's untruncated answer
    return optimize.brentq(lambda s: _trunc_t_std(s, nu, w) - 1.0, 0.5 * s0, 3.0 * s0,
                           xtol=1e-12)


def timing_law(mu: float, sigma: float, nu: float, w: float):
    """Start law centred on the nominal `mu`, std sigma after truncation at +-w sigma."""
    if math.isinf(nu):
        return TruncNormal(mu, sigma, mu - w * sigma, mu + w * sigma)
    return TruncStudentT(mu, t_scale(nu, w) * sigma, nu, mu - w * sigma, mu + w * sigma)


def law_std(sigma: float, nu: float, w: float) -> float:
    """Achieved std of `timing_law` (as a check on the calibration)."""
    if math.isinf(nu):
        return TruncNormal(0.0, sigma, -w * sigma, w * sigma).std()
    return sigma * _trunc_t_std(t_scale(nu, w), nu, w)


def with_air_timing(g, sigma: float, nu: float, w: float):
    """Same graph, every air contact's start law replaced (nominal kept)."""
    out = []
    for c in g.contacts:
        if c.layer == "air":
            out.append(c.with_laws(start=timing_law(c.start.mu, sigma, nu, w)))
        else:
            out.append(c)
    return g.replaced(out)


def remap(route, g):
    return None if route is None else [g.by_id(c.cid) for c in route]


# ------------------------------------------------------------- one instance

def one_instance(task, rng):
    sigma, idx, quick = task
    r_sel, r_ev = rng.spawn(2)
    g, p = make_instance(r_sel, sigma, margin=MARGIN)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    m_eval = 8000 if quick else M_EVAL

    cands = candidate_routes(g, src, dst, size, t0, r_sel, 40)
    beliefs = {
        "cc": g,
        "cc_t": with_air_timing(g, sigma, T_BELIEF_NU, TRUTH_W),
        "cc_infl": with_air_timing(g, INFLATE * sigma, math.inf, BELIEF_W),
    }

    picks = {}                                  # (policy, eps) -> (route, belief graph, feasible)
    det_route, _ = cgr_route(g.mean_plan(), src, dst, size, t0)
    picks[("det", None)] = (remap(det_route, g), g, True)
    sla_route, _ = pick_route(g, src, dst, size, t0, D, r_sel, m=M_SEL, cands=cands)
    picks[("sla", None)] = (sla_route, g, True)
    for eps in EPSS:
        m = admission_draws(eps, len(cands), DELTA)
        for name, gb in beliefs.items():
            route, st = pick_route_chance(
                gb, src, dst, size, t0, D, r_sel, m=m, eps=eps, admission="lcb",
                delta=DELTA, cands=[remap(r, gb) for r in cands],
            )
            picks[(name, eps)] = (route, gb, bool(st.get("feasible")))

    truths = {nu: with_air_timing(g, sigma, nu, TRUTH_W) for nu in NUS}
    cache: dict = {}

    def score(route, graph, key):
        if key not in cache:
            ev = evaluate_route(remap(route, graph), size, t0, D, m_eval, r_ev)
            cache[key] = (ev["fails"], ev["n"], ev["mean_delivery"])
        return cache[key]

    recs = []
    for (pol, eps), (route, gb, feas) in picks.items():
        cids = tuple(c.cid for c in route) if route else ()
        air = any(c.layer == "air" for c in route) if route else False
        pf, pn, _ = score(route, gb, ("belief", pol, cids))
        for nu in NUS:
            fails, n, md = score(route, truths[nu], ("truth", nu, cids))
            recs.append({
                "sigma": sigma, "idx": idx, "nu": nu, "policy": pol, "eps": eps,
                "fails": fails, "n": n, "lat": md - t0 if math.isfinite(md) else math.nan,
                "air": air, "feasible": feas, "pred": pf / pn,
            })
    return recs


# ------------------------------------------------------------- aggregation

def ci95(xs):
    xs = np.asarray(xs, dtype=float)
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def aggregate(recs):
    groups: dict = {}
    for r in recs:
        groups.setdefault((r["sigma"], r["nu"], r["policy"], r["eps"]), []).append(r)
    rows = []
    for (sigma, nu, pol, eps), rs in groups.items():
        budget = EPS_REF if eps is None else eps
        viol = np.array([r["fails"] / r["n"] for r in rs])
        fails = sum(r["fails"] for r in rs)
        n = sum(r["n"] for r in rs)
        lat = [r["lat"] for r in rs if math.isfinite(r["lat"])]
        rows.append({
            "sigma": sigma, "nu": nu, "inv_nu": 0.0 if math.isinf(nu) else 1.0 / nu,
            "policy": pol, "eps": "" if eps is None else eps, "budget": budget,
            "std_true": law_std(sigma, nu, TRUTH_W) / sigma,
            "viol": float(viol.mean()), "ci_viol": ci95(viol),
            "viol_ub": cp_upper(fails, n, 0.05), "max_viol": float(viol.max()),
            "frac_over": float(np.mean(viol > budget)),
            "frac_breach": float(np.mean([cp_lower(r["fails"], r["n"], 0.025) > budget
                                          for r in rs])),
            "pred": float(np.mean([r["pred"] for r in rs])),
            "frac_air": float(np.mean([r["air"] for r in rs])),
            "feasible_frac": float(np.mean([r["feasible"] for r in rs])),
            "lat": float(np.mean(lat)) if lat else math.nan,
            "ci_lat": ci95(lat) if len(lat) > 1 else 0.0,
            "fails": fails, "n": n, "R": len(rs), "M_eval": rs[0]["n"],
        })
    order = {p: i for i, p in enumerate(POLICIES)}
    rows.sort(key=lambda r: (r["sigma"], order[r["policy"]],
                             -1.0 if r["eps"] == "" else -r["eps"], r["inv_nu"]))
    return rows


# ------------------------------------------------------------------- figure

def plot(rows, path):
    import matplotlib.pyplot as plt

    def series(sigma, pol, eps):
        rs = [r for r in rows if r["sigma"] == sigma and r["policy"] == pol
              and r["eps"] == ("" if eps is None else eps)]
        rs.sort(key=lambda r: r["inv_nu"])
        return rs

    xt = [0.0 if math.isinf(nu) else 1.0 / nu for nu in NUS]
    xl = [r"$\infty$" if math.isinf(nu) else f"{nu:g}" for nu in NUS]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))

    panels = [
        (axes[0, 0], 20.0, 0.05, [("det", None), ("sla", None), ("cc", 0.05),
                                  ("cc_t", 0.05), ("cc_infl", 0.05)]),
        (axes[0, 1], 20.0, 0.01, [("sla", None), ("cc", 0.01), ("cc_t", 0.01),
                                  ("cc_infl", 0.01)]),
        (axes[1, 0], 40.0, 0.05, [("det", None), ("sla", None), ("cc", 0.05),
                                  ("cc_t", 0.05), ("cc_infl", 0.05)]),
    ]
    tags = ["(a)", "(b)", "(c)"]
    handles = {}
    for (ax, sigma, eps, pols), tag in zip(panels, tags):
        top = eps
        for pol, e in pols:
            rs = series(sigma, pol, e)
            if not rs:
                continue
            h = ax.errorbar([r["inv_nu"] for r in rs], [r["viol"] for r in rs],
                            yerr=[r["ci_viol"] for r in rs], capsize=2, label=LABEL[pol],
                            **STY[pol])
            handles.setdefault(pol, h)
            top = max(top, max(r["viol"] + r["ci_viol"] for r in rs))
        ax.axhline(eps, color=COL_BOUND, linestyle=":", linewidth=1.0)
        ax.annotate(rf"budget $\varepsilon={eps:g}$", (-0.01, eps), color="#52514e",
                    fontsize=6.5, ha="left", va="bottom")
        ax.set_xticks(xt, xl)
        ax.set_xlabel(r"true tail: Student-$t$ degrees of freedom $\nu$")
        ax.set_ylabel("held-out SLA-violation rate")
        ax.set_title(rf"{tag} $\sigma={sigma:g}$ s, chance-constrained at "
                     rf"$\varepsilon={eps:g}$", loc="left")
        ax.set_ylim(0, 1.15 * top)
    fig.legend([handles[p] for p in POLICIES if p in handles],
               [LABEL[p] for p in POLICIES if p in handles],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)

    ax = axes[1, 1]
    for pol, eps, ls in (("cc", 0.05, "-"), ("cc", 0.01, "--"), ("cc_t", 0.01, "-."),
                         ("cc_infl", 0.01, ":")):
        rs = series(20.0, pol, eps)
        if not rs:
            continue
        sty = dict(STY[pol])
        sty["linestyle"] = ls
        ax.plot([r["inv_nu"] for r in rs], [r["frac_over"] for r in rs],
                label=rf"{LABEL[pol].split(',')[1].strip()}, $\varepsilon={eps:g}$", **sty)
    ax.set_xticks(xt, xl)
    ax.set_xlabel(r"true tail: Student-$t$ degrees of freedom $\nu$")
    ax.set_ylabel(r"instances with violation $>\varepsilon$")
    ax.set_title(r"(d) $\sigma=20$ s, per-instance breaches", loc="left")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=6.3)

    fig.tight_layout(pad=0.4, rect=(0, 0.07, 1, 1))
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")


# --------------------------------------------------------------------- main

def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()
    R = 12 if quick else 300
    tasks = [(sigma, i, quick) for sigma in SIGMAS for i in range(R)]
    recs = [r for out in pmap(one_instance, tasks, seed=SEED) for r in out]
    rows = aggregate(recs)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "robust_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        e = "  -  " if r["eps"] == "" else f"{r['eps']:.3f}"
        print(f"sigma={r['sigma']:4.0f} nu={r['nu']:>4} {r['policy']:8s} eps={e}"
              f"  viol={r['viol']:.4f} (pred {r['pred']:.4f}, ub {r['viol_ub']:.4f})"
              f"  over={r['frac_over']:.2f} breach={r['frac_breach']:.2f}"
              f"  air={r['frac_air']:.2f}  lat={r['lat']:6.1f}")

    plot(rows, str(ROOT / "figures" / "robust_timing"))
    print(f"done in {time.time() - t_start:.1f}s -> results/robust_timing.csv, "
          f"figures/robust_timing.pdf")


if __name__ == "__main__":
    main()
