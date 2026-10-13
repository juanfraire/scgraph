"""Planning on stale information (revision T8, R1.e).

Reviewer 1's second objection: a ground-central planner works from telemetry
that is already old when the table reaches the nodes, and the air layer keeps
moving in the meantime. This runner ages the planner's information by Delta_a
seconds and measures what that costs, on UC1 (make_instance_cpd, the CPD
orbital backbone plus three jittered air relays).

Truth model. Each air contact's start is nominal + jitter + drift. The jitter
is the instance's TruncNormal(0, sigma) as before. The drift accumulates while
the planner's information ages: one draw per air contact and instance,
d_c = kappa * sqrt(Delta_a) * z_c with z_c ~ N(0, 1), a random walk with
diffusion kappa. The truth graph replaces each air contact's start law by
TruncNormal(nominal + d_c, sigma, +-3.5 sigma) via StochasticContact.with_laws;
the space layer is timing-certain (A1) and does not drift. The same z_c are
reused at every Delta_a, so the sweep is paired: an instance only moves along
its own drift direction as its information ages.

KAPPA = 1 s per sqrt(s): the drift's standard deviation is 7.7 s after one
minute, 30 s after 15 minutes and 60 s after an hour.

Planners (all see the nominal plan, none sees d_c):
  det           deterministic CGR on the nominal (mean) plan.
  maxp_unaware  pick_route (argmax P[T <= D]) on the stale laws, sigma only.
  maxp_aware    pick_route on sigma_eff = sqrt(sigma^2 + kappa^2 Delta_a), the
                law a planner gets by widening its timing law for the age of
                its information.
  cc_unaware    pick_route_chance, eps = 0.05, admission = "lcb", on sigma.
  cc_aware      pick_route_chance on sigma_eff.
  maxp_fresh,   the same two routers on the truth law itself (the drift is
  cc_fresh      known): the reference for what fresh information would buy.

Every committed route is scored on held-out draws of the truth
(verify.montecarlo.evaluate_route, a stream no planner touched), with exact
Clopper-Pearson intervals (scgraph/stats.py) on the pooled failure counts.

Two jitter levels. sigma = 40 s is the setting of the paper's UC1 and of this
study's main question. At that jitter the SLA-aware routers already retreat to
the timing-certain backbone, so stale air information cannot hurt them; the
second level, sigma = 10 s, is a well-characterised air segment where both
routers commit to the air relays, which is where staleness can bite.

Usage: python experiments/run_staleness.py [--quick] [--replot]
Outputs: results/staleness.csv, figures/staleness.{pdf,png}
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
from _parallel import default_workers, pmap
from _style import COL_BASE1, COL_BASE2, COL_BOUND, COL_SCG

from scgraph.distributions import TruncNormal
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, pick_route, pick_route_chance
from scgraph.scenarios.uc1_event import make_instance_cpd
from scgraph.stats import clopper_pearson, cp_lower, cp_upper
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
SIGMAS = [40.0, 10.0]
DELTAS = [0.0, 60.0, 300.0, 900.0, 1800.0, 3600.0]
KAPPA = 1.0                 # drift diffusion, s / sqrt(s)
EPS = 0.05
DELTA_CONF = 0.05
N_REAL = 40
POLICIES = ["det", "maxp_unaware", "maxp_aware", "cc_unaware", "cc_aware",
            "maxp_fresh", "cc_fresh"]


def _tn(mu: float, s: float) -> TruncNormal:
    return TruncNormal(mu, s, lo=mu - 3.5 * s, hi=mu + 3.5 * s)


def _restart(g, laws: dict):
    """The graph with some contacts' start laws replaced (cid -> law)."""
    return g.replaced([c.with_laws(start=laws[c.cid]) if c.cid in laws else c
                       for c in g.contacts])


def _plan(g, p, rng, m):
    """(max-P route, its stats, chance-constrained route, its stats) on graph g."""
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    cands = candidate_routes(g, src, dst, size, t0, rng, N_REAL)
    r1, s1 = pick_route(g, src, dst, size, t0, D, rng, m=m, eps=EPS, cands=cands)
    r2, s2 = pick_route_chance(g, src, dst, size, t0, D, rng, m=m, eps=EPS,
                               admission="lcb", delta=DELTA_CONF, cands=cands)
    return r1, s1, r2, s2


def one_instance(task, rng):
    """All Delta_a levels for one UC1 instance; returns one row per (Delta, policy)."""
    sigma, i, m, m_eval = task
    rng_inst, rng_sel, rng_eval = rng.spawn(3)
    g, p = make_instance_cpd(rng_inst, sigma)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    air = [c for c in g.contacts if c.layer == "air"]
    nominal = {c.cid: c.start.mu for c in air}
    z = {c.cid: float(rng_inst.standard_normal()) for c in air}

    det_route, _ = cgr_route(g.mean_plan(), src, dst, size, t0)
    det = [g.by_id(c.cid) for c in det_route] if det_route else None
    stale = _plan(g, p, rng_sel, m)                 # independent of Delta_a

    rows = []
    for delta in DELTAS:
        sd = KAPPA * math.sqrt(delta)
        if delta == 0.0:
            truth, aware, fresh = g, stale, stale
        else:
            truth = _restart(g, {cid: _tn(nominal[cid] + sd * z[cid], sigma)
                                 for cid in nominal})
            s_eff = math.sqrt(sigma ** 2 + sd ** 2)
            g_aware = _restart(g, {cid: _tn(nominal[cid], s_eff) for cid in nominal})
            aware = _plan(g_aware, p, rng_sel, m)
            fresh = _plan(truth, p, rng_sel, m)

        chosen = {
            "det": (det, None),
            "maxp_unaware": (stale[0], stale[1]),
            "maxp_aware": (aware[0], aware[1]),
            "cc_unaware": (stale[2], stale[3]),
            "cc_aware": (aware[2], aware[3]),
            "maxp_fresh": (fresh[0], fresh[1]),
            "cc_fresh": (fresh[2], fresh[3]),
        }
        for pol in POLICIES:
            route, stats = chosen[pol]
            rt = [truth.by_id(c.cid) for c in route] if route else None
            ev = evaluate_route(rt, size, t0, D, m_eval, rng_eval)
            rows.append({
                "sigma": sigma, "inst": i, "delta": delta, "policy": pol,
                "viol": ev["viol"], "fails": ev["fails"], "n": ev["n"],
                "lat": ev["mean_delivery"] - t0,
                "air": bool(route) and any(c.layer == "air" for c in route),
                "promised": (1.0 - stats["p_success"]) if stats else math.nan,
                "feasible": stats.get("feasible", True) if stats else True,
            })
    return rows


def aggregate(raw):
    out = []
    for sigma in SIGMAS:
        for delta in DELTAS:
            sd = KAPPA * math.sqrt(delta)
            for pol in POLICIES:
                rs = [r for r in raw if r["sigma"] == sigma and r["delta"] == delta
                      and r["policy"] == pol]
                if not rs:
                    continue
                v = np.array([r["viol"] for r in rs])
                fails = int(sum(r["fails"] for r in rs))
                n = int(sum(r["n"] for r in rs))
                lo, hi = clopper_pearson(fails, n)
                lat = np.array([r["lat"] for r in rs if math.isfinite(r["lat"])])
                prom = np.array([r["promised"] for r in rs
                                 if not math.isnan(r["promised"])])
                R = len(rs)
                out.append({
                    "sigma": sigma, "delta_s": delta, "drift_sd_s": sd,
                    "sigma_eff_s": math.sqrt(sigma ** 2 + sd ** 2),
                    "policy": pol,
                    "viol": float(v.mean()),
                    "ci_viol": 1.96 * v.std(ddof=1) / math.sqrt(R) if R > 1 else 0.0,
                    "fails": fails, "n": n,
                    "viol_lo": lo, "viol_hi": hi,
                    "viol_ub": cp_upper(fails, n, DELTA_CONF),
                    "max_inst_viol": float(v.max()),
                    "frac_over_eps": float(np.mean(v > EPS)),
                    "frac_breach": float(np.mean([
                        cp_lower(r["fails"], r["n"], DELTA_CONF) > EPS for r in rs])),
                    "promised_viol": float(prom.mean()) if prom.size else math.nan,
                    "lat": float(lat.mean()) if lat.size else math.nan,
                    "ci_lat": (1.96 * lat.std(ddof=1) / math.sqrt(lat.size)
                               if lat.size > 1 else 0.0),
                    "frac_air": float(np.mean([r["air"] for r in rs])),
                    "feasible_frac": float(np.mean([r["feasible"] for r in rs])),
                    "R": R, "M_eval": rs[0]["n"], "kappa": KAPPA, "eps": EPS,
                })
    return out


def run(quick: bool = False):
    R = 10 if quick else 100
    m = 800 if quick else 2000
    m_eval = 2000 if quick else 10_000
    tasks = [(s, i, m, m_eval) for s in SIGMAS for i in range(R)]
    workers = min(3, default_workers())
    print(f"  {len(tasks)} instances, {workers} workers, M={m}, M_eval={m_eval}")
    raw = [r for rows in pmap(one_instance, tasks, seed=SEED, workers=workers)
           for r in rows]
    return aggregate(raw)


def load():
    with open(ROOT / "results" / "staleness.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "policy":
                r[k] = float(v)
    return rows


STYLES = {
    "det": dict(color=COL_BASE1, linestyle="--", marker="s", label="deterministic CGR"),
    "maxp_unaware": dict(color=COL_BASE2, linestyle="-.", marker="^",
                         label="max-P, stale-unaware"),
    "maxp_aware": dict(color=COL_SCG, linestyle="-", marker="o",
                       label=r"max-P, staleness-aware"),
    "cc_unaware": dict(color=COL_BASE2, linestyle=":", marker="v",
                       markerfacecolor="white", label=r"chance $\varepsilon$=0.05, stale-unaware"),
    "cc_aware": dict(color=COL_SCG, linestyle="--", marker="D", markersize=3,
                     markerfacecolor="white", label=r"chance $\varepsilon$=0.05, staleness-aware"),
    "cc_fresh": dict(color=COL_BOUND, linestyle=":", marker="x",
                     label=r"chance $\varepsilon$=0.05, fresh (drift known)"),
}


def plot(rows):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True)
    ticks = [KAPPA * math.sqrt(d) for d in DELTAS]
    labels = ["0", "1", "5", "15", "30", "60"]
    for j, sigma in enumerate(SIGMAS):
        ax_v, ax_l = axes[0, j], axes[1, j]
        for pol, st in STYLES.items():
            rs = sorted([r for r in rows if r["sigma"] == sigma and r["policy"] == pol],
                        key=lambda r: r["delta_s"])
            if not rs:
                continue
            x = [r["drift_sd_s"] for r in rs]
            ax_v.plot(x, [r["viol"] for r in rs], **st)
            ax_l.plot(x, [r["lat"] for r in rs], **st)
        ax_v.axhline(EPS, color=COL_BOUND, linewidth=0.8, linestyle="-", alpha=0.5)
        ax_v.set_title(rf"UC1, $\sigma$ = {sigma:.0f} s")
        ax_v.set_ylabel("held-out SLA violation")
        ax_l.set_ylabel("mean delivery time [s]")
        ax_l.set_xlabel(r"drift s.d. $\kappa\sqrt{\Delta_a}$ [s]")
        for ax in (ax_v, ax_l):
            ax.set_xticks(ticks)
        ax_v.set_ylim(bottom=-0.01)
        ax_l.set_ylim(bottom=0)
        top = ax_v.secondary_xaxis("top")
        top.set_xticks(ticks)
        top.set_xticklabels(labels)
        top.set_xlabel(r"information age $\Delta_a$ [min]", fontsize=7)
        ax_v.xaxis.set_tick_params(labelbottom=False)
    for ax in axes[1]:
        ax.set_xticklabels([f"{t:.0f}" for t in ticks])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.01), handlelength=3.2)
    fig.tight_layout(pad=0.4, rect=(0, 0.08, 1, 1))
    fig.savefig(str(ROOT / "figures" / "staleness.pdf"))
    fig.savefig(str(ROOT / "figures" / "staleness.png"))


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    t0 = time.time()
    _style.setup()
    if replot:
        rows = load()
        print(f"replotting {len(rows)} rows from results/staleness.csv")
    else:
        print(f"Staleness sweep on UC1 (kappa={KAPPA} s/sqrt(s), eps={EPS}):")
        rows = run(quick)
        (ROOT / "results").mkdir(exist_ok=True)
        with open(ROOT / "results" / "staleness.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"  sigma={r['sigma']:4.0f} Delta={r['delta_s']:6.0f}s"
                  f" {r['policy']:13s} viol={r['viol']:.4f}"
                  f" [{r['viol_lo']:.4f},{r['viol_hi']:.4f}]"
                  f" over_eps={r['frac_over_eps']:.2f} breach={r['frac_breach']:.2f}"
                  f" promised={r['promised_viol']:.4f} lat={r['lat']:6.1f}"
                  f" air={r['frac_air']:.2f} feas={r['feasible_frac']:.2f}")
    (ROOT / "figures").mkdir(exist_ok=True)
    plot(rows)
    print(f"done in {time.time() - t0:.1f}s -> results/staleness.csv, "
          "figures/staleness.pdf")


if __name__ == "__main__":
    main()
