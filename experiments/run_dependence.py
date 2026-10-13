"""Strength and structure of inter-contact dependence (revision T6, R3.2 / R2.6), UC3.

UC3 replicates a bundle over D2D relays grouped in districts. Each district
collapses (every relay contact in it drops to a tenth of its existence
probability) with marginal probability rho = 0.5, and the router budgets
N = 2 copies. The submitted study let the two districts collapse
independently. Here the districts' collapses are coupled, and the number of
districts varies, so that the result can be read against both the STRENGTH and
the STRUCTURE of the dependence (R3).

Instance builder (`make_instance_dep`, local to this file; it extends
`uc3_emergency.make_instance` without modifying it):
  districts   n_d in {2, 3, 4}; six relays split across them as evenly as
              possible (3+3, 2+2+2, 2+2+1+1). District j's first hand-off
              opens in [20, 60], [80, 140], [140, 200], [200, 260] s, so the
              first two districts are exactly UC3's regions A and B.
  coupling    a common-shock mixture. With probability kappa the coupled
              districts share one collapse coin (all collapse, w.p. rho, or
              none); with probability 1 - kappa every district draws its own.
              Each marginal stays rho, and any coupled pair has
              P[both] = rho^2 + kappa rho (1 - rho), i.e. collapse
              correlation kappa. kappa = 1 makes the coupled districts fail
              as one.
  structure   "common": every district loads on the shock (exchangeable).
              "pair": only districts A and B (the two fastest) share it; the
              others stay independent, so some diversification choices are
              better than others and the router has to know which.
At n_d = 2 and kappa = 0 the builder reproduces `make_instance` draw for draw
(same contacts, same latent states in the same order); `--check` asserts it.

Policies at equal copy budget N = 2, all scored on held-out joint draws of the
TRUE model (`evaluate_route_set`, independent stream):
  scg       greedy joint-success selection with the true latent model
  scg_mis   the same greedy selection on a latent model that keeps the
            district structure but assumes kappa = 0 (independent districts)
  bern      Bernoulli-TEG corner (`bernoulli_select`): marginals only
  naive     the N nominally fastest candidate routes
  stale     the single nominal-best route of the pre-collapse plan
  epidemic  every candidate route (unbounded overhead; reference)
`pred_mis` is the delivery ratio the misspecified planner predicts for its own
choice (its belief, same held-out draws), so pred_mis - scg_mis is its
over-confidence.

Usage: python experiments/run_dependence.py [--quick] [--check]
Outputs: results/robust_dependence.csv, figures/robust_dependence.{pdf,png}
"""

from __future__ import annotations

import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap
from _style import COL_BASE1, COL_BASE2, COL_BOUND, COL_SCG

from scgraph.arrival import delivery_time
from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac
from scgraph.graph import LatentModel, StochasticContactGraph
from scgraph.routing.bernoulli import bernoulli_select
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes
from scgraph.routing.replication import greedy_select
from scgraph.verify.montecarlo import evaluate_route_set

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261015
RHO = 0.5
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CELLS = [(2, "common"), (3, "common"), (4, "common"), (3, "pair"), (4, "pair")]
N_COPIES = 2
N_RELAYS = 6
BERN_SLOT = 30.0
M_SEL = 2000             # joint-success draws per greedy evaluation (run_uc3: M // 2)
M_BERN = 4000
M_EVAL = 20_000
DISTRICTS = ["A", "B", "C", "D"]
DISTRICT_S1 = [(20.0, 60.0), (80.0, 140.0), (140.0, 200.0), (200.0, 260.0)]

POLICIES = ["epidemic", "scg", "scg_mis", "bern", "naive", "stale"]
LABEL = {
    "epidemic": "epidemic (unbounded)",
    "scg": "SCG, true dependence",
    "scg_mis": r"SCG, assumes $\kappa=0$",
    "bern": "Bernoulli TEG, independent",
    "naive": "naive, nominal-fastest",
    "stale": "stale plan (1 copy)",
}
STY = {
    "epidemic": dict(color=COL_BOUND, linestyle=":", marker=""),
    "scg": dict(color=COL_SCG, linestyle="-", marker="o"),
    "scg_mis": dict(color="#1baf7a", linestyle=(0, (5, 1, 1, 1)), marker="D"),
    "bern": dict(color="#7b4fa8", linestyle="-.", marker="^"),
    "naive": dict(color=COL_BASE2, linestyle="--", marker="v"),
    "stale": dict(color=COL_BASE1, linestyle="--", marker="s"),
}


# ------------------------------------------------------------ the instance

def district_latent(n_d: int, rho: float, kappa: float, structure: str,
                    collapse_scale: float = 0.1) -> LatentModel:
    """Common-shock latent model over the 2^n_d district collapse patterns.

    States are enumerated with the first district varying fastest, which for
    n_d = 2 is the order `uc3_emergency.make_instance` uses.
    """
    names = DISTRICTS[:n_d]
    coupled = list(range(n_d)) if structure == "common" else [0, 1]
    states = []
    for bits in itertools.product((0, 1), repeat=n_d):
        pat = bits[::-1]                                   # district 0 fastest
        p_ind = 1.0
        for b in pat:
            p_ind *= rho if b else 1.0 - rho
        shock = [pat[j] for j in coupled]
        if all(shock):
            p_sh = rho
        elif not any(shock):
            p_sh = 1.0 - rho
        else:
            p_sh = 0.0
        p_rest = 1.0
        for j, b in enumerate(pat):
            if j not in coupled:
                p_rest *= rho if b else 1.0 - rho
        p = (1.0 - kappa) * p_ind + kappa * p_sh * p_rest
        if p > 0.0:
            states.append((p, {g: (collapse_scale if b else 1.0) for g, b in zip(names, pat)}))
    return LatentModel(states=states)


def make_instance_dep(rng: np.random.Generator, kappa: float, n_districts: int = 2,
                      rho: float = RHO, structure: str = "common",
                      n_relays: int = N_RELAYS, p_base: float = 0.9,
                      collapse_scale: float = 0.1):
    """UC3 with n_districts districts whose collapses are coupled at kappa."""
    src, dst = 0, 1
    g = StochasticContactGraph(num_nodes=2 + n_relays)
    size = 50.0
    cid = 0

    def relay_pair(node: int, district: str, s1: float):
        nonlocal cid
        rate = 10.0
        dur = float(rng.uniform(80.0, 140.0))
        arr1 = s1 + size / rate
        s2 = arr1 + float(rng.uniform(10.0, 30.0))
        g.add(StochasticContact(cid, src, node, Dirac(s1), Dirac(dur), Dirac(rate),
                                owlt=0.2, layer="ground", p_exist=p_base, group=district))
        cid += 1
        g.add(StochasticContact(cid, node, dst, Dirac(s2), Dirac(dur), Dirac(rate),
                                owlt=0.2, layer="ground", p_exist=p_base, group=district))
        cid += 1

    node = 2
    for j, members in enumerate(np.array_split(np.arange(n_relays), n_districts)):
        lo, hi = DISTRICT_S1[j]
        for _ in members:
            relay_pair(node, DISTRICTS[j], s1=float(rng.uniform(lo, hi)))
            node += 1

    g.latent = district_latent(n_districts, rho, kappa, structure, collapse_scale)
    params = {"src": src, "dst": dst, "size": size, "t0": 0.0, "deadline": 400.0}
    return g, params


def with_latent(g, latent):
    return StochasticContactGraph(num_nodes=g.num_nodes, contacts=list(g.contacts),
                                  horizon=g.horizon, latent=latent)


def nominal_arrival(route, size, t0):
    return delivery_time([c.mean_det() for c in route], size, t0)


def check_reproduces_uc3(n: int = 20):
    """n_d = 2, kappa = 0 must rebuild uc3_emergency.make_instance exactly."""
    from scgraph.scenarios.uc3_emergency import make_instance

    for seed in range(n):
        a, pa = make_instance(np.random.default_rng(seed), RHO)
        b, pb = make_instance_dep(np.random.default_rng(seed), 0.0, 2)
        assert pa == pb
        assert len(a.contacts) == len(b.contacts)
        for x, y in zip(a.contacts, b.contacts):
            assert (x.cid, x.frm, x.to, x.group, x.p_exist, x.owlt) == \
                   (y.cid, y.frm, y.to, y.group, y.p_exist, y.owlt)
            assert (x.start.value, x.duration.value, x.rate.value) == \
                   (y.start.value, y.duration.value, y.rate.value)
        assert a.latent.states == b.latent.states
    # the coupling has the stated marginals and pairwise collapse probability
    for n_d, structure in CELLS:
        for kappa in KAPPAS:
            lat = district_latent(n_d, RHO, kappa, structure)
            names = DISTRICTS[:n_d]
            col = {g: sum(p for p, s in lat.states if s[g] < 1.0) for g in names}
            assert all(abs(v - RHO) < 1e-12 for v in col.values())
            both = sum(p for p, s in lat.states if s["A"] < 1.0 and s["B"] < 1.0)
            assert abs(both - (RHO**2 + kappa * RHO * (1 - RHO))) < 1e-12
    print(f"check: make_instance_dep(kappa=0, 2 districts) == make_instance on {n} seeds;"
          " marginals and P[A and B collapse] exact in every cell")


# ------------------------------------------------------------- one instance

def one_instance(task, rng):
    kappa, n_d, structure, idx, quick = task
    r_sel, r_bern, r_ev = rng.spawn(3)
    g, p = make_instance_dep(r_sel, kappa, n_d, structure=structure)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    lat_mis = district_latent(n_d, RHO, 0.0, structure)
    m_sel = 1000 if quick else M_SEL
    m_eval = 4000 if quick else M_EVAL

    cands = candidate_routes(g, src, dst, size, t0, r_sel, n_realizations=30)
    rec = {"kappa": kappa, "n_d": n_d, "structure": structure, "idx": idx,
           "n_cands": len(cands)}
    if not cands:
        for k in POLICIES:
            rec[k] = 0.0
        rec["pred_mis"] = 0.0
        return rec

    stale_route, _ = cgr_route(g.mean_plan(include_threshold=0.0), src, dst, size, t0)
    sets = {
        "stale": [[g.by_id(c.cid) for c in stale_route]] if stale_route else [],
        "naive": sorted(cands, key=lambda r: nominal_arrival(r, size, t0))[:N_COPIES],
        "scg": greedy_select(cands, N_COPIES, size, t0, D, m_sel, r_sel, latent=g.latent),
        "scg_mis": greedy_select(cands, N_COPIES, size, t0, D, m_sel, r_sel, latent=lat_mis),
        "bern": bernoulli_select(g, src, dst, size, t0, D, BERN_SLOT, N_COPIES, r_bern,
                                 m=M_BERN),
        "epidemic": cands,
    }
    for k in POLICIES:
        rec[k] = evaluate_route_set(sets[k], size, t0, D, m_eval, r_ev,
                                    latent=g.latent)["p_success"]
    rec["pred_mis"] = evaluate_route_set(sets["scg_mis"], size, t0, D, m_eval, r_ev,
                                         latent=lat_mis)["p_success"]

    def districts(routes):
        return sorted({c.group for r in routes for c in r})

    for k in ("scg", "scg_mis", "bern", "naive"):
        ds = districts(sets[k])
        rec[f"spread_{k}"] = float(len(ds) == N_COPIES and len(sets[k]) == N_COPIES)
        rec[f"picks_AB_{k}"] = float(set(ds) <= {"A", "B"} and len(ds) == 2)
    return rec


# ------------------------------------------------------------- aggregation

def ci95(xs):
    xs = np.asarray(xs, dtype=float)
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def aggregate(recs):
    groups: dict = {}
    for r in recs:
        groups.setdefault((r["structure"], r["n_d"], r["kappa"]), []).append(r)
    rows = []
    for (structure, n_d, kappa), rs in groups.items():
        row = {"structure": structure, "n_districts": n_d, "kappa": kappa, "rho": RHO,
               "p_both_collapse": RHO**2 + kappa * RHO * (1 - RHO)
               if structure == "common" or n_d == 2 else math.nan,
               "R": len(rs), "N": N_COPIES}
        for k in POLICIES:
            row[k] = float(np.mean([r[k] for r in rs]))
            row[f"ci_{k}"] = ci95([r[k] for r in rs])
        row["pred_mis"] = float(np.mean([r["pred_mis"] for r in rs]))
        row["overconf_mis"] = row["pred_mis"] - row["scg_mis"]
        row["gain_scg_vs_naive"] = row["scg"] - row["naive"]
        row["loss_mis"] = row["scg"] - row["scg_mis"]
        row["ci_loss_mis"] = ci95([r["scg"] - r["scg_mis"] for r in rs])
        for k in ("scg", "scg_mis", "bern", "naive"):
            row[f"spread_{k}"] = float(np.mean([r.get(f"spread_{k}", 0.0) for r in rs]))
            row[f"picks_AB_{k}"] = float(np.mean([r.get(f"picks_AB_{k}", 0.0) for r in rs]))
        row["n_cands"] = float(np.mean([r["n_cands"] for r in rs]))
        rows.append(row)
    order = {c: i for i, c in enumerate(CELLS)}
    rows.sort(key=lambda r: (order[(r["n_districts"], r["structure"])], r["kappa"]))
    return rows


# ------------------------------------------------------------------- figure

def plot(rows, path):
    import matplotlib.pyplot as plt

    def cell(n_d, structure):
        return sorted([r for r in rows if r["n_districts"] == n_d
                       and r["structure"] == structure], key=lambda r: r["kappa"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    handles = {}

    for ax, (n_d, structure), title in (
        (axes[0, 0], (2, "common"), "(a) two districts (UC3's A and B)"),
        (axes[0, 1], (3, "pair"), "(b) three districts, only A and B coupled"),
    ):
        rs = cell(n_d, structure)
        for k in POLICIES:
            if k == "epidemic":
                h, = ax.plot([r["kappa"] for r in rs], [r[k] for r in rs], **STY[k])
            else:
                h = ax.errorbar([r["kappa"] for r in rs], [r[k] for r in rs],
                                yerr=[r[f"ci_{k}"] for r in rs], capsize=2, **STY[k])
            handles.setdefault(k, h)
        ax.set_xlabel(r"collapse correlation $\kappa$ between coupled districts")
        ax.set_ylabel("held-out delivery ratio by deadline")
        ax.set_title(title, loc="left")
        ax.set_ylim(0.3, 0.92)
        ax.set_xticks(KAPPAS)
    axes[0, 0].annotate(r"SCG with the true model and with $\kappa=0$" "\n"
                        "choose the same copies (curves coincide)",
                        (0.02, 0.33), color="#52514e", fontsize=6.3)

    # (c) value of correlation-aware replication over naive, by structure
    ax = axes[1, 0]
    blues = {2: "#8fbcef", 3: "#2a78d6", 4: "#14457f"}
    for n_d, structure, ls in ((2, "common", "-"), (3, "common", "-"), (4, "common", "-"),
                               (3, "pair", "--"), (4, "pair", "--")):
        rs = cell(n_d, structure)
        ax.plot([r["kappa"] for r in rs], [r["gain_scg_vs_naive"] for r in rs],
                color=blues[n_d], linestyle=ls, marker="o",
                label=f"{n_d} districts, {'all coupled' if structure == 'common' else 'A-B coupled'}")
    ax.axhline(0.0, color=COL_BOUND, linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"collapse correlation $\kappa$")
    ax.set_ylabel("delivery gain of SCG over naive\n(same copy budget)")
    ax.set_title("(c) what correlation-aware replication buys", loc="left")
    ax.set_xticks(KAPPAS)
    ax.legend(loc="lower left", fontsize=6.0)

    # (d) over-confidence of the planner that assumes independent districts
    ax = axes[1, 1]
    for n_d, structure, ls in ((2, "common", "-"), (3, "common", "-"), (4, "common", "-"),
                               (3, "pair", "--"), (4, "pair", "--")):
        rs = cell(n_d, structure)
        ax.plot([r["kappa"] for r in rs], [r["overconf_mis"] for r in rs],
                color=blues[n_d], linestyle=ls, marker="o")
    ax.axhline(0.0, color=COL_BOUND, linestyle=":", linewidth=1.0)
    ax.set_xlabel(r"collapse correlation $\kappa$")
    ax.set_ylabel("predicted minus realised\ndelivery ratio")
    ax.set_title(r"(d) over-confidence of the $\kappa=0$ planner", loc="left")
    ax.set_xticks(KAPPAS)

    fig.legend([handles[k] for k in POLICIES], [LABEL[k] for k in POLICIES],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
    fig.tight_layout(pad=0.4, rect=(0, 0.075, 1, 1))
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")


# --------------------------------------------------------------------- main

def main():
    quick = "--quick" in sys.argv
    t_start = time.time()
    if "--check" in sys.argv:
        check_reproduces_uc3()
    _style.setup()
    R = 6 if quick else 200
    tasks = [(kappa, n_d, structure, i, quick)
             for (n_d, structure) in CELLS for kappa in KAPPAS for i in range(R)]
    recs = pmap(one_instance, tasks, seed=SEED, chunksize=4)
    rows = aggregate(recs)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with open(ROOT / "results" / "robust_dependence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"{r['structure']:6s} d={r['n_districts']} kappa={r['kappa']:.2f}  "
              + "  ".join(f"{k}={r[k]:.3f}" for k in POLICIES)
              + f"  pred_mis={r['pred_mis']:.3f}  A+B picked scg/mis="
              f"{r['picks_AB_scg']:.2f}/{r['picks_AB_scg_mis']:.2f}")

    plot(rows, str(ROOT / "figures" / "robust_dependence"))
    print(f"done in {time.time() - t_start:.1f}s -> results/robust_dependence.csv, "
          f"figures/robust_dependence.pdf")


if __name__ == "__main__":
    main()
