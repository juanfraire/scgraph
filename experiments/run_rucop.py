"""BRUF / RUCoP against the stochastic contact graph (revision T1, R2.4).

The uncertainty-aware DTN line (BRUF, RUCoP) computes optimal contingent
policies on a slotted plan with independent per-slot contact successes.
scgraph/routing/rucop.py reimplements both on the eq. (14) projection of a
stochastic contact graph. This runner puts them next to the paper's routers,
with every policy executed on the SAME held-out realizations of the full laws
(paired), never on draws used to build it.

  (a) UC1, timing uncertainty, sigma in {15, 30, 60} s:
        det          deterministic CGR on the mean plan          static
        pcgr         probabilistic CGR (Walter-style)             static
        bern@D       Bernoulli max-product path at slot D         static
        sla          SLA-aware route, argmax P[T <= D] (eq. 6)    static
        bruf@D       BRUF-1 at slot D                             contingent
        sla_cont     SLA-aware ranked table (plan_proxy)          contingent
      Metrics: SLA-violation rate with exact 95% Clopper-Pearson intervals,
      mean delivery time of on-time deliveries.
  (b) UC3, existence + correlated collapse, rho in {0, ..., 0.9}, N = 2:
        naive, pcgr, bern (Bernoulli-select), scg (correlation-aware greedy)
                                                                  static sets
        rucop2@30, rucop2@10, bruf1@30 (independent marginals)    contingent
        rucop2_zor   RUCoP-2 solved on the realized latent state  oracle
        oracle       CGR on each realization                      omniscient
  (c) Tractability: states, transitions and compute time of BRUF-1 and
      RUCoP-N (N = 1, 2, 3) against the slot width, on UC1, UC3 and a
      2-hour multi-hop plan on the 48-satellite CPD skeleton, next to the
      SCG per-route scoring cost (route_stats at M = 2000).

Usage: python experiments/run_rucop.py [--quick] [--part uc1|uc3|tract] [--replot]
Outputs: results/rucop_uc1.csv, results/rucop_uc3.csv,
         results/rucop_tractability.csv,
         figures/rucop_uc1.{pdf,png}, figures/rucop_uc3.{pdf,png},
         figures/rucop_tractability.{pdf,png}
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap
from _style import COL_BASE1, COL_BASE2, COL_BOUND, COL_SCG

from scgraph.arrival import delivery_time
from scgraph.graph import StochasticContactGraph
from scgraph.routing.bernoulli import bernoulli_route, bernoulli_select
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.placement import execute as placement_execute
from scgraph.routing.placement import plan_proxy
from scgraph.routing.quantile import candidate_routes, pick_route, route_stats
from scgraph.routing.replication import greedy_select
from scgraph.routing.rucop import (
    BudgetExceeded,
    ExecStats,
    bruf,
    conditioned_graph,
    execute_policy,
    execute_route,
    execute_route_set,
    pcgr_select,
    project,
    realize_with_state,
    rucop,
)
from scgraph.scenarios.uc1_event import EVENT_SITE, GATEWAY, make_instance_cpd
from scgraph.scenarios.uc3_emergency import make_instance as make_uc3
from scgraph.stats import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013

UC1_SIGMAS = [15.0, 30.0, 60.0]
UC1_SLOTS = [10.0, 30.0, 60.0]
UC3_RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
UC3_SLOT, UC3_SLOT_FINE = 30.0, 10.0
N_COPIES = 2
M_PROJ = 4000            # draws per contact for the q projection
M_SCORE = 2000           # draws per candidate for SLA-aware scoring

COL_MDP = "#1baf7a"      # BRUF / RUCoP family (palette slot 3, aqua)
COL_BERN = "#7b4fa8"     # Bernoulli TEG corner, as in uc1_h1 / uc3_h3
MDP_SHADES = {1: "#7fd3b2", 2: "#1baf7a", 3: "#0b6a49", 4: "#053a27"}   # N = 1..4


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _acc():
    return {"n": 0, "fail": 0, "n_ontime": 0, "lat_ontime": 0.0,
            "n_all": 0, "lat_all": 0.0, "recovered": 0}


def _add(acc, arr, t0, D, failures=0):
    acc["n"] += 1
    if arr <= D:
        acc["n_ontime"] += 1
        acc["lat_ontime"] += arr - t0
        if failures:
            acc["recovered"] += 1
    else:
        acc["fail"] += 1
    if math.isfinite(arr):
        acc["n_all"] += 1
        acc["lat_all"] += arr - t0


def _cids(route):
    return [c.cid for c in route] if route else []


def _layers(graph, cids):
    return "+".join(sorted({graph.by_id(c).layer for c in cids})) if cids else "none"


def _ci(xs):
    xs = np.asarray(xs, dtype=float)
    return float(1.96 * xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0


def _write(name, rows):
    (ROOT / "results").mkdir(exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(ROOT / "results" / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _read(name):
    out = []
    with open(ROOT / "results" / name) as f:
        for row in csv.DictReader(f):
            r = {}
            for k, v in row.items():
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = v
            out.append(r)
    return out


# --------------------------------------------------------------------------
# (a) UC1
# --------------------------------------------------------------------------

UC1_STATIC = ["det", "pcgr"] + [f"bern@{int(d)}" for d in UC1_SLOTS] + ["sla"]
UC1_CONT = [f"bruf@{int(d)}" for d in UC1_SLOTS] + ["sla_cont"]
UC1_SENS = [f"bruf_slot@{int(d)}" for d in UC1_SLOTS]     # slot-locked execution
UC1_POLICIES = UC1_STATIC + UC1_CONT + UC1_SENS


def uc1_task(task, rng):
    sigma, _i, m_eval = task
    inst_rng, plan_rng, eval_rng = rng.spawn(3)
    g, p = make_instance_cpd(inst_rng, sigma)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    gw = g.window(t0, D)

    routes, policies, meta = {}, {}, {}
    tic = time.perf_counter()
    det, _ = cgr_route(gw.mean_plan(), src, dst, size, t0)
    routes["det"] = _cids(det)
    meta["plan_s:det"] = time.perf_counter() - tic

    tic = time.perf_counter()
    cands = candidate_routes(gw, src, dst, size, t0, plan_rng, n_realizations=40)
    t_cands = time.perf_counter() - tic
    tic = time.perf_counter()
    routes["pcgr"] = _cids((pcgr_select(gw, cands, size, t0, budget=1) or [None])[0])
    meta["plan_s:pcgr"] = t_cands + time.perf_counter() - tic
    tic = time.perf_counter()
    sla, _st = pick_route(gw, src, dst, size, t0, D, plan_rng, m=M_SCORE, cands=cands)
    routes["sla"] = _cids(sla)
    meta["plan_s:sla"] = t_cands + time.perf_counter() - tic

    tic = time.perf_counter()
    proxy = plan_proxy(gw, src, dst, size, t0, D, plan_rng, m=M_SCORE, n_realizations=40)
    meta["plan_s:sla_cont"] = time.perf_counter() - tic
    proxy_choose = proxy.chooser()

    for dt in UC1_SLOTS:
        model = project(gw, src, dst, size, t0, D, dt, plan_rng, m=M_PROJ)
        tic = time.perf_counter()
        r, p_model = bernoulli_route(gw, src, dst, size, t0, D, dt, plan_rng, q=model.q)
        routes[f"bern@{int(dt)}"] = _cids(r)
        meta[f"pred:bern@{int(dt)}"] = p_model
        meta[f"plan_s:bern@{int(dt)}"] = model.project_seconds + time.perf_counter() - tic
        pol = bruf(model)
        policies[f"bruf@{int(dt)}"] = pol
        meta[f"pred:bruf@{int(dt)}"] = pol.value
        meta[f"plan_s:bruf@{int(dt)}"] = model.project_seconds + pol.seconds
        meta[f"states:bruf@{int(dt)}"] = pol.n_states
    for k, cids in routes.items():
        meta[f"layer:{k}"] = _layers(gw, cids)

    acc = {k: _acc() for k in UC1_POLICIES}
    for _ in range(m_eval):
        real = gw.realize(eval_rng)
        by_cid = {c.cid: c for c in real.contacts}
        for k, cids in routes.items():
            _add(acc[k], execute_route(by_cid, cids, size, t0), t0, D)
        for k, pol in policies.items():
            st = ExecStats()
            _add(acc[k], execute_policy(pol, by_cid, size, D, stats=st), t0, D, st.failures)
            st = ExecStats()
            _add(acc[k.replace("bruf@", "bruf_slot@")],
                 execute_policy(pol, by_cid, size, D, mode="slot", stats=st), t0, D, st.failures)
        _add(acc["sla_cont"], placement_execute(real, src, dst, size, t0, proxy_choose), t0, D)
    return {"sigma": sigma, "acc": acc, "meta": meta}


def run_uc1(quick, workers):
    R = 6 if quick else 40
    m_eval = 200 if quick else 2000
    tasks = [(s, i, m_eval) for s in UC1_SIGMAS for i in range(R)]
    t = time.time()
    res = pmap(uc1_task, tasks, seed=SEED, workers=workers)
    print(f"UC1: {len(tasks)} instances in {time.time() - t:.0f}s")
    return aggregate_uc1(res, R, m_eval)


def aggregate_uc1(res, R, m_eval):
    rows = []
    for sigma in UC1_SIGMAS:
        part = [r for r in res if r["sigma"] == sigma]
        for k in UC1_POLICIES:
            fails = sum(r["acc"][k]["fail"] for r in part)
            n = sum(r["acc"][k]["n"] for r in part)
            n_on = sum(r["acc"][k]["n_ontime"] for r in part)
            n_all = sum(r["acc"][k]["n_all"] for r in part)
            lo, hi = clopper_pearson(fails, n)
            per = [r["acc"][k]["fail"] / r["acc"][k]["n"] for r in part]
            kk = k.replace("bruf_slot@", "bruf@")
            preds = [r["meta"].get(f"pred:{kk}") for r in part if f"pred:{kk}" in r["meta"]]
            states = [r["meta"].get(f"states:{kk}") for r in part if f"states:{kk}" in r["meta"]]
            layers = [r["meta"].get(f"layer:{k}") for r in part if f"layer:{k}" in r["meta"]]
            rows.append({
                "kind": "policy", "sigma": sigma, "policy": k,
                "family": ("contingent" if k in UC1_CONT else
                           "sensitivity" if k in UC1_SENS else "static"),
                "slot": float(k.split("@")[1]) if "@" in k else math.nan,
                "viol": fails / n, "viol_lo": lo, "viol_hi": hi,
                "fails": fails, "n": n,
                "viol_inst_mean": float(np.mean(per)), "viol_inst_ci": _ci(per),
                "lat_ontime": sum(r["acc"][k]["lat_ontime"] for r in part) / n_on if n_on else math.nan,
                "lat_all": sum(r["acc"][k]["lat_all"] for r in part) / n_all if n_all else math.nan,
                "recovered_frac": sum(r["acc"][k]["recovered"] for r in part) / n,
                "model_pred_mean": float(np.mean(preds)) if preds else math.nan,
                "states_mean": float(np.mean(states)) if states else math.nan,
                "plan_s_mean": float(np.mean([r["meta"][f"plan_s:{kk}"] for r in part])),
                "frac_air_route": (sum(1 for x in layers if "air" in x) / len(layers)) if layers else math.nan,
                "R": len(part), "M_eval": m_eval,
            })
        # paired per-instance contrasts (difference in violation rate)
        pairs = [(f"bruf@{int(d)}", f"bern@{int(d)}") for d in UC1_SLOTS]
        pairs += [(f"bruf_slot@{int(d)}", f"bruf@{int(d)}") for d in UC1_SLOTS]
        pairs += [("sla_cont", "sla"), ("bruf@10", "sla"), ("bruf@30", "sla"),
                  ("bruf@60", "sla"), ("bern@30", "sla"), ("det", "sla")]
        for a, b in pairs:
            d = [r["acc"][a]["fail"] / r["acc"][a]["n"] - r["acc"][b]["fail"] / r["acc"][b]["n"]
                 for r in part]
            rows.append({"kind": "contrast", "sigma": sigma, "policy": f"{a}-{b}",
                         "viol": float(np.mean(d)), "viol_inst_ci": _ci(d),
                         "R": len(part), "M_eval": m_eval})
        line = "  ".join(
            f"{r['policy']}={r['viol']:.4f}" for r in rows
            if r["kind"] == "policy" and r["sigma"] == sigma
        )
        print(f"sigma={sigma:4.0f}  {line}")
    return rows


UC1_LABEL = {
    "det": "deterministic CGR (mean plan)",
    "pcgr": "probabilistic CGR",
    "bern@60": r"Bernoulli path, $\Delta$=60 s",
    "bern@30": r"Bernoulli path, $\Delta$=30 s",
    "bern@10": r"Bernoulli path, $\Delta$=10 s",
    "sla": "SLA-aware route (SCG)",
    "bruf@60": r"BRUF-1, $\Delta$=60 s",
    "bruf@30": r"BRUF-1, $\Delta$=30 s",
    "bruf@10": r"BRUF-1, $\Delta$=10 s",
    "sla_cont": "SLA-aware ranked table (SCG)",
}


def _uc1_color(k):
    if k == "det":
        return COL_BASE1, "s", True
    if k == "pcgr":
        return COL_BOUND, "D", True
    if k.startswith("bern"):
        return COL_BERN, "^", True
    if k.startswith("bruf"):
        return COL_MDP, "o", True
    if k == "sla":
        return COL_SCG, "o", True
    return COL_SCG, "o", False          # sla_cont: hollow


def plot_uc1(rows):
    import matplotlib.pyplot as plt

    order = ["det", "pcgr", "bern@60", "bern@30", "bern@10", "sla",
             "bruf@60", "bruf@30", "bruf@10", "sla_cont"]
    pol = [r for r in rows if r["kind"] == "policy"]
    fig, axes = plt.subplots(2, len(UC1_SIGMAS), figsize=(7.2, 4.6), sharey=True)
    ys = {k: len(order) - 1 - i for i, k in enumerate(order)}
    for j, sigma in enumerate(UC1_SIGMAS):
        ax_v, ax_l = axes[0, j], axes[1, j]
        for k in order:
            r = next(x for x in pol if x["sigma"] == sigma and x["policy"] == k)
            col, mk, filled = _uc1_color(k)
            y = ys[k]
            ax_v.errorbar(r["viol"], y, xerr=[[r["viol"] - r["viol_lo"]], [r["viol_hi"] - r["viol"]]],
                          fmt=mk, color=col, mfc=col if filled else "white", mec=col,
                          ms=4.5, capsize=2, lw=1.2)
            if math.isfinite(r["lat_ontime"]):
                ax_l.plot(r["lat_ontime"], y, mk, color=col, mfc=col if filled else "white",
                          mec=col, ms=4.5)
        for ax in (ax_v, ax_l):
            ax.axhline(len(order) - 6.5, color="#bbbbbb", lw=0.6)
            ax.set_yticks([ys[k] for k in order])
            ax.set_yticklabels([UC1_LABEL[k] for k in order])
            ax.grid(axis="y", visible=False)
        ax_v.set_title(rf"$\sigma$ = {sigma:.0f} s", fontsize=8)
        xmax = max(0.01, max(x["viol_hi"] for x in pol if x["sigma"] == sigma
                             and x["policy"] in order))
        ax_v.set_xlim(-0.03 * xmax, 1.05 * xmax)
        ax_v.set_xlabel("SLA-violation rate (95% CP)")
        ax_l.set_xlabel("mean delivery time,\non-time bundles [s]")
    for ax in axes[:1, -1]:
        sep = len(order) - 6.5
        ax.text(0.98, sep + 0.15, "static", fontsize=6.5, color="#52514e", ha="right",
                va="bottom", transform=ax.get_yaxis_transform())
        ax.text(0.98, sep - 0.15, "contingent", fontsize=6.5, color="#52514e", ha="right",
                va="top", transform=ax.get_yaxis_transform())
    _style.save(fig, str(ROOT / "figures" / "rucop_uc1"))


# --------------------------------------------------------------------------
# (b) UC3
# --------------------------------------------------------------------------

UC3_STATIC = ["naive", "pcgr", "bern", "scg"]
UC3_CONT = ["bruf1@30", "rucop2@30", "rucop2@10", "sla_cont1"]
UC3_REF = ["rucop2_zor@30", "oracle"]
UC3_SENS = ["rucop2_slot@30"]                  # slot-locked execution
UC3_POLICIES = UC3_STATIC + UC3_CONT + UC3_REF + UC3_SENS


def uc3_task(task, rng):
    rho, _i, m_eval = task
    inst_rng, plan_rng, eval_rng = rng.spawn(3)
    g, p = make_uc3(inst_rng, rho)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]

    sets, policies, meta = {}, {}, {}
    cands = candidate_routes(g, src, dst, size, t0, plan_rng, n_realizations=30)
    nominal = lambda r: delivery_time([c.mean_det() for c in r], size, t0)  # noqa: E731
    sets["naive"] = [_cids(r) for r in sorted(cands, key=nominal)[:N_COPIES]]
    sets["pcgr"] = [_cids(r) for r in pcgr_select(g, cands, size, t0, budget=N_COPIES)]
    sets["scg"] = [_cids(r) for r in greedy_select(cands, N_COPIES, size, t0, D, M_SCORE,
                                                   plan_rng, latent=g.latent)] if cands else []
    sets["bern"] = [_cids(r) for r in bernoulli_select(g, src, dst, size, t0, D, UC3_SLOT,
                                                      N_COPIES, plan_rng, m=M_PROJ)]

    proxy = plan_proxy(g, src, dst, size, t0, D, plan_rng, m=M_SCORE, n_realizations=30)
    proxy_choose = proxy.chooser()
    m30 = project(g, src, dst, size, t0, D, UC3_SLOT, plan_rng, m=M_PROJ)
    m10 = project(g, src, dst, size, t0, D, UC3_SLOT_FINE, plan_rng, m=M_PROJ)
    policies["bruf1@30"] = bruf(m30)
    policies["rucop2@30"] = rucop(m30, N_COPIES)
    policies["rucop2@10"] = rucop(m10, N_COPIES)
    zpol = []
    for z in range(len(g.latent.states)):
        mz = project(conditioned_graph(g, z), src, dst, size, t0, D, UC3_SLOT, plan_rng, m=M_PROJ)
        zpol.append(rucop(mz, N_COPIES))
    for k, pol in policies.items():
        meta[f"pred:{k}"] = pol.value
        meta[f"states:{k}"] = pol.n_states
    meta["pred:rucop2_zor@30"] = float(sum(pz * zp.value for (pz, _), zp in zip(g.latent.states, zpol)))

    acc = {k: _acc() for k in UC3_POLICIES}
    for _ in range(m_eval):
        real, z = realize_with_state(g, eval_rng)
        by_cid = {c.cid: c for c in real.contacts}
        for k, rs in sets.items():
            _add(acc[k], execute_route_set(by_cid, rs, size, t0), t0, D)
        for k, pol in policies.items():
            st = ExecStats()
            _add(acc[k], execute_policy(pol, by_cid, size, D, stats=st), t0, D, st.failures)
        st = ExecStats()
        _add(acc["rucop2_slot@30"], execute_policy(policies["rucop2@30"], by_cid, size, D,
                                                   mode="slot", stats=st), t0, D, st.failures)
        st = ExecStats()
        _add(acc["rucop2_zor@30"], execute_policy(zpol[z], by_cid, size, D, stats=st), t0, D,
             st.failures)
        _add(acc["sla_cont1"], placement_execute(real, src, dst, size, t0, proxy_choose), t0, D)
        _, arr = cgr_route(real, src, dst, size, t0)
        _add(acc["oracle"], arr, t0, D)
    return {"rho": rho, "acc": acc, "meta": meta}


def run_uc3(quick, workers):
    R = 6 if quick else 40
    m_eval = 200 if quick else 2000
    rhos = [0.0, 0.5, 0.9] if quick else UC3_RHOS
    tasks = [(r, i, m_eval) for r in rhos for i in range(R)]
    t = time.time()
    res = pmap(uc3_task, tasks, seed=SEED + 1, workers=workers)
    print(f"UC3: {len(tasks)} instances in {time.time() - t:.0f}s")
    return aggregate_uc3(res, rhos, m_eval)


def aggregate_uc3(res, rhos, m_eval):
    rows = []
    for rho in rhos:
        part = [r for r in res if r["rho"] == rho]
        for k in UC3_POLICIES:
            ok = sum(r["acc"][k]["n_ontime"] for r in part)
            n = sum(r["acc"][k]["n"] for r in part)
            lo, hi = clopper_pearson(ok, n)
            per = [r["acc"][k]["n_ontime"] / r["acc"][k]["n"] for r in part]
            kk = k.replace("_slot@", "@")
            preds = [r["meta"][f"pred:{kk}"] for r in part if f"pred:{kk}" in r["meta"]]
            states = [r["meta"][f"states:{kk}"] for r in part if f"states:{kk}" in r["meta"]]
            rows.append({
                "kind": "policy", "rho": rho, "policy": k,
                "family": ("static" if k in UC3_STATIC else
                           "contingent" if k in UC3_CONT else
                           "sensitivity" if k in UC3_SENS else "reference"),
                "delivery": ok / n, "lo": lo, "hi": hi, "k": ok, "n": n,
                "inst_mean": float(np.mean(per)), "inst_ci": _ci(per),
                "recovered_frac": sum(r["acc"][k]["recovered"] for r in part) / n,
                "model_pred_mean": float(np.mean(preds)) if preds else math.nan,
                "states_mean": float(np.mean(states)) if states else math.nan,
                "R": len(part), "M_eval": m_eval,
            })
        pairs = [("rucop2@30", "scg"), ("rucop2@30", "bern"), ("scg", "bern"),
                 ("rucop2@10", "rucop2@30"), ("rucop2_zor@30", "rucop2@30"),
                 ("rucop2@30", "bruf1@30"), ("pcgr", "bern"), ("rucop2_slot@30", "rucop2@30"),
                 ("rucop2@30", "naive"), ("sla_cont1", "bruf1@30")]
        for a, b in pairs:
            d = [r["acc"][a]["n_ontime"] / r["acc"][a]["n"] - r["acc"][b]["n_ontime"] / r["acc"][b]["n"]
                 for r in part]
            rows.append({"kind": "contrast", "rho": rho, "policy": f"{a}-{b}",
                         "delivery": float(np.mean(d)), "inst_ci": _ci(d),
                         "R": len(part), "M_eval": m_eval})
        line = "  ".join(f"{r['policy']}={r['delivery']:.3f}" for r in rows
                         if r["kind"] == "policy" and r["rho"] == rho)
        print(f"rho={rho:.1f}  {line}")
    return rows


def plot_uc3(rows):
    import matplotlib.pyplot as plt

    pol = [r for r in rows if r["kind"] == "policy"]
    rhos = sorted({r["rho"] for r in pol})

    def series(k, key="delivery"):
        return [next(r for r in pol if r["rho"] == x and r["policy"] == k)[key] for x in rhos]

    def band(k):
        lo, hi, mid = series(k, "lo"), series(k, "hi"), series(k)
        return [m - a for m, a in zip(mid, lo)], [b - m for m, b in zip(mid, hi)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    ax1.plot(rhos, series("oracle"), label="oracle (sees the realization)", color=COL_BOUND,
             linestyle=":", marker="")
    ax1.errorbar(rhos, series("rucop2@30"), yerr=band("rucop2@30"), capsize=2,
                 label=r"RUCoP-2, $\Delta$=30 s (contingent)", color=COL_MDP, marker="o")
    ax1.errorbar(rhos, series("scg"), yerr=band("scg"), capsize=2,
                 label="SCG, correlation-aware (static)", color=COL_SCG, marker="o",
                 linestyle="-")
    ax1.errorbar(rhos, series("bern"), yerr=band("bern"), capsize=2,
                 label="Bernoulli-select (static)", color=COL_BERN, linestyle="-.", marker="^")
    ax1.errorbar(rhos, series("naive"), yerr=band("naive"), capsize=2,
                 label="naive, nominal-fastest (static)", color=COL_BASE2, linestyle="--",
                 marker="s")
    ax1.set_xlabel(r"collapse severity $\rho$")
    ax1.set_ylabel("delivery ratio by deadline (95% CP)")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("N = 2 copies, held-out joint realizations", fontsize=8)
    ax1.legend(loc="lower left", handlelength=1.8, labelspacing=0.3)

    ax2.plot(rhos, series("rucop2@30", "model_pred_mean"), color=COL_MDP, linestyle="--",
             marker="o", mfc="white", label=r"RUCoP-2 $\Delta$=30 s, model's own value")
    ax2.plot(rhos, series("rucop2@30"), color=COL_MDP, marker="o",
             label=r"RUCoP-2 $\Delta$=30 s, realized")
    ax2.plot(rhos, series("rucop2@10"), color=MDP_SHADES[3], marker="^", linestyle="-.",
             label=r"RUCoP-2 $\Delta$=10 s, realized")
    ax2.plot(rhos, series("bruf1@30"), color=MDP_SHADES[1], marker="v", linestyle=":",
             label=r"BRUF-1 $\Delta$=30 s, realized")
    ax2.plot(rhos, series("rucop2_zor@30"), color=COL_BOUND, marker="D", linestyle="--",
             ms=3, label="RUCoP-2 told the latent state")
    ax2.plot(rhos, series("sla_cont1"), color=COL_SCG, marker="o", mfc="white",
             linestyle="-", label="SCG ranked table, 1 copy, realized")
    ax2.set_xlabel(r"collapse severity $\rho$")
    ax2.set_title("contingent policies: model belief vs outcome", fontsize=8)
    ax2.legend(loc="lower left", handlelength=1.8, labelspacing=0.3)
    _style.save(fig, str(ROOT / "figures" / "rucop_uc3"))


# --------------------------------------------------------------------------
# (c) tractability
# --------------------------------------------------------------------------

TRACT_SLOTS = {"uc1": [120, 60, 30, 15, 10, 5, 2, 1],
               "uc3": [120, 60, 30, 15, 10, 5, 2, 1],
               "multihop": [240, 120, 60, 30, 15, 10]}
TRACT_COPIES = {"uc1": (1, 2, 3), "uc3": (1, 2, 3), "multihop": (1, 2, 3, 4)}
FAMILY_LABEL = {"uc1": "UC1 (CPD skeleton, 15 nodes)",
                "uc3": "UC3 (D2D, 8 nodes)",
                "multihop": "multi-hop, 48 sats, 2 h horizon"}


def multihop_instance(t0: float, horizon: float = 7200.0, p_exist: float = 0.8):
    """RUCoP-style benchmark on the 48-satellite CPD plan: Dirac timing,
    independent contact failure, a 2-hour horizon (as in the RUCoP ring-road
    plans of 1 to 3 hours)."""
    from scgraph.io import load_cpd_plan

    sk = load_cpd_plan(ROOT / "scenarios" / "scale_48")
    cs = [c.with_laws(p_exist=p_exist) for c in sk.contacts]
    g = StochasticContactGraph(num_nodes=sk.num_nodes, contacts=cs, horizon=sk.horizon)
    p = {"src": sk.node(EVENT_SITE), "dst": sk.node(GATEWAY), "size": 100.0,
         "t0": t0, "deadline": t0 + horizon}
    return g, p


def _tract_instances(quick):
    n = 1 if quick else 3
    out = []
    for i in range(n):
        g, p = make_instance_cpd(np.random.default_rng(SEED + 100 + i), 30.0)
        out.append(("uc1", i, g.window(p["t0"], p["deadline"]), p))
    for i in range(n):
        g, p = make_uc3(np.random.default_rng(SEED + 200 + i), 0.5)
        out.append(("uc3", i, g, p))
    for i, t0 in enumerate([2000.0, 7000.0, 12000.0][:n]):
        g, p = multihop_instance(t0)
        out.append(("multihop", i, g.window(p["t0"], p["deadline"]), p))
    return out


def run_tract(quick):
    budget = 20.0 if quick else 300.0
    max_states = 1_000_000 if quick else 6_000_000
    m_proj = 500 if quick else 2000
    rows = []
    for fam, i, g, p in _tract_instances(quick):
        src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
        nodes = len({c.frm for c in g.contacts} | {c.to for c in g.contacts})
        base = {"family": fam, "instance": i, "contacts": len(g.contacts), "nodes": nodes,
                "horizon_s": D - t0}
        rng = np.random.default_rng(SEED + 300 + i)

        # SCG: candidate generation once, then per-route scoring at M = 2000
        tic = time.perf_counter()
        cands = candidate_routes(g, src, dst, size, t0, rng, n_realizations=40)
        t_gen = time.perf_counter() - tic
        per_route = []
        for r in cands:
            tic = time.perf_counter()
            route_stats(r, size, t0, D, M_SCORE, rng, latent=g.latent)
            per_route.append(time.perf_counter() - tic)
        rows.append({**base, "method": "scg_route_stats", "slot": math.nan,
                     "seconds": statistics.median(per_route) if per_route else math.nan,
                     "candidates": len(cands)})
        rows.append({**base, "method": "scg_candidates", "slot": math.nan,
                     "seconds": t_gen, "candidates": len(cands)})
        rows.append({**base, "method": "scg_total", "slot": math.nan,
                     "seconds": t_gen + sum(per_route), "candidates": len(cands)})
        print(f"[{fam} #{i}] contacts={len(g.contacts)} nodes={nodes} H={D - t0:.0f}s "
              f"scg: gen={t_gen:.3f}s route_stats={statistics.median(per_route) * 1e3:.2f}ms "
              f"x {len(cands)}", flush=True)

        censored = set()
        for dt in TRACT_SLOTS[fam]:
            model = project(g, src, dst, size, t0, D, float(dt), rng, m=m_proj)
            row0 = {**base, "slot": float(dt), "n_slots": model.n_slots}
            rows.append({**row0, "method": "projection", "seconds": model.project_seconds})
            jobs = [("bruf1", 1, True)] + [(f"rucop{n}", n, False) for n in TRACT_COPIES[fam]]
            msg = []
            for name, n, complete in jobs:
                if name in censored:
                    rows.append({**row0, "method": name, "censored": 1, "skipped": 1})
                    continue
                try:
                    pol = (bruf(model, keep_table=False) if complete else
                           rucop(model, n, keep_table=False, max_seconds=budget,
                                 max_states=max_states))
                    rows.append({**row0, "method": name, "states": pol.n_states,
                                 "transitions": pol.n_transitions, "actions": pol.n_actions,
                                 "seconds": pol.seconds, "value": pol.value, "censored": 0})
                    msg.append(f"{name}: {pol.n_states} st {pol.n_transitions} tr "
                               f"{pol.seconds:.2f}s v={pol.value:.3f}")
                except BudgetExceeded as e:
                    censored.add(name)
                    rows.append({**row0, "method": name, "states": e.stats["n_states"],
                                 "transitions": e.stats["n_transitions"],
                                 "seconds": e.stats["seconds"], "censored": 1})
                    msg.append(f"{name}: BUDGET ({e.stats['n_states']} st, "
                               f"{e.stats['seconds']:.0f}s)")
            print(f"   dt={dt:4d} K={model.n_slots:4d} proj={model.project_seconds:.2f}s  "
                  + " | ".join(msg), flush=True)
    return rows


def plot_tract(rows):
    import matplotlib.pyplot as plt

    fams = ["uc1", "uc3", "multihop"]
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    # N = 1 is drawn from the pruned solve, so every line counts the same
    # thing (forward-reachable states); BRUF's complete table is in the CSV
    methods = [("rucop1", 1, "BRUF / RUCoP-1", "o", "-"), ("rucop2", 2, "RUCoP-2", "s", "--"),
               ("rucop3", 3, "RUCoP-3", "^", "-."), ("rucop4", 4, "RUCoP-4", "D", ":")]
    for j, fam in enumerate(fams):
        ax_t, ax_s = axes[0, j], axes[1, j]
        fr = [r for r in rows if r["family"] == fam]
        slots = sorted({r["slot"] for r in fr if isinstance(r["slot"], float)
                        and math.isfinite(r["slot"])}, reverse=True)
        for name, n, lab, mk, ls in methods:
            xs, ts, ss, cx, cy = [], [], [], [], []
            for dt in slots:
                pts = [r for r in fr if r["method"] == name and r["slot"] == dt]
                done = [r for r in pts if r.get("censored") == 0.0]
                if done and len(done) == len(pts):
                    xs.append(dt)
                    ts.append(statistics.median(r["seconds"] for r in done))
                    ss.append(statistics.median(r["states"] for r in done))
                elif any(r.get("censored") == 1.0 and r.get("skipped") != 1.0 for r in pts):
                    cx.append(dt)
                    cy.append(max(r["seconds"] for r in pts if r.get("skipped") != 1.0))
            col = MDP_SHADES[n]
            if not xs and not cx:
                continue
            ax_t.plot(xs, ts, marker=mk, linestyle=ls, color=col, label=lab)
            ax_s.plot(xs, ss, marker=mk, linestyle=ls, color=col, label=lab)
            if cx:
                ax_t.plot(cx, cy, marker="x", linestyle="", color=col, ms=6, mew=1.5)
        proj = [(dt, statistics.median(r["seconds"] for r in fr
                                       if r["method"] == "projection" and r["slot"] == dt))
                for dt in slots]
        ax_t.plot([a for a, _ in proj], [b for _, b in proj], color=COL_BERN, linestyle=":",
                  marker=".", label="projection (eq. 14)")
        rs = statistics.median(r["seconds"] for r in fr if r["method"] == "scg_route_stats")
        tot = statistics.median(r["seconds"] for r in fr if r["method"] == "scg_total")
        ax_t.axhline(rs, color=COL_SCG, linestyle=":", lw=1.2, label="SCG, one route at M=2000")
        ax_t.axhline(tot, color=COL_SCG, linestyle="-", lw=1.2, label="SCG, full selection")
        for ax in (ax_t, ax_s):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.invert_xaxis()
            ax.set_xlabel(r"slot width $\Delta$ [s]")
        ax_t.set_title(FAMILY_LABEL[fam], fontsize=8)
        if j == 0:
            ax_t.set_ylabel("compute time [s]")
            ax_s.set_ylabel("(state, slot) pairs")
    handles, labels = [], []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    handles.append(Line2D([], [], marker="x", linestyle="", color="#52514e", mew=1.5))
    labels.append("budget hit (censored)")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5,
               handlelength=2.2, columnspacing=1.2, frameon=False)
    fig.tight_layout(pad=0.4, rect=(0, 0.09, 1, 1))
    fig.savefig(str(ROOT / "figures" / "rucop_tractability") + ".pdf")
    fig.savefig(str(ROOT / "figures" / "rucop_tractability") + ".png")


# --------------------------------------------------------------------------


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    part = None
    if "--part" in sys.argv:
        part = sys.argv[sys.argv.index("--part") + 1]
    workers = None
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    _style.setup()
    (ROOT / "figures").mkdir(exist_ok=True)
    t_all = time.time()

    if part in (None, "uc1"):
        t = time.time()
        rows = _read("rucop_uc1.csv") if replot else run_uc1(quick, workers)
        if not replot:
            _write("rucop_uc1.csv", rows)
        plot_uc1(rows)
        print(f"(a) UC1 done in {time.time() - t:.0f}s")
    if part in (None, "uc3"):
        t = time.time()
        rows = _read("rucop_uc3.csv") if replot else run_uc3(quick, workers)
        if not replot:
            _write("rucop_uc3.csv", rows)
        plot_uc3(rows)
        print(f"(b) UC3 done in {time.time() - t:.0f}s")
    if part in (None, "tract"):
        t = time.time()
        rows = _read("rucop_tractability.csv") if replot else run_tract(quick)
        if not replot:
            _write("rucop_tractability.csv", rows)
        plot_tract(rows)
        print(f"(c) tractability done in {time.time() - t:.0f}s")
    print(f"total {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
