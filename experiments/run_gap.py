"""Optimality gap of candidate generation (revision, R1/R2/R3).

The SLA-aware router picks argmax P[T <= D] over an enumerated candidate set
(Sec. V-C), because the quantile key has no optimal substructure. Reviewers
asked how often that set misses the best route, by how much, and what the
router does when no candidate meets the chance constraint. Per instance:

  (a) selection   `pick_route` over `candidate_routes` at n_realizations in
                  BUDGETS (0, 10, 40 = the paper's default, 160). The four
                  candidate sets are nested by construction: each is generated
                  from the same seed, and realizations are drawn sequentially.
                  Plus one generator variant, "40rc": the n = 40 set with
                  risk-class exclusion routes added (`risk_class_routes`),
                  measured here only; the library's generator is unchanged.
  (b) oracle      every node-simple route that the support envelope allows to
                  arrive by D (routing/enumerate.py), plus every candidate
                  route, scored on common random numbers through a three-stage
                  funnel: all routes on M_S1 draws, the best K2 on M_S2 fresh
                  draws, the best K3 on M_S3 fresh draws. Oracle = argmax P on
                  the last stage (ties: lower mean delivery, as pick_route).
                  Candidate routes and selected routes always reach the last
                  stage, so the oracle is never scored below them.
  (c) held-out    selected routes and the oracle on the SAME M_HO fresh paired
                  draws (`joint_route_set_deliveries`). Per budget: exact
                  identity (same contact ids), statistical tie (exact McNemar
                  test on the discordant pairs, not significant at 95%), the
                  gap P_oracle - P_selected with its paired 95% CI, and whether
                  the oracle was in the candidate set at all.
  (d) infeasibility  `pick_route_chance` (admission="lcb", delta = 0.05,
                  draws sized by stats.admission_draws) at every budget and eps
                  in EPSILONS. When it reports feasible=False, does the
                  enumerated set hold a route that clears 1 - eps? Checked on
                  the last funnel stage (Clopper-Pearson lower bound,
                  Bonferroni over the stage) and on the held-out draws.

Diagnostics. The funnel rank of the final oracle at stages 1 and 2 (far from
the K cut means the cut did not decide the oracle). For UC4, where the hop
bound binds, the oracle restricted to routes of at most min_hops + s hops for
s in HOP_LEVELS, which shows whether a larger bound would still pay.

Scenarios. UC1 (`make_instance_cpd`) at sigma 20/40/60 s with the default
40 s margin, plus a contested block (sigma 40 s, margin -30 s: the backbone
misses the deadline). UC4 (`uc4_continuum.make_instance`) at sigma 40 s,
margin 0 and 60 s.

Usage: python experiments/run_gap.py [--quick] [--only uc1|uc4] [--replot]
Outputs: results/gap_uc1.csv, results/gap_uc4.csv          (one row per instance)
         results/gap_uc1_agg.csv, results/gap_uc4_agg.csv  (per setting and budget)
         figures/gap.{pdf,png}
--quick writes *_quick.csv and figures/gap_quick.* so it never clobbers a
full run. --replot redraws the figure from the CSVs.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap

from scgraph.arrival import joint_route_set_deliveries
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.enumerate import enumerate_routes, screen_routes
from scgraph.routing.quantile import candidate_routes, pick_route, pick_route_chance
from scgraph.scenarios import uc4_continuum
from scgraph.scenarios.uc1_event import make_instance_cpd
from scgraph.stats import admission_draws, cp_lower

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261014
BUDGETS = (0, 10, 40, 160)
# the budgets plus one generator variant: the n = 40 set augmented with
# risk-class exclusion routes (see `risk_class_routes`); not part of the library
VARIANTS = BUDGETS + ("40rc",)
EPSILONS = (0.01, 0.05, 0.10)
DELTA = 0.05
HOP_LEVELS = (0, 2, 4, 6, 8, 12, 16, 20)
MATERIAL = 0.01        # a gap below one point of probability is reported as immaterial

FULL = dict(m_sel=2000, m_s1=1000, k2=2000, k2_level=200, m_s2=5000,
            k3=200, k3_level=20, m_s3=20_000, m_ho=20_000)
QUICK = dict(m_sel=600, m_s1=300, k2=300, k2_level=50, m_s2=1500,
             k3=50, k3_level=10, m_s3=4000, m_ho=4000)
ENUM = {
    # UC1 is small enough to enumerate with no effective hop bound
    "uc1": dict(hop_slack=99, max_routes=200_000, max_seconds=120.0),
    "uc4": dict(hop_slack=24, max_routes=500_000, max_seconds=300.0),
}
ENUM_QUICK = {"uc1": ENUM["uc1"], "uc4": dict(hop_slack=12, max_routes=100_000, max_seconds=60.0)}

SETTINGS = {
    "uc1": [dict(sigma=s, margin=40.0, R=40) for s in (20.0, 40.0, 60.0)]
           + [dict(sigma=40.0, margin=-30.0, R=40)],
    "uc4": [dict(sigma=40.0, margin=m, R=40) for m in (0.0, 60.0)],
}


def key(route) -> tuple:
    return tuple(c.cid for c in route) if route else ()


def _best(idx, k, md):
    """Index in `idx` with the most successes, ties to the lower mean delivery."""
    idx = list(idx)
    if not idx:
        return None
    return min(idx, key=lambda i: (-int(k[i]), float(md[i])))


def _ranked(idx, k, md):
    idx = np.asarray(list(idx), dtype=int)
    if idx.size == 0:
        return idx
    order = np.lexsort((md[idx], -k[idx]))
    return idx[order]


def paired(o: np.ndarray, s: np.ndarray) -> dict:
    """Paired comparison of two on-time indicator vectors (oracle, selected)."""
    n = o.size
    b = int(np.count_nonzero(o & ~s))      # oracle on time, selected late
    c = int(np.count_nonzero(s & ~o))      # selected on time, oracle late
    gap = (b - c) / n
    d = o.astype(np.int8) - s.astype(np.int8)
    se = float(d.std(ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    pval = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"gap": gap, "lo": gap - 1.96 * se, "hi": gap + 1.96 * se, "p": pval, "b": b, "c": c}


def risk_class_routes(g, src, dst, size, t0):
    """Mean-plan routes with a whole class of uncertain contacts excluded.

    Classes: contacts whose existence is uncertain (p_exist < 1), contacts
    whose timing is random, and both. For each class, the mean-plan CGR route
    on the remaining contacts and its single-contact exclusions. This is the
    generic form of what makes UC1's backbone a candidate: the slow route that
    avoids the risky contacts altogether, even when it shares other contacts
    with the fast ones (where the disjoint sweep cannot reach it).
    """
    mean = g.mean_plan()
    unc_exist = frozenset(c.cid for c in g.contacts if c.p_exist < 1.0)
    unc_time = frozenset(c.cid for c in g.contacts
                         if not (c.start.is_point_mass and c.duration.is_point_mass))
    out = {}
    for excl in (unc_exist, unc_time, unc_exist | unc_time):
        r, _ = cgr_route(mean, src, dst, size, t0, excluded_cids=excl)
        if r is None:
            continue
        out.setdefault(key(r), [g.by_id(c.cid) for c in r])
        for c in r:
            alt, _ = cgr_route(mean, src, dst, size, t0, excluded_cids=excl | {c.cid})
            if alt is not None:
                out.setdefault(key(alt), [g.by_id(x.cid) for x in alt])
    return list(out.values())


def describe(g, route) -> str:
    """Link classes along a route (UC4 `kind`, else the layer), e.g. uav-uav-haps_up-..."""
    return "-".join((g.by_id(c).kind or g.by_id(c).layer) for c in route)


def n_uncertain(g, route) -> int:
    """Contacts on the route whose existence or timing is random."""
    return sum(1 for c in route
               if g.by_id(c).p_exist < 1.0 or not (g.by_id(c).start.is_point_mass
                                                    and g.by_id(c).duration.is_point_mass))


def make(task, rng):
    if task["scen"] == "uc1":
        return make_instance_cpd(rng, task["sigma"], margin=task["margin"])
    return uc4_continuum.make_instance(rng, task["sigma"], margin=task["margin"])


def one_instance(task, rng):
    quick = task["quick"]
    P = QUICK if quick else FULL
    r_inst, r_cand, r_sel, r_cc, r_s1, r_s2, r_s3, r_ho = rng.spawn(8)
    T = {}
    t_all = time.perf_counter()

    g, p = make(task, r_inst)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]

    # (a) nested candidate sets and the SLA-aware choice at each budget
    t = time.perf_counter()
    cand_seed = int(r_cand.integers(2**63))
    cands, t_gen, t_pick = {}, {}, {}
    for n in BUDGETS:
        t1 = time.perf_counter()
        cands[n] = candidate_routes(g, src, dst, size, t0, np.random.default_rng(cand_seed), n)
        t_gen[n] = time.perf_counter() - t1
    t1 = time.perf_counter()
    base = {key(r) for r in cands[40]}
    cands["40rc"] = cands[40] + [r for r in risk_class_routes(g, src, dst, size, t0)
                                 if key(r) not in base]
    t_gen["40rc"] = t_gen[40] + time.perf_counter() - t1
    T["cand"] = time.perf_counter() - t
    t = time.perf_counter()
    sel = {}
    for n in VARIANTS:
        t1 = time.perf_counter()
        r, _ = pick_route(g, src, dst, size, t0, D, r_sel, m=P["m_sel"], cands=cands[n])
        t_pick[n] = time.perf_counter() - t1
        sel[n] = key(r)
    T["sel"] = time.perf_counter() - t

    # (d) chance-constrained choice, lower-bound admission
    t = time.perf_counter()
    cc = {}
    for n in VARIANTS:
        for eps in EPSILONS:
            m = P["m_sel"] if quick else admission_draws(eps, len(cands[n]), DELTA)
            r, st = pick_route_chance(g, src, dst, size, t0, D, r_cc, m=m, eps=eps,
                                      admission="lcb", delta=DELTA, cands=cands[n])
            cc[(n, eps)] = (key(r), bool(st.get("feasible", False)))
    T["cc"] = time.perf_counter() - t

    # (b) the oracle: enumerate, then the screening funnel
    t = time.perf_counter()
    en = enumerate_routes(g, src, dst, size, t0, D,
                          **(ENUM_QUICK if quick else ENUM)[task["scen"]])
    T["enum"] = time.perf_counter() - t
    t = time.perf_counter()
    n_enum = len(en.routes)
    enum_set = set(en.routes)
    cand_keys = list(dict.fromkeys(key(r) for n in VARIANTS for r in cands[n]))
    extra = [ck for ck in cand_keys if ck and ck not in enum_set]
    pool = list(en.routes) + extra
    pos = {r: i for i, r in enumerate(pool)}
    hops = np.array([len(r) for r in pool], dtype=int)
    in_enum = np.arange(len(pool)) < n_enum
    always = {pos[ck] for ck in cand_keys if ck in pos}
    always |= {pos[s] for s in sel.values() if s in pos}
    always |= {pos[v[0]] for v in cc.values() if v[0] in pos}

    min_h = en.min_hops if en.min_hops is not None else 0
    levels = [(s, min(min_h + s, en.hop_bound)) for s in HOP_LEVELS]

    s1 = screen_routes(g, pool, size, t0, D, P["m_s1"], r_s1)
    rank1 = _ranked(range(len(pool)), s1["k"], s1["mean_delivery"])
    stage2 = set(rank1[:P["k2"]].tolist()) | always
    for _, b in levels:
        sub = rank1[(hops[rank1] <= b) & in_enum[rank1]]
        stage2 |= set(sub[:P["k2_level"]].tolist())
    idx2 = sorted(stage2)
    s2 = screen_routes(g, [pool[i] for i in idx2], size, t0, D, P["m_s2"], r_s2)
    k2 = np.full(len(pool), -1, dtype=np.int64)
    md2 = np.full(len(pool), math.inf)
    k2[idx2], md2[idx2] = s2["k"], s2["mean_delivery"]
    rank2 = _ranked(idx2, k2, md2)
    stage3 = set(rank2[:P["k3"]].tolist()) | always
    for _, b in levels:
        sub = rank2[(hops[rank2] <= b) & in_enum[rank2]]
        stage3 |= set(sub[:P["k3_level"]].tolist())
    idx3 = sorted(stage3)
    s3 = screen_routes(g, [pool[i] for i in idx3], size, t0, D, P["m_s3"], r_s3)
    k3 = np.full(len(pool), -1, dtype=np.int64)
    md3 = np.full(len(pool), math.inf)
    k3[idx3], md3[idx3] = s3["k"], s3["mean_delivery"]
    m3 = P["m_s3"]
    T["screen"] = time.perf_counter() - t

    o = _best(idx3, k3, md3)
    oracle = pool[o]
    p3 = lambda i: float(k3[i]) / m3 if i is not None and i in stage3 else 0.0
    r1pos = {int(i): j for j, i in enumerate(rank1)}
    r2pos = {int(i): j for j, i in enumerate(rank2)}

    out = {
        "scen": task["scen"], "sigma": task["sigma"], "margin": task["margin"], "inst": task["i"],
        "D_rel": D - t0,
        "n_enum": n_enum, "enum_capped": en.capped, "enum_timed_out": en.timed_out,
        "min_hops": en.min_hops if en.min_hops is not None else -1, "hop_bound": en.hop_bound,
        "hop_bound_free": en.hop_bound_free, "n_usable": en.n_usable,
        "n_pool": len(pool), "n_cand_outside_enum": len(extra),
        "n_stage2": len(idx2), "n_stage3": len(idx3),
        "oracle_hops": len(oracle), "oracle_slack": len(oracle) - min_h,
        "oracle_path": describe(g, oracle), "oracle_n_unc": n_uncertain(g, oracle),
        "oracle_from_enum": bool(in_enum[o]),
        "oracle_rank1": r1pos[o], "oracle_rank2": r2pos[o],
        "oracle_p3": p3(o),
        "p1_top": float(s1["p"][rank1[0]]) if len(rank1) else 0.0,
        "p1_cut": float(s1["p"][rank1[min(P["k2"], len(rank1)) - 1]]) if len(rank1) else 0.0,
    }
    for s, b in levels:
        sub = [i for i in idx3 if hops[i] <= b and in_enum[i]]
        out[f"orP_s{s}"] = p3(_best(sub, k3, md3))
    out["orP_enum"] = p3(_best([i for i in idx3 if in_enum[i]], k3, md3))

    # (c) held-out paired evaluation
    t = time.perf_counter()
    ho_keys = list(dict.fromkeys([oracle] + [sel[n] for n in VARIANTS]
                                 + [v[0] for v in cc.values()]))
    nonempty = [k_ for k_ in ho_keys if k_]
    ontime = {(): np.zeros(P["m_ho"], dtype=bool)}
    if nonempty:
        tt = joint_route_set_deliveries([[g.by_id(c) for c in k_] for k_ in nonempty],
                                        size, t0, P["m_ho"], r_ho, latent=g.latent)
        for k_, row in zip(nonempty, tt):
            ontime[k_] = row <= D
    T["ho"] = time.perf_counter() - t
    o_on = ontime[oracle]
    out["oracle_p_ho"] = float(o_on.mean())

    for n in VARIANTS:
        sk = sel[n]
        st = paired(o_on, ontime[sk])
        ident = sk == oracle
        tie = (not ident) and st["p"] >= 0.05
        cand_set = {key(r) for r in cands[n]}
        bc = _best([pos[k_] for k_ in cand_set if k_ in pos], k3, md3)
        out.update({
            f"n{n}_cands": len(cands[n]),
            f"n{n}_sel_hops": len(sk),
            f"n{n}_sel_path": describe(g, sk), f"n{n}_sel_n_unc": n_uncertain(g, sk),
            f"n{n}_t_router": t_gen[n] + t_pick[n],
            f"n{n}_p_ho": float(ontime[sk].mean()),
            f"n{n}_ident": ident, f"n{n}_tie": tie, f"n{n}_hit": ident or tie,
            f"n{n}_worse": (not ident) and st["p"] < 0.05 and st["gap"] > 0,
            f"n{n}_better": (not ident) and st["p"] < 0.05 and st["gap"] < 0,
            f"n{n}_gap": st["gap"], f"n{n}_gap_lo": st["lo"], f"n{n}_gap_hi": st["hi"],
            f"n{n}_pval": st["p"],
            f"n{n}_oracle_in": oracle in cand_set,
            f"n{n}_sel_in_enum": sk in enum_set,
            f"n{n}_p3_sel": p3(pos.get(sk)),
            f"n{n}_p3_bestcand": p3(bc),
            # generation gap and ranking gap, on the stage-3 draws
            f"n{n}_gen_gap": p3(o) - p3(bc),
            f"n{n}_rank_gap": p3(bc) - p3(pos.get(sk)),
        })
        for eps in EPSILONS:
            ck, feas = cc[(n, eps)]
            tag = f"n{n}_e{eps:g}"
            ok_c = [pos[k_] for k_ in cand_set if k_ in pos]
            lcb = lambda i: cp_lower(int(k3[i]), m3, DELTA / len(idx3))
            out.update({
                f"{tag}_feasible": feas,
                f"{tag}_p_ho": float(ontime[ck].mean()),
                f"{tag}_over": feas and float(ontime[ck].mean()) < 1.0 - eps,
                f"{tag}_is_oracle": ck == oracle,
                # is there a route that clears 1 - eps, on the stage-3 draws?
                f"{tag}_oracle_feas": lcb(o) >= 1.0 - eps,
                f"{tag}_oracle_feas_ho": out["oracle_p_ho"] >= 1.0 - eps,
                f"{tag}_cand_feas": any(lcb(i) >= 1.0 - eps for i in ok_c),
            })
    T["total"] = time.perf_counter() - t_all
    for k_, v in T.items():
        out[f"t_{k_}"] = v
    return out


# ---------------------------------------------------------------------------
# aggregation and output
# ---------------------------------------------------------------------------


def _mean(xs):
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else math.nan


def _ci(xs):
    xs = np.asarray([float(x) for x in xs])
    return 1.96 * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def aggregate(rows):
    out = []
    settings = sorted({(r["scen"], r["sigma"], r["margin"]) for r in rows},
                      key=lambda s: (s[0], -s[2], s[1]))
    for scen, sigma, margin in settings:
        sub = [r for r in rows if (r["scen"], r["sigma"], r["margin"]) == (scen, sigma, margin)]
        base = {
            "scen": scen, "sigma": sigma, "margin": margin, "R": len(sub),
            "n_enum_mean": _mean(r["n_enum"] for r in sub),
            "n_enum_median": float(np.median([r["n_enum"] for r in sub])),
            "n_enum_max": max(r["n_enum"] for r in sub),
            "enum_capped": sum(bool(r["enum_capped"]) for r in sub),
            "enum_timed_out": sum(bool(r["enum_timed_out"]) for r in sub),
            "hop_bound_free": sum(bool(r["hop_bound_free"]) for r in sub),
            "min_hops_mean": _mean(r["min_hops"] for r in sub),
            "hop_bound_mean": _mean(r["hop_bound"] for r in sub),
            "n_cand_outside_enum_mean": _mean(r["n_cand_outside_enum"] for r in sub),
            "oracle_hops_mean": _mean(r["oracle_hops"] for r in sub),
            "oracle_slack_max": max(r["oracle_slack"] for r in sub),
            "oracle_from_enum": _mean(r["oracle_from_enum"] for r in sub),
            "oracle_rank1_max": max(r["oracle_rank1"] for r in sub),
            "oracle_rank2_max": max(r["oracle_rank2"] for r in sub),
            "oracle_p_ho": _mean(r["oracle_p_ho"] for r in sub),
            "t_enum_mean": _mean(r["t_enum"] for r in sub),
            "t_screen_mean": _mean(r["t_screen"] for r in sub),
            "t_total_mean": _mean(r["t_total"] for r in sub),
        }
        for s in HOP_LEVELS:
            base[f"orP_s{s}_deficit"] = _mean(r["oracle_p3"] - r[f"orP_s{s}"] for r in sub)
        for n in VARIANTS:
            g = [r[f"n{n}_gap"] for r in sub]
            row = dict(base)
            row.update({
                "n_real": n,
                "n_cands_mean": _mean(r[f"n{n}_cands"] for r in sub),
                "hit": _mean(r[f"n{n}_hit"] for r in sub),
                "ident": _mean(r[f"n{n}_ident"] for r in sub),
                "tie": _mean(r[f"n{n}_tie"] for r in sub),
                "worse": _mean(r[f"n{n}_worse"] for r in sub),
                # significant AND larger than one point of probability
                "worse_material": _mean(r[f"n{n}_worse"] and r[f"n{n}_gap"] > MATERIAL
                                        for r in sub),
                "better": _mean(r[f"n{n}_better"] for r in sub),
                "oracle_in": _mean(r[f"n{n}_oracle_in"] for r in sub),
                "miss_gen": _mean((not r[f"n{n}_hit"]) and (not r[f"n{n}_oracle_in"]) for r in sub),
                "miss_rank": _mean((not r[f"n{n}_hit"]) and r[f"n{n}_oracle_in"] for r in sub),
                "gap_mean": _mean(g), "gap_ci": _ci(g),
                "gap_q50": float(np.quantile(g, 0.5)), "gap_q90": float(np.quantile(g, 0.9)),
                "gap_q95": float(np.quantile(g, 0.95)), "gap_max": float(np.max(g)),
                "gap_hi_max": max(r[f"n{n}_gap_hi"] for r in sub),
                "gen_gap_mean": _mean(r[f"n{n}_gen_gap"] for r in sub),
                "rank_gap_mean": _mean(r[f"n{n}_rank_gap"] for r in sub),
                "p_sel_ho": _mean(r[f"n{n}_p_ho"] for r in sub),
                "t_router_mean": _mean(r[f"n{n}_t_router"] for r in sub),
                "sel_in_enum": _mean(r[f"n{n}_sel_in_enum"] for r in sub),
            })
            for eps in EPSILONS:
                tag = f"n{n}_e{eps:g}"
                inf = [r for r in sub if not r[f"{tag}_feasible"]]
                fea = [r for r in sub if r[f"{tag}_feasible"]]
                e = f"e{eps:g}"
                row.update({
                    f"{e}_infeasible": len(inf) / len(sub),
                    f"{e}_n_infeasible": len(inf),
                    # among the router's infeasible instances
                    f"{e}_inf_oracle_feas": _mean(r[f"{tag}_oracle_feas"] for r in inf),
                    f"{e}_inf_oracle_feas_ho": _mean(r[f"{tag}_oracle_feas_ho"] for r in inf),
                    f"{e}_inf_cand_feas": _mean(r[f"{tag}_cand_feas"] for r in inf),
                    f"{e}_inf_fallback_is_oracle": _mean(r[f"{tag}_is_oracle"] for r in inf),
                    f"{e}_inf_fallback_p_ho": _mean(r[f"{tag}_p_ho"] for r in inf),
                    f"{e}_inf_oracle_p_ho": _mean(r["oracle_p_ho"] for r in inf),
                    # among the admitted ones: held-out success below 1 - eps
                    f"{e}_feas_over": _mean(r[f"{tag}_over"] for r in fea),
                    f"{e}_oracle_feas_all": _mean(r[f"{tag}_oracle_feas"] for r in sub),
                })
            out.append(row)
    return out


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read(path):
    def conv(v):
        if v in ("True", "False"):
            return v == "True"
        try:
            x = float(v)
            return int(x) if x.is_integer() and "." not in v and "e" not in v.lower() else x
        except ValueError:
            return v
    with open(path) as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


# Figure groups: UC1 at the default 40 s margin is pooled over sigma (the
# per-sigma rows are in the CSVs). Hue = scenario, style = margin regime.
GROUPS = [
    ("UC1, margin 40 s ($\\sigma$ 20-60 s)", lambda r: r["scen"] == "uc1" and r["margin"] > 0,
     dict(color="#2a78d6", marker="o", linestyle="-")),
    ("UC1, margin -30 s", lambda r: r["scen"] == "uc1" and r["margin"] <= 0,
     dict(color="#2a78d6", marker="s", linestyle="--", markerfacecolor="white")),
    ("UC4, margin 60 s", lambda r: r["scen"] == "uc4" and r["margin"] > 0,
     dict(color="#eb6834", marker="^", linestyle="-")),
    ("UC4, margin 0 s", lambda r: r["scen"] == "uc4" and r["margin"] <= 0,
     dict(color="#eb6834", marker="D", linestyle="--", markerfacecolor="white")),
]


def label(scen, sigma, margin):
    if scen == "uc1":
        return f"UC1 sigma={sigma:.0f} margin={margin:.0f}"
    return f"UC4 sigma={sigma:.0f} margin={margin:.0f}"


def plot(rows, stem):
    """(a) hit rate and (b) held-out gap against the candidate budget,
    (c) UC4: how often a tighter hop bound would lose the oracle."""
    import matplotlib.pyplot as plt

    groups = [(name, [r for r in rows if sel(r)], sty) for name, sel, sty in GROUPS]
    groups = [g for g in groups if g[1]]
    xs = list(VARIANTS)
    xpos = np.arange(len(xs))
    xt = [str(x) if x != "40rc" else "40\n+RC" for x in xs]
    nb = len(BUDGETS)
    ms = 4.0

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    ax = axes[0]
    for name, sub, sty in groups:
        hit = [np.mean([r[f"n{n}_hit"] for r in sub]) for n in xs]
        ax.plot(xpos[:nb], hit[:nb], label=name, markersize=ms, **sty)
        ax.plot(xpos[nb:], hit[nb:], markersize=ms, **{**sty, "linestyle": "none"})
    ax.axvline(nb - 0.5, color="#bbbbbb", linewidth=0.6)
    ax.set_xticks(xpos, xt)
    ax.set_xlabel("sampled realizations $n$")
    ax.set_ylabel("fraction of instances")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(a) selected = MC best (identical or tied)", fontsize=6.5)

    ax = axes[1]
    w = 0.18
    for j, (name, sub, sty) in enumerate(groups):
        for xi, n in enumerate(xs):
            g = np.array([r[f"n{n}_gap"] for r in sub])
            x = xi + (j - (len(groups) - 1) / 2) * w
            ax.vlines(x, 0.0, np.quantile(g, 0.9), color=sty["color"], linewidth=1.0,
                      linestyle=sty["linestyle"])
            ax.plot([x], [g.max()], marker="_", color=sty["color"], markersize=5)
            ax.plot([x], [g.mean()], marker=sty["marker"], color=sty["color"], markersize=ms,
                    markerfacecolor=sty.get("markerfacecolor", sty["color"]), linestyle="none")
    ax.axhline(0.0, color="#999999", linewidth=0.6)
    ax.axvline(nb - 0.5, color="#bbbbbb", linewidth=0.6)
    ax.set_xticks(xpos, xt)
    ax.set_xlabel("sampled realizations $n$")
    ax.set_ylabel(r"$P_{\mathrm{oracle}} - P_{\mathrm{selected}}$, held out")
    ax.set_title("(b) gap: mean (marker), 90th pct (bar), max (tick)", fontsize=6.5)

    ax = axes[2]
    for name, sub, sty in groups:
        if sub[0]["scen"] != "uc4":
            continue
        lv = list(HOP_LEVELS)
        lost = [np.mean([r["oracle_p3"] - r[f"orP_s{h}"] > 0.005 for r in sub]) for h in lv]
        ax.plot(lv, lost, markersize=ms, **sty)
    ax.set_xlabel("hop bound = shortest feasible + $s$")
    ax.set_ylabel("fraction of instances")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(list(HOP_LEVELS))
    ax.set_title("(c) UC4: a bound at $s$ loses > 0.005", fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=6.5,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.92))
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png")


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    sfx = "_quick" if quick else ""
    _style.setup()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    all_rows = []
    for j, scen in enumerate(("uc1", "uc4")):
        path = ROOT / "results" / f"gap_{scen}{sfx}.csv"
        if replot or (only is not None and only != scen):
            if path.exists():
                rows = read(path)
                all_rows += rows
                if replot:       # re-aggregate, keeping the recorded wall time
                    apath = ROOT / "results" / f"gap_{scen}_agg{sfx}.csv"
                    wall = read(apath)[0].get("wall_s_scenario", math.nan) if apath.exists() else math.nan
                    agg = aggregate(rows)
                    for r in agg:
                        r["wall_s_scenario"] = wall
                    write(apath, agg)
            continue
        t_start = time.time()
        tasks = [{"scen": scen, "sigma": s["sigma"], "margin": s["margin"], "i": i, "quick": quick}
                 for s in SETTINGS[scen] for i in range(4 if quick else s["R"])]
        rows = pmap(one_instance, tasks, seed=SEED + j, workers=1 if quick else 3)
        wall = time.time() - t_start
        write(path, rows)
        agg = aggregate(rows)
        for r in agg:
            r["wall_s_scenario"] = wall
        write(ROOT / "results" / f"gap_{scen}_agg{sfx}.csv", agg)
        all_rows += rows
        print(f"[{scen}] {len(rows)} instances in {wall:.0f} s")
        for r in agg:
            print(f"  {label(r['scen'], r['sigma'], r['margin'])} n={str(r['n_real']):>4s}: "
                  f"cands {r['n_cands_mean']:5.1f}  hit {r['hit']:.2f}  ident {r['ident']:.2f}  "
                  f"worse {r['worse']:.2f}  oracle_in {r['oracle_in']:.2f}  "
                  f"gap mean {r['gap_mean']:+.4f} q90 {r['gap_q90']:+.4f} max {r['gap_max']:+.4f}  "
                  f"infeas e.01/.05/.1 {r['e0.01_infeasible']:.2f}/{r['e0.05_infeasible']:.2f}/"
                  f"{r['e0.1_infeasible']:.2f}  enum {r['n_enum_mean']:.0f} "
                  f"(max {r['n_enum_max']}, capped {r['enum_capped']})")
    if all_rows:
        plot(all_rows, str(ROOT / "figures" / f"gap{sfx}"))


if __name__ == "__main__":
    main()
