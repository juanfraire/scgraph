"""SLA-aware route selection (Sec. V): quantile cost and max-P(success).

Candidate routes come from the deterministic router run on the mean plan and
on sampled realizations (each realization's optimal route is a candidate);
each candidate is then scored under the laws by per-route Monte Carlo, and the
router picks argmax P[T <= D] (eq. 6) or, equivalently for a fixed epsilon,
min Q_{1-eps} (eqs. 8-10).
"""

from __future__ import annotations

import math

import numpy as np

from scgraph.arrival import sample_route_deliveries
from scgraph.contact import StochasticContact
from scgraph.graph import StochasticContactGraph
from scgraph.routing.deterministic import cgr_route
from scgraph.stats import cp_lower


def candidate_routes(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    rng: np.random.Generator,
    n_realizations: int = 40,
    risk_classes: bool = False,
) -> list[list[StochasticContact]]:
    """Candidate route set: mean-plan route + optimal routes of sampled realizations.

    `risk_classes` (revision) appends, for each class of uncertain contacts
    (random existence, random timing, both), the mean-plan route that avoids
    the whole class, plus its single-contact exclusions. It reaches the slow,
    certain route that shares contacts with the fast risky ones, which the
    disjoint sweep cannot reach on a meshed backbone (results/notes/gap.md:
    on UC4 it lifts the share of instances whose best route is offered from
    25% to 95%). The routes are deterministic searches appended after the
    others, so they consume no random numbers; off by default, which keeps
    the submitted use cases draw-for-draw unchanged.
    """
    seen: dict[tuple[int, ...], list[StochasticContact]] = {}

    def add(det_route):
        if det_route is None:
            return
        key = tuple(c.cid for c in det_route)
        if key not in seen:
            seen[key] = [graph.by_id(cid) for cid in key]

    mean_plan = graph.mean_plan()
    mean_route, _ = cgr_route(mean_plan, src, dst, size, t0)
    add(mean_route)
    # alternatives on the mean plan (CGR-style single-contact exclusion)
    if mean_route is not None:
        for c in mean_route:
            alt, _ = cgr_route(
                mean_plan, src, dst, size, t0, excluded_cids=frozenset({c.cid})
            )
            add(alt)
    # disjoint sweep (contact suppression): repeatedly exclude all contacts of
    # the routes found so far and re-route, so slower-but-disjoint alternatives
    # (e.g. the certain backbone) always enter the candidate set
    suppressed: set[int] = set(c.cid for c in mean_route) if mean_route else set()
    while True:
        alt, _ = cgr_route(
            mean_plan, src, dst, size, t0, excluded_cids=frozenset(suppressed)
        )
        if alt is None:
            break
        add(alt)
        suppressed.update(c.cid for c in alt)
    for _ in range(n_realizations):
        r, _ = cgr_route(graph.realize(rng), src, dst, size, t0)
        add(r)
    if risk_classes:
        unc_exist = frozenset(c.cid for c in graph.contacts if c.p_exist < 1.0)
        unc_time = frozenset(c.cid for c in graph.contacts
                             if not (c.start.is_point_mass and c.duration.is_point_mass))
        for excl in (unc_exist, unc_time, unc_exist | unc_time):
            r, _ = cgr_route(mean_plan, src, dst, size, t0, excluded_cids=excl)
            if r is None:
                continue
            add(r)
            for c in r:
                alt, _ = cgr_route(mean_plan, src, dst, size, t0,
                                   excluded_cids=excl | {c.cid})
                add(alt)
    return list(seen.values())


def route_stats(
    route: list[StochasticContact],
    size: float,
    t0: float,
    deadline: float,
    m: int,
    rng: np.random.Generator,
    eps: float = 0.05,
    latent=None,
    delta: float = 0.05,
) -> dict:
    """P[T<=D] (eq. 5) and Q_{1-eps} (eq. 8) for one route, by Monte Carlo.

    Besides the point estimate, returns the success count `k` out of `n` draws
    and the one-sided (1 - delta) Clopper-Pearson lower bound `p_lcb`, which is
    what an admission decision should rest on (revision, R2/R3).
    """
    t = sample_route_deliveries(route, size, t0, m, rng, latent=latent)
    k = int(np.count_nonzero(t <= deadline))
    p = k / m
    q = float(np.quantile(t, 1.0 - eps)) if np.isfinite(t).mean() >= 1.0 - eps else math.inf
    return {
        "p_success": float(p),
        "k": k,
        "n": int(m),
        "p_lcb": cp_lower(k, m, delta),
        "quantile": q,
        "mean_delivery": float(np.mean(t[np.isfinite(t)])) if np.isfinite(t).any() else math.inf,
        "samples": t,
    }


def pick_route(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    rng: np.random.Generator,
    m: int = 2000,
    eps: float = 0.05,
    n_realizations: int = 40,
    cands: list[list[StochasticContact]] | None = None,
) -> tuple[list[StochasticContact] | None, dict]:
    """The SLA-aware router: argmax P[T<=D] over the candidate set (eq. 6).

    The returned statistics come from the draws used to select the route, so
    they are optimistic by construction (the winner of a noisy comparison).
    Report a route's performance with `verify.montecarlo.evaluate_route` on
    an independent stream instead.
    """
    if cands is None:
        cands = candidate_routes(graph, src, dst, size, t0, rng, n_realizations)
    best, best_stats = None, {"p_success": -1.0}
    for route in cands:
        s = route_stats(route, size, t0, deadline, m, rng, eps=eps, latent=graph.latent)
        # primary: P(success); tie-break: mean delivery time
        better = s["p_success"] > best_stats["p_success"] + 1e-12 or (
            abs(s["p_success"] - best_stats["p_success"]) <= 1e-12
            and s.get("mean_delivery", math.inf) < best_stats.get("mean_delivery", math.inf)
        )
        if better:
            best, best_stats = route, s
    return best, best_stats


def pick_route_chance(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    rng: np.random.Generator,
    m: int = 2000,
    eps: float = 0.05,
    n_realizations: int = 40,
    admission: str = "lcb",
    delta: float = 0.05,
    cands: list[list[StochasticContact]] | None = None,
) -> tuple[list[StochasticContact] | None, dict]:
    """The chance-constrained form (eq. 7): minimise latency s.t. P[T<=D] >= 1-eps.

    Admissibility is tested on P[T<=D] directly, not on Q_{1-eps}; eq. (9) is the
    equivalence that lets us do that.

    `admission` sets what "meets the SLA" means for a candidate scored on m
    draws:
      "point"  the estimate k/m clears 1 - eps (the original rule). Selecting
               the fastest admitted route then favours candidates whose
               estimate happened to come out high, so the admitted route can
               sit above budget once measured on fresh draws.
      "lcb"    the one-sided Clopper-Pearson lower bound clears 1 - eps, at
               level delta / |candidates| (Bonferroni). Under the model, the
               probability that ANY admitted route truly violates the budget
               is then at most delta. This is the statement the paper calls
               SLA admission at confidence 1 - delta.

    This is where epsilon earns its keep. Tightening it forces the router onto
    slower, safer routes; loosening it lets the router spend reliability to buy
    delivery time. When no candidate meets the constraint the problem is
    infeasible, and we fall back to argmax P[T<=D] so the router still returns
    its best effort (and the caller can see the constraint was not met).
    """
    if cands is None:
        cands = candidate_routes(graph, src, dst, size, t0, rng, n_realizations)
    scored = [
        (r, route_stats(r, size, t0, deadline, m, rng, eps=eps, latent=graph.latent))
        for r in cands
    ]
    if not scored:
        return None, {"p_success": 0.0, "feasible": False, "n_candidates": 0}

    if admission == "point":
        ok = [(r, s) for r, s in scored if s["p_success"] >= 1.0 - eps - 1e-12]
    elif admission == "lcb":
        d = delta / len(scored)
        ok = [(r, s) for r, s in scored if cp_lower(s["k"], s["n"], d) >= 1.0 - eps]
    else:
        raise ValueError(f"unknown admission rule {admission!r}")
    if ok:
        route, stats = min(ok, key=lambda rs: rs[1].get("mean_delivery", math.inf))
        stats = {**stats, "feasible": True, "n_candidates": len(scored)}
        return route, stats

    route, stats = max(scored, key=lambda rs: rs[1]["p_success"])
    return route, {**stats, "feasible": False, "n_candidates": len(scored)}
