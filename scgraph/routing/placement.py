"""Where the planning plane runs (Sec. VI), as an executable comparison.

Section VI splits the work into a modeling plane, a planning plane, and a
forwarding plane, and argues that *where* the planning plane is hosted changes
the achieved SLA. This module makes that claim measurable by fixing three
trivial placements and executing each against the same realizations.

What the placements differ in is deliberately narrow, so the result attributes
cleanly:

  ground   a ground control centre. Global view and the full compute budget, so
           it solves the whole candidate set, but its product is a single route
           table uplinked ahead of time. It has no contingency: once a contact
           on the chosen route fails to materialize, the bundle is stuck.

  proxy    a platform or compute-satellite in the sky. It precomputes a small
           *ranked* table per node, which the forwarding plane resolves online
           by walking down the ranking until a contact is actually available.
           It reasons globally but on a reduced candidate budget.

  dist     planning pushed to the nodes. Maximum freshness, no downstream view:
           each node ranks only its own outgoing contacts, by earliest arrival
           at the next hop, with no notion of whether that hop can reach the
           destination.

The axis being varied is therefore contingency depth and view scope, which is
the compute-and-locality trade of Sec. VI-B. Staleness of the *model* is a
separate axis, exercised by UC3's stale-plan baseline, and is not varied here.

All three are executed by the same driver against the same drawn realizations,
so the spread between them is attributable to the placement alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from scgraph.contact import DetContact, StochasticContact
from scgraph.graph import Realization, StochasticContactGraph
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, route_stats

MAX_HOPS = 6


def _by_cid(real: Realization) -> dict[int, DetContact]:
    return {c.cid: c for c in real.contacts}


def _hop(c: DetContact, size: float, a: float) -> float:
    """Arrival after taking contact c at time a; inf if it does not fit."""
    tau = max(a, c.t_start)
    e = tau + size / c.rate
    if e > c.t_end:
        return math.inf
    return e + c.owlt


def execute(
    real: Realization,
    src: int,
    dst: int,
    size: float,
    t0: float,
    choose,
) -> float:
    """Run a forwarding policy against one realization; inf on failure.

    `choose(node, time, avail, used) -> DetContact | None` is the online
    decision the forwarding plane makes; the placements differ only in it.
    """
    avail = _by_cid(real)
    node, a, used = src, t0, set()
    for _ in range(MAX_HOPS):
        if node == dst:
            return a
        c = choose(node, a, avail, used)
        if c is None:
            return math.inf
        nxt = _hop(c, size, a)
        if not math.isfinite(nxt):
            return math.inf
        used.add(c.cid)
        node, a = c.to, nxt
    return a if node == dst else math.inf


# --------------------------------------------------------------------------
# planning: what each placement computes offline
# --------------------------------------------------------------------------


@dataclass
class GroundPlan:
    """One route, full candidate budget, no contingency."""

    cids: list[int]

    def chooser(self):
        def choose(node, a, avail, used):
            for cid in self.cids:
                if cid in used:
                    continue
                c = avail.get(cid)
                # the plan is followed blindly: the next contact on the route
                # either is there and fits, or the bundle is stuck
                return c if (c is not None and c.frm == node) else None
            return None

        return choose


@dataclass
class ProxyPlan:
    """Ranked route table keyed by (node, time bucket), resolved online.

    Bucketing by arrival time is what lets a precomputed table stay useful once
    the bundle is running late: the entry consulted at t is the one planned for
    bundles arriving around t, so the proxy is not forced to re-solve online.
    """

    size: float
    t0: float
    deadline: float
    n_buckets: int
    table: dict[tuple[int, int], list[list[int]]] = field(default_factory=dict)

    def bucket(self, a: float) -> int:
        span = max(1e-9, self.deadline - self.t0)
        k = int((a - self.t0) / span * self.n_buckets)
        return min(max(k, 0), self.n_buckets - 1)

    def chooser(self):
        def choose(node, a, avail, used):
            for cids in self.table.get((node, self.bucket(a)), []):
                if not cids:
                    continue
                c = avail.get(cids[0])
                if c is None or c.frm != node or c.cid in used:
                    continue
                if math.isfinite(_hop(c, self.size, a)):
                    return c
            return None

        return choose


@dataclass
class DistributedPlan:
    """No precomputation: rank local outgoing contacts by earliest arrival."""

    size: float

    def chooser(self):
        def choose(node, a, avail, used):
            best, best_arr = None, math.inf
            for c in avail.values():
                if c.frm != node or c.cid in used:
                    continue
                arr = _hop(c, self.size, a)
                if arr < best_arr:
                    best, best_arr = c, arr
            return best

        return choose


def plan_ground(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    rng: np.random.Generator,
    m: int = 1000,
    n_realizations: int = 40,
) -> GroundPlan:
    """Full-budget SLA-aware search, collapsed to a single uplinked route."""
    from scgraph.routing.quantile import pick_route

    route, _ = pick_route(
        graph, src, dst, size, t0, deadline, rng,
        m=m, n_realizations=n_realizations,
    )
    return GroundPlan([c.cid for c in route] if route else [])


def plan_proxy(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    rng: np.random.Generator,
    m: int = 300,
    n_realizations: int = 8,
    depth: int = 3,
    n_buckets: int = 4,
) -> ProxyPlan:
    """Ranked table per (node, time bucket), on a reduced candidate budget.

    For every node the bundle could occupy and every arrival bucket, rank
    candidate routes to the destination by P[deliver by deadline] and keep the
    top `depth`. The reduced m / n_realizations model the proxy's smaller
    compute allowance relative to a ground centre.
    """
    table: dict[tuple[int, int], list[list[int]]] = {}
    nodes = {src} | {c.to for c in graph.contacts}
    span = max(1e-9, deadline - t0)
    for u in nodes:
        if u == dst:
            continue
        for k in range(n_buckets):
            t_k = t0 + span * k / n_buckets
            cands = candidate_routes(graph, u, dst, size, t_k, rng, n_realizations)
            if not cands:
                continue
            scored = [
                (r, route_stats(r, size, t_k, deadline, m, rng, latent=graph.latent))
                for r in cands
            ]
            scored.sort(key=lambda rs: -rs[1]["p_success"])
            table[(u, k)] = [[c.cid for c in r] for r, _ in scored[:depth]]
    return ProxyPlan(size, t0, deadline, n_buckets, table)


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------

PLACEMENTS = ("ground", "proxy", "dist")


def placement_delivery(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    rng: np.random.Generator,
    m: int = 1000,
    plan_m: int = 600,
) -> dict[str, float]:
    """P[deliver by deadline] under each placement, on shared realizations."""
    ground = plan_ground(graph, src, dst, size, t0, deadline, rng, m=plan_m)
    proxy = plan_proxy(graph, src, dst, size, t0, deadline, rng)
    dist = DistributedPlan(size)

    choosers = {
        "ground": ground.chooser(),
        "proxy": proxy.chooser(),
        "dist": dist.chooser(),
    }
    hits = {k: 0 for k in choosers}
    for _ in range(m):
        real = graph.realize(rng)          # one realization, all three see it
        for k, ch in choosers.items():
            if execute(real, src, dst, size, t0, ch) <= deadline:
                hits[k] += 1
    return {k: v / m for k, v in hits.items()}
