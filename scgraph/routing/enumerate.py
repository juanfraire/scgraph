"""Exhaustive route enumeration under the support envelope (revision, R1/R2/R3).

The SLA-aware router ranks an *enumerated candidate set* (Sec. V-C), because
the quantile key has no optimal substructure and cannot drive a Dijkstra
label. The reviewers asked how much that costs. Answering needs a reference
set that provably contains every route a realization could deliver on time,
so that the best route in it is the best route, period. This module builds
that set.

The support envelope. Each stochastic contact is replaced by its most
favourable corner over the support of its laws:

  s_lo   earliest possible start, the lower end of the start law's support
  e_hi   latest possible end, start upper end plus duration upper end
         (`StochasticContact.support_window`)
  tx     shortest possible transfer, size / r_hi, where r_hi is the upper end
         of the rate support, or its `rate_q` quantile when the support is
         unbounded (rate_q = 1 keeps the true supremum, tx = 0)
  exists the contact can exist at all (p_exist > 0, under every latent scale)

Running the arrival recursion, eqs. (2)-(4), on the envelope gives an arrival
time that lower-bounds the arrival in every realization. The recursion is
monotone (a later arrival, a later start, a lower rate and an earlier end can
only delay delivery or break feasibility), so by induction along the route a
realized arrival is never earlier than the envelope arrival, and a realized
transfer that fits its window implies the envelope transfer fits too. Hence:

  If some realization delivers route pi by the deadline D, the envelope
  delivers pi by D, and `enumerate_routes` returns pi (provided pi is
  node-simple, has at most `hop_bound` contacts, and the route cap was not hit).

With rate_q < 1 the statement holds up to the truncated rate tail: a route the
envelope excludes delivers on time only if one of its unbounded-rate contacts
draws a rate above its rate_q quantile, so its success probability is at most
k (1 - rate_q), with k the number of such contacts on it. This holds whatever
the dependence between contacts (union bound on the marginals). With rate_q =
1 the statement is exact.

Restricting to node-simple routes loses nothing. A route that revisits a node
is dominated pathwise by the route with the cycle cut out: the shortcut
reaches the node earlier, and an earlier arrival can only help every later
hop (same monotonicity), while it needs a subset of the contacts to exist.

Pruning. A depth-first search extends routes contact by contact on the
envelope. Two bounds cut branches that cannot reach the destination in time,
and neither can cut a route the enumeration should return:

  hop distance  the static (time-free) hop distance to the destination over
                usable contacts; a branch with fewer remaining hops is cut
  latest arrival (time_prune) the latest envelope arrival time at a node from
                which the destination is still reachable by D within r more
                hops, computed backwards over walks (a superset of the simple
                routes, so the bound is valid for them)

The hop bound is the shortest envelope-feasible hop count plus `hop_slack`,
capped by `max_hops`. When the bound is at least the number of nodes minus
one it cannot bind, and the enumeration is complete over all routes.

`screen_routes` scores a large enumerated set on common random numbers: every
contact is drawn once per sample and shared by all routes, in exactly the
order `arrival.joint_route_set_deliveries` uses, and routes are evaluated
along a prefix tree so that shared prefixes cost one pass. Per-route success
counts are those `joint_route_set_deliveries` would give on the same stream.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from scgraph.arrival import factor_draws
from scgraph.graph import StochasticContactGraph

Route = tuple[int, ...]


@dataclass(frozen=True)
class EnvelopeContact:
    """The most favourable corner of one stochastic contact's support."""

    cid: int
    frm: int
    to: int
    s_lo: float    # earliest possible start
    e_hi: float    # latest possible end
    tx: float      # shortest possible transfer time for the bundle
    owlt: float


def _max_latent_scale(c, latent) -> float:
    if latent is None or c.group is None:
        return 1.0
    return max(s.get(c.group, 1.0) for _, s in latent.states)


def envelope_contact(c, size: float, rate_q: float = 0.999, latent=None) -> EnvelopeContact | None:
    """Envelope of one contact, or None when it can never exist."""
    if c.p_exist * _max_latent_scale(c, latent) <= 0.0:
        return None
    s_lo, s_hi = c.start.support()
    _, d_hi = c.duration.support()
    # support_window(), with the sampler's clip of the duration at zero
    e_hi = s_hi + max(0.0, d_hi)
    _, r_hi = c.rate.support()
    if not math.isfinite(r_hi):
        r_hi = math.inf if rate_q >= 1.0 else float(c.rate.quantile(rate_q))
    r_hi = max(r_hi, 1e-9)                      # the sampler's floor on the rate
    tx = 0.0 if math.isinf(r_hi) else size / r_hi
    return EnvelopeContact(c.cid, c.frm, c.to, float(s_lo), float(e_hi), float(tx), float(c.owlt))


def _step(a: float, ec: EnvelopeContact) -> float:
    """One envelope hop of eqs. (2)-(4); inf when the transfer cannot fit."""
    tau = a if a > ec.s_lo else ec.s_lo
    e = tau + ec.tx
    if e > ec.e_hi:
        return math.inf
    return e + ec.owlt


def envelope_arrival(route, graph: StochasticContactGraph, size: float, t0: float,
                     rate_q: float = 0.999) -> float:
    """Envelope (optimistic) arrival of a route given as cids or contacts."""
    a = t0
    for c in route:
        sc = graph.by_id(c) if isinstance(c, (int, np.integer)) else c
        ec = envelope_contact(sc, size, rate_q, graph.latent)
        if ec is None:
            return math.inf
        a = _step(a, ec)
        if not math.isfinite(a):
            return math.inf
    return a


@dataclass
class Enumeration:
    """Result of `enumerate_routes`. Routes are tuples of contact ids."""

    routes: list[Route]
    min_hops: int | None          # shortest envelope-feasible hop count (None: no route)
    hop_bound: int                # routes have at most this many contacts
    capped: bool                  # the route cap was hit (the set is incomplete)
    timed_out: bool               # the time budget ran out (the set is incomplete)
    n_usable: int                 # contacts that pass the static envelope filter
    n_nodes: int                  # nodes touched by usable contacts
    expansions: int               # DFS extensions tried
    elapsed: float                # seconds
    params: dict = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Every envelope-feasible node-simple route of <= hop_bound contacts is in."""
        return not (self.capped or self.timed_out)

    @property
    def hop_bound_free(self) -> bool:
        """The hop bound cannot bind: a simple route has at most n_nodes - 1 hops."""
        return self.hop_bound >= self.n_nodes - 1

    def contacts(self, graph: StochasticContactGraph) -> list[list]:
        return [[graph.by_id(cid) for cid in r] for r in self.routes]


class _Stop(Exception):
    pass


def enumerate_routes(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    *,
    hop_slack: int = 2,
    max_hops: int | None = None,
    max_routes: int = 200_000,
    rate_q: float = 0.999,
    time_prune: bool = True,
    max_seconds: float | None = None,
) -> Enumeration:
    """All node-simple routes src -> dst whose envelope arrival is <= deadline.

    hop_slack    the hop bound is (shortest feasible hop count) + hop_slack
    max_hops     absolute cap on the hop bound (None: number of nodes - 1)
    max_routes   stop once this many routes are found (`capped` is then set)
    rate_q       quantile standing in for an unbounded rate supremum (1: exact)
    time_prune   use the latest-arrival bound on top of the hop distance
    max_seconds  optional wall-clock budget (`timed_out` is then set)

    The search is deterministic: contacts are tried in (earliest start, cid)
    order, so a capped enumeration is reproducible.
    """
    t_begin = time.perf_counter()
    params = dict(hop_slack=hop_slack, max_hops=max_hops, max_routes=max_routes,
                  rate_q=rate_q, time_prune=time_prune, max_seconds=max_seconds)

    # static filter: contacts some realization could use for this bundle
    usable: list[EnvelopeContact] = []
    for c in graph.contacts:
        if c.frm == c.to or c.to == src or c.frm == dst:
            continue
        ec = envelope_contact(c, size, rate_q, graph.latent)
        if ec is None:
            continue
        a = _step(t0, ec)                  # released no earlier than t0
        if a > deadline:
            continue
        usable.append(ec)
    usable.sort(key=lambda ec: (ec.s_lo, ec.cid))
    nodes = {src, dst} | {ec.frm for ec in usable} | {ec.to for ec in usable}
    n_nodes = len(nodes)

    out: dict[int, list[EnvelopeContact]] = defaultdict(list)
    for ec in usable:
        out[ec.frm].append(ec)

    def result(routes, min_hops, hop_bound, capped=False, timed_out=False, expansions=0):
        return Enumeration(routes, min_hops, hop_bound, capped, timed_out, len(usable),
                           n_nodes, expansions, time.perf_counter() - t_begin, params)

    # shortest envelope-feasible hop count: earliest arrival by hop layers
    # (a shortest feasible walk is node-simple, so this is the simple-route minimum)
    ea = {src: t0}
    min_hops = None
    for h in range(1, n_nodes):
        nxt = dict(ea)
        for ec in usable:
            a = ea.get(ec.frm)
            if a is None:
                continue
            a2 = _step(a, ec)
            if a2 <= deadline and a2 < nxt.get(ec.to, math.inf):
                nxt[ec.to] = a2
        if nxt.get(dst, math.inf) <= deadline:
            min_hops = h
            break
        if nxt == ea:
            break
        ea = nxt
    if min_hops is None:                          # proven: no route at any hop count
        return result([], None, n_nodes - 1)

    hop_bound = min(min_hops + hop_slack, n_nodes - 1)
    if max_hops is not None:
        hop_bound = min(hop_bound, max_hops)
    if hop_bound < min_hops:                      # max_hops excludes every route
        return result([], min_hops, hop_bound)

    # static hop distance to dst over usable contacts (reverse BFS)
    rev: dict[int, list[int]] = defaultdict(list)
    for ec in usable:
        rev[ec.to].append(ec.frm)
    hopdist = {dst: 0}
    frontier = [dst]
    while frontier:
        nf = []
        for v in frontier:
            for u in rev[v]:
                if u not in hopdist:
                    hopdist[u] = hopdist[v] + 1
                    nf.append(u)
        frontier = nf

    # latest envelope arrival at v from which dst is reachable by D in <= r hops
    ldt: list[dict[int, float]] = [{dst: deadline}]
    if time_prune:
        for r in range(1, hop_bound + 1):
            prev = ldt[-1]
            cur = dict(prev)
            for ec in usable:
                lw = prev.get(ec.to)
                if lw is None:
                    continue
                b = min(ec.e_hi, lw - ec.owlt) - ec.tx   # latest transmission start
                if ec.s_lo <= b and b > cur.get(ec.frm, -math.inf):
                    cur[ec.frm] = b
            ldt.append(cur)

    routes: list[Route] = []
    path: list[int] = []
    visited = {src}
    deadline_s = None if max_seconds is None else t_begin + max_seconds
    counter = [0]
    flags = {"capped": False, "timed_out": False}
    inf = math.inf

    def dfs(v: int, a: float, h: int) -> None:
        for ec in out[v]:
            w = ec.to
            if w in visited:
                continue
            counter[0] += 1
            tau = a if a > ec.s_lo else ec.s_lo
            e = tau + ec.tx
            if e > ec.e_hi:
                continue
            a2 = e + ec.owlt
            if a2 > deadline:
                continue
            if w == dst:
                routes.append(tuple(path) + (ec.cid,))
                if len(routes) >= max_routes:
                    flags["capped"] = True
                    raise _Stop
                continue
            r = hop_bound - h - 1                 # hops left after taking ec
            if r <= 0 or hopdist.get(w, inf) > r:
                continue
            if time_prune and a2 > ldt[r].get(w, -inf):
                continue
            if deadline_s is not None and (counter[0] & 0xFFF) == 0 \
                    and time.perf_counter() > deadline_s:
                flags["timed_out"] = True
                raise _Stop
            path.append(ec.cid)
            visited.add(w)
            dfs(w, a2, h + 1)
            visited.discard(w)
            path.pop()

    try:
        if not time_prune or t0 <= ldt[hop_bound].get(src, -inf):
            dfs(src, t0, 0)
    except _Stop:
        pass
    return result(routes, min_hops, hop_bound, flags["capped"], flags["timed_out"], counter[0])


def screen_routes(
    graph: StochasticContactGraph,
    routes: list[Route],
    size: float,
    t0: float,
    deadline: float,
    m: int,
    rng: np.random.Generator,
    return_ontime: bool = False,
) -> dict:
    """Score many routes on common random numbers, along a prefix tree.

    Returns per-route arrays `k` (on-time count out of m), `p` (k / m) and
    `mean_delivery` (mean over delivered draws, inf if none), plus the (n, m)
    boolean on-time matrix when `return_ontime` is set. Draws are consumed in
    the order `joint_route_set_deliveries` uses, so the counts equal what it
    would give on the same generator state; only memory and time differ.
    """
    n = len(routes)
    if n == 0:
        out = {"k": np.zeros(0, int), "p": np.zeros(0), "mean_delivery": np.zeros(0), "n": m}
        if return_ontime:
            out["ontime"] = np.zeros((0, m), bool)
        return out
    latent = graph.latent

    contacts = {}
    for r in routes:
        for cid in r:
            if cid not in contacts:
                contacts[cid] = graph.by_id(cid)

    if latent is not None:
        probs = [p for p, _ in latent.states]
        idx = rng.choice(len(latent.states), size=m, p=probs)
    else:
        idx = None
    zf = factor_draws(list(contacts.values()), m, rng)

    draws = {}
    for cid, c in contacts.items():
        if c.group is not None and idx is not None:
            p_scale = np.array([latent.states[i][1].get(c.group, 1.0) for i in idx])
        else:
            p_scale = 1.0
        ex, ts, te, rate = c.sample_attrs(rng, m, p_scale=p_scale, zf=zf)
        draws[cid] = (ex, ts, te, rate, c.owlt)

    k = np.zeros(n, dtype=np.int64)
    mean_del = np.full(n, math.inf)
    ontime = np.zeros((n, m), dtype=bool) if return_ontime else None

    order = sorted(range(n), key=lambda i: routes[i])
    stack = [np.full(m, t0, dtype=float)]
    prev: Route = ()
    for i in order:
        r = routes[i]
        lcp = 0
        lim = min(len(prev), len(r))
        while lcp < lim and prev[lcp] == r[lcp]:
            lcp += 1
        del stack[lcp + 1:]
        for cid in r[lcp:]:
            ex, ts, te, rate, owlt = draws[cid]
            a = stack[-1]
            tau = np.maximum(a, ts)
            e = tau + size / rate
            feasible = ex & (e <= te) & np.isfinite(a)
            stack.append(np.where(feasible, e + owlt, np.inf))
        a = stack[len(r)]
        ok = a <= deadline
        k[i] = int(np.count_nonzero(ok))
        fin = np.isfinite(a)
        if fin.any():
            mean_del[i] = float(np.mean(a[fin]))
        if return_ontime:
            ontime[i] = ok
        prev = r

    out = {"k": k, "p": k / m, "mean_delivery": mean_del, "n": m}
    if return_ontime:
        out["ontime"] = ontime
    return out
