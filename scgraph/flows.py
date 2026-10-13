"""Several bundles on one realized contact plan: queueing, finite storage, booking.

The paper's arrival recursion (eqs. 2-4, `arrival.delivery_time`) follows ONE
atomic bundle through a committed route with unbounded storage. This module
lets K bundles share the plan, which is what a reviewer asks about when the
SLA-aware router sends traffic to the orbital backbone: do the bundles then
queue behind each other, fill relay storage, and miss the deadline anyway?

Executor (`run_flows`), on ONE realization:

  * every bundle has a size, a release time at its source, a deadline, and a
    committed route (contact ids in hop order);
  * a contact is a single server: it transmits one bundle at a time, in FIFO
    order of arrival at its sender (ties broken by bundle index), so a
    transmission starts at max(arrival at sender, contact start, contact
    busy-until), which is eq. (2) with the backlog added;
  * transfer is atomic at a constant rate (eq. 3). A transfer that cannot
    finish by the contact end is cut at the end: the contact stays busy until
    then and the bundle is lost at that hop (cause "window"). A bundle whose
    turn comes after the window has closed is lost the same way, at once;
  * on success the bundle reaches the next node one-way light time later
    (eq. 4);
  * every node other than the bundles' sources and destinations stores at most
    B bundles. A bundle is resident from its arrival until its transmission
    out finishes (or it is lost there); an arrival that finds B resident
    bundles is dropped (cause "buffer"). Sources and destinations are
    unbounded;
  * a committed contact absent from the realization (existence draw failed)
    loses the bundle on arrival at its sender (cause "missing"); a bundle the
    policy could not route at all has cause "noroute";
  * events are processed in global time order, departures before arrivals
    before window openings at equal times, so a slot freed at t is available
    to a bundle arriving at t.

Contacts are independent servers: a node may transmit on two contacts at once
if both are open (CPD resolves interface conflicts in the skeleton). There is
no bundle lifetime: a late bundle is still delivered, and counted as late.

With one bundle and unbounded storage the executor reduces exactly to
`arrival.delivery_time`; `tests/test_flows.py` checks that, FIFO queueing,
buffer drops and the window cut-off.

Two routing helpers for the flow experiment live here as well:

  `cgr_route_booked`       CGR on a plan whose contacts already carry booked
                           volume (SABR-style residual volume): a contact is
                           usable only if the bundle fits after the backlog.
  `assign_contention_aware`  sequential SLA-aware assignment: bundle j takes the
                           candidate that maximises its own on-time probability
                           given bundles 1..j-1 already committed, scored by
                           Monte Carlo over the executor.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from scgraph.contact import DetContact
from scgraph.graph import Realization

LOSS_CAUSES = ("window", "buffer", "missing", "noroute")

# event priorities at equal times: storage is freed before it is claimed, and a
# bundle arriving exactly at a window opening queues before the opening serves
_DEPART, _ARRIVE, _OPEN = 0, 1, 2


@dataclass(frozen=True)
class Bundle:
    """One bundle of a flow: where it goes, when, how big, and on which route."""

    src: int
    dst: int
    size: float
    release: float
    deadline: float
    route: tuple[int, ...] = ()

    def on(self, route) -> "Bundle":
        """The same bundle committed to another route (contact ids or contacts)."""
        cids = tuple(c if isinstance(c, (int, np.integer)) else c.cid for c in route)
        return replace(self, route=tuple(int(c) for c in cids))


@dataclass
class FlowResult:
    """Per-bundle outcome of one executor run (index = position in the input)."""

    delivery: np.ndarray                 # delivery time, inf if lost
    cause: list[str]                     # "ok" or one of LOSS_CAUSES
    lost_node: list[int | None]          # node where a lost bundle was dropped
    peak_occupancy: dict[int, int] = field(default_factory=dict)

    def ontime(self, deadlines) -> np.ndarray:
        return self.delivery <= np.asarray(deadlines, dtype=float)

    def status(self, deadlines) -> list[str]:
        """"ontime", "late", or the loss cause, per bundle."""
        d = np.asarray(deadlines, dtype=float)
        out = []
        for i, c in enumerate(self.cause):
            if c != "ok":
                out.append(c)
            else:
                out.append("ontime" if self.delivery[i] <= d[i] else "late")
        return out


def contact_map(plan) -> dict[int, DetContact]:
    """cid -> DetContact for a Realization (or pass a mapping through)."""
    if isinstance(plan, Realization):
        return {c.cid: c for c in plan.contacts}
    if isinstance(plan, Mapping):
        return plan
    return {c.cid: c for c in plan}


def run_flows(
    plan,
    bundles: Sequence[Bundle],
    buffer: float = math.inf,
    unbounded: set[int] | None = None,
) -> FlowResult:
    """Execute committed bundles on one realized plan; see the module docstring.

    `plan` is a Realization or a mapping cid -> DetContact (build it once with
    `contact_map` when the same plan is reused). `buffer` is B, the storage of
    every intermediate node in bundles. `unbounded` lists nodes exempt from B
    and defaults to every bundle's source and destination.
    """
    cmap = contact_map(plan)
    n = len(bundles)
    delivery = np.full(n, math.inf)
    cause = ["ok"] * n
    lost_node: list[int | None] = [None] * n
    if unbounded is None:
        unbounded = {b.src for b in bundles} | {b.dst for b in bundles}
    bounded = math.isfinite(buffer)

    occ: dict[int, int] = defaultdict(int)
    peak: dict[int, int] = defaultdict(int)
    queue: dict[int, deque] = defaultdict(deque)
    busy: set[int] = set()
    open_pending: set[int] = set()
    hop = [0] * n

    events: list = []
    seq = 0

    def push(t, prio, b, kind, data):
        nonlocal seq
        heapq.heappush(events, (t, prio, b, seq, kind, data))
        seq += 1

    def holds(node: int) -> bool:
        # residency is counted at every intermediate node (peak_occupancy
        # reports it even when B is infinite); B is enforced only if finite
        return node not in unbounded

    def release_slot(node: int) -> None:
        if holds(node):
            occ[node] -= 1

    def lose(b: int, why: str, node: int) -> None:
        cause[b] = why
        lost_node[b] = node

    def serve(cid: int, t: float) -> None:
        c = cmap[cid]
        q = queue[cid]
        while q and cid not in busy:
            if t < c.t_start:
                if cid not in open_pending:
                    open_pending.add(cid)
                    push(c.t_start, _OPEN, -1, "open", cid)
                return
            b = q.popleft()
            end = t + bundles[b].size / c.rate          # eqs. (2)-(3) with backlog
            if end > c.t_end:
                if t >= c.t_end:                        # window over before its turn
                    release_slot(c.frm)
                    lose(b, "window", c.frm)
                    continue
                busy.add(cid)                           # cut at the window end
                push(c.t_end, _DEPART, b, "fail", cid)
                return
            busy.add(cid)
            push(end, _DEPART, b, "done", cid)
            return

    for i, b in enumerate(bundles):
        if not b.route:
            lose(i, "noroute", b.src)
        else:
            push(b.release, _ARRIVE, i, "arrive", b.src)

    while events:
        t, _, b, _, kind, data = heapq.heappop(events)
        if kind == "arrive":
            node = data
            route = bundles[b].route
            h = hop[b]
            if h == len(route):
                delivery[b] = t
                continue
            if holds(node):
                if bounded and occ[node] >= buffer:
                    lose(b, "buffer", node)
                    continue
                occ[node] += 1
                peak[node] = max(peak[node], occ[node])
            cid = route[h]
            c = cmap.get(cid)
            if c is None:
                release_slot(node)
                lose(b, "missing", node)
                continue
            if c.frm != node:
                raise ValueError(
                    f"bundle {b}: contact {cid} leaves node {c.frm}, "
                    f"but the bundle is at node {node}"
                )
            queue[cid].append(b)
            if cid not in busy:
                serve(cid, t)
        elif kind in ("done", "fail"):
            cid = data
            c = cmap[cid]
            busy.discard(cid)
            release_slot(c.frm)
            if kind == "done":
                hop[b] += 1
                push(t + c.owlt, _ARRIVE, b, "arrive", c.to)   # eq. (4)
            else:
                lose(b, "window", c.frm)
            serve(cid, t)
        else:  # window opening
            cid = data
            open_pending.discard(cid)
            if cid not in busy:
                serve(cid, t)

    return FlowResult(delivery, cause, lost_node, dict(peak))


# ---------------------------------------------------------------------------
# SABR-style residual-volume booking on the mean plan (policy P2)
# ---------------------------------------------------------------------------


def cgr_route_booked(
    real: Realization,
    src: int,
    dst: int,
    size: float,
    t0: float,
    booked: Mapping[int, float],
    excluded_cids: frozenset[int] = frozenset(),
) -> tuple[list[DetContact] | None, float]:
    """Earliest-arrival route on a plan whose contacts carry booked volume.

    `booked[cid]` is the volume already committed to contact cid. The bundle
    is assumed to be sent after that backlog, so its earliest transmission
    opportunity is max(arrival, t_start + booked / rate), and the contact is
    usable only if the transfer still ends by t_end. For a bundle that arrives
    before the backlog clears this is exactly the residual-volume test
    booked + size <= rate * (t_end - t_start). With nothing booked it is
    `routing.deterministic.cgr_route`, route for route.
    """
    outgoing: dict[int, list[DetContact]] = defaultdict(list)
    for c in real.contacts:
        if c.cid not in excluded_cids:
            outgoing[c.frm].append(c)

    best: dict[int, tuple[float, int]] = {src: (t0, 0)}
    parent: dict[int, tuple[int, DetContact]] = {}
    pq: list[tuple[float, int, int]] = [(t0, 0, src)]
    while pq:
        a, hops, u = heapq.heappop(pq)
        if (a, hops) > best.get(u, (math.inf, 0)):
            continue
        if u == dst:
            break
        for c in outgoing[u]:
            tau = max(a, c.t_start + booked.get(c.cid, 0.0) / c.rate)
            e = tau + size / c.rate
            if e > c.t_end:
                continue
            a2, h2 = e + c.owlt, hops + 1
            if (a2, h2) < best.get(c.to, (math.inf, 0)):
                best[c.to] = (a2, h2)
                parent[c.to] = (u, c)
                heapq.heappush(pq, (a2, h2, c.to))

    if dst not in best or not math.isfinite(best[dst][0]):
        return None, math.inf
    route: list[DetContact] = []
    node = dst
    while node != src:
        prev, c = parent[node]
        route.append(c)
        node = prev
    route.reverse()
    return route, best[dst][0]


def book_routes(
    plan: Realization, template: Sequence[Bundle]
) -> tuple[list[tuple[int, ...]], dict[int, float]]:
    """Route bundles one after another with `cgr_route_booked`, booking as we go.

    Returns the committed routes (empty tuple when nothing fits) and the final
    booked volume per contact.
    """
    booked: dict[int, float] = defaultdict(float)
    routes: list[tuple[int, ...]] = []
    for b in template:
        r, _ = cgr_route_booked(plan, b.src, b.dst, b.size, b.release, booked)
        if r is None:
            routes.append(())
            continue
        for c in r:
            booked[c.cid] += b.size
        routes.append(tuple(c.cid for c in r))
    return routes, dict(booked)


# ---------------------------------------------------------------------------
# Contention-aware SLA assignment (policy P4)
# ---------------------------------------------------------------------------


def assign_contention_aware(
    candidates: Sequence[Sequence[int]],
    template: Sequence[Bundle],
    plans: Sequence[Mapping[int, DetContact]],
    buffer: float = math.inf,
    unbounded: set[int] | None = None,
) -> tuple[list[tuple[int, ...]], list[dict]]:
    """Commit bundles in order, each to the candidate best for it under contention.

    For bundle j every candidate route is scored by running the executor on
    each of the `plans` (Monte-Carlo realizations, the selection stream) with
    bundles 1..j-1 on their committed routes and bundle j on the candidate.
    The winner maximises bundle j's on-time frequency; ties go to the larger
    expected number of on-time bundles in the set, then to bundle j's earlier
    mean delivery. All candidates are scored on the same plans (common random
    numbers), so the comparison is not blurred by sampling noise between them.

    Pruning is exact for this objective: other bundles can only delay or drop
    bundle j, never speed it up, so its on-time frequency alone on the plans
    bounds its frequency under contention. A candidate whose bound is below
    the best frequency already found cannot win and is skipped.

    Returns the committed routes and, per bundle, the winning score.
    """
    cands = [tuple(int(c) for c in r) for r in candidates]
    if not cands:
        return [() for _ in template], [{} for _ in template]
    m = len(plans)
    if unbounded is None:
        unbounded = {b.src for b in template} | {b.dst for b in template}

    # single-bundle on-time frequency of each candidate: the pruning bound,
    # computed once per distinct bundle signature (all equal in a burst)
    bounds: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def bound_for(tb: Bundle):
        key = (tb.src, tb.dst, tb.size, tb.release, tb.deadline)
        if key not in bounds:
            alone = np.zeros(len(cands))
            for k, r in enumerate(cands):
                b = tb.on(r)
                hits = 0
                for p in plans:
                    hits += run_flows(p, [b], buffer, unbounded).delivery[0] <= b.deadline
                alone[k] = hits / m
            bounds[key] = (alone, np.argsort(-alone, kind="stable"))
        return bounds[key]

    committed: list[Bundle] = []
    log: list[dict] = []
    for j, tb in enumerate(template):
        alone, order = bound_for(tb)
        best_key, best_r, best_s = None, None, None
        for k in order:
            if best_key is not None and alone[k] < best_key[0] - 1e-12:
                break
            trial = committed + [tb.on(cands[k])]
            dl = np.array([b.deadline for b in trial])
            own_hits, tot_hits, own_t = 0, 0, []
            for p in plans:
                res = run_flows(p, trial, buffer, unbounded)
                ok = res.delivery <= dl
                own_hits += bool(ok[-1])
                tot_hits += int(ok.sum())
                if math.isfinite(res.delivery[-1]):
                    own_t.append(res.delivery[-1])
            p_own = own_hits / m
            mean_own = float(np.mean(own_t)) if own_t else math.inf
            key = (p_own, tot_hits / m, -mean_own)
            if best_key is None or key > best_key:
                best_key, best_r = key, cands[k]
                best_s = {"p_own": p_own, "exp_ontime": tot_hits / m,
                          "mean_own": mean_own, "p_alone": float(alone[k])}
        committed.append(tb.on(best_r))
        log.append(best_s)
    return [b.route for b in committed], log
