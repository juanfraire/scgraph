"""The discrete-time Bernoulli TEG corner of G (Sec. III-A), as an ablation baseline.

The uncertainty-aware DTN line (BRUF / RUCoP / LSS) keeps a time-indexed contact
plan but abstracts each contact to a *binary existence* flag on a *discretized*
time grid, and optimizes *expected delivery probability*. The paper claims that
model is a coarse marginal of eq. (1). This module builds that marginal
explicitly, by projection:

    q_c[k] = P[ contact c is up for the whole of slot k ]

marginalizing over the contact's start/duration laws, its existence probability,
and (for UC3) the latent state. Everything eq. (1) knows about *where inside the
window* the contact sits, and about *which contacts fail together*, is discarded
by construction: q is a per-contact, per-slot scalar.

A route is then chosen to maximize the product of q over the (contact, slot)
pairs it uses, one hop per slot, which is the slotted independent-Bernoulli
objective of that line. The chosen route is afterwards evaluated under the FULL
laws, exactly like every other policy, so the comparison isolates what the
coarse model costs at routing time.

As the slot width goes to zero the projection retains more of the timing in
eq. (1), and on independent contacts the policy converges toward the SLA-aware
one (the ablation of Sec. VII-C). It cannot converge in general: which
contacts fail together is discarded at every slot width, so a finer grid does
not give back correlated failure (run_r1_uc3grid.py), and with the published
one-slot-per-hop rule a grid finer than a hop's transfer time books contacts
the bundle reaches too late (run_r1_bernoulli.py; see `advance`).
"""

from __future__ import annotations

import bisect
import heapq
import math

import numpy as np

from scgraph.contact import StochasticContact
from scgraph.graph import StochasticContactGraph

# Below this probability a (contact, slot) pair is treated as unusable, which
# keeps -log q finite and the time-expanded graph sparse.
Q_FLOOR = 1e-4


def _marginal_scales(latent, group: str | None, m: int, rng: np.random.Generator):
    """Latent-state existence scales, drawn marginally.

    Drawing the latent state independently per contact is precisely the
    modelling error of the independent-Bernoulli abstraction: the marginal
    existence probability comes out right, the joint structure is lost.
    """
    if latent is None or group is None:
        return 1.0
    return latent.sample_scales(rng, m, group)


def slot_usability(
    contact: StochasticContact,
    n_slots: int,
    slot_width: float,
    size: float,
    m: int,
    rng: np.random.Generator,
    latent=None,
    t0: float = 0.0,
) -> np.ndarray:
    """(n_slots,) array of q_c[k]: P[c can carry the bundle from the top of slot k].

    The bundle is offered at the slot boundary t_k, so the contact must already
    be open and must stay open long enough to drain it (eq. 14):

        q_c[k] = P[ X^c=1,  T_s <= t_k,  T_e >= t_k + size/R^c ].

    Feasibility is evaluated only at slot boundaries, which is exactly the
    temporal resolution the discrete model gives up. Sizing the test by the
    actual rate law (instead of demanding the contact span the whole slot)
    keeps the coarse grid from being penalized for contact *duration*, so the
    slot-width sweep isolates resolution loss alone.

    The grid starts at `t0`, the bundle's release, so its size is set by how
    long the bundle has to live and not by the absolute epoch of the plan. A
    plan whose horizon starts hours before the bundle would otherwise inflate
    the slot count without adding information.
    """
    scales = _marginal_scales(latent, contact.group, m, rng)
    exists, t_s, t_e, rate = contact.sample_attrs(rng, m, p_scale=scales)
    edges = t0 + np.arange(n_slots, dtype=float) * slot_width
    transfer = size / rate                                   # (m,)
    open_by = t_s[:, None] <= edges[None, :]                 # (m, n_slots)
    drains = t_e[:, None] >= edges[None, :] + transfer[:, None]
    return (open_by & drains & exists[:, None]).mean(axis=0)


def bernoulli_projection(
    graph: StochasticContactGraph,
    n_slots: int,
    slot_width: float,
    size: float,
    m: int,
    rng: np.random.Generator,
    t0: float = 0.0,
) -> dict[int, np.ndarray]:
    """Project G onto the slotted on/off TEG: {cid -> q[k]}."""
    return {
        c.cid: slot_usability(
            c, n_slots, slot_width, size, m, rng, latent=graph.latent, t0=t0
        )
        for c in graph.contacts
    }


def bernoulli_route(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    slot_width: float,
    rng: np.random.Generator,
    m: int = 4000,
    q: dict[int, np.ndarray] | None = None,
    advance: int | dict[int, int] = 1,
) -> tuple[list[StochasticContact] | None, float]:
    """Max expected-delivery-probability route in the slotted Bernoulli model.

    One hop per slot (the abstraction assumes a slot is wide enough to carry a
    bundle), independent per-slot existence, product objective. Dijkstra on
    -log q over states (node, slot).

    `advance` (revision 2, R1.3) sets how many slots a hop takes: a hop over
    contact c taken in slot j lands in slot j + advance (an int for every
    contact, or a per-contact {cid: slots} map). The default 1 is the
    published rule, the one BRUF and RUCoP use. 0 chains any number of hops
    inside one slot, each tested as if offered at the slot boundary (an
    optimistic reading). A map such as ceil(transfer / Delta) makes the slot
    bookkeeping follow the hop's duration, which the default does not do when
    a transfer is longer than a slot.

    Returns (route as stochastic contacts, the model's own predicted success
    probability). That prediction is the coarse model's belief, not a claim
    about the true laws; scoring the route under eq. (1) is the caller's job.
    """
    n_slots = max(1, int(math.ceil((deadline - t0) / slot_width)))
    if q is None:
        q = bernoulli_projection(graph, n_slots, slot_width, size, m, rng, t0=t0)
    def land(cid: int) -> int:
        return advance.get(cid, 1) if isinstance(advance, dict) else advance

    outgoing: dict[int, list[StochasticContact]] = {}
    for c in graph.contacts:
        outgoing.setdefault(c.frm, []).append(c)
    # usable slots per contact, ascending: the search visits exactly the
    # (contact, slot) pairs with q above the floor, in the same order as a
    # scan over every slot would, so routes are unchanged and long grids stay
    # tractable on large plans
    usable = {cid: np.flatnonzero(v > Q_FLOOR) for cid, v in q.items()}

    start = (src, 0)
    best: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[tuple[int, int], StochasticContact]] = {}
    pq: list[tuple[float, int, int]] = [(0.0, src, 0)]
    goal: tuple[int, int] | None = None

    while pq:
        cost, u, k = heapq.heappop(pq)
        if cost > best.get((u, k), math.inf) + 1e-12:
            continue
        if u == dst:
            goal = (u, k)
            break
        for c in outgoing.get(u, []):
            qc = q[c.cid]
            js = usable[c.cid]
            # the hop may be taken in any slot at or after the current one
            for j in js[bisect.bisect_left(js, k):]:
                j = int(j)
                if j >= n_slots:
                    break
                qj = qc[j]
                nxt = (c.to, j + land(c.cid))
                if nxt[1] > n_slots:
                    continue
                cand = cost - math.log(qj)
                if cand < best.get(nxt, math.inf) - 1e-12:
                    best[nxt] = cand
                    parent[nxt] = ((u, k), c)
                    heapq.heappush(pq, (cand, nxt[0], nxt[1]))

    if goal is None:
        return None, 0.0

    route: list[StochasticContact] = []
    node = goal
    while node != start:
        prev, c = parent[node]
        route.append(c)
        node = prev
    route.reverse()
    return route, math.exp(-best[goal])


def bernoulli_select(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    slot_width: float,
    budget: int,
    rng: np.random.Generator,
    m: int = 4000,
) -> list[list[StochasticContact]]:
    """Budget `budget` copies under the independent-Bernoulli belief (eq. 12).

    Greedy on 1 - prod(1 - p_j) using the model's own per-route predictions,
    with used contacts suppressed between picks so the copies are disjoint.
    Because the belief is independent, this policy budgets replication by
    existence probability but cannot see correlated failure: that gap is the
    point of the UC3 comparison.
    """
    n_slots = max(1, int(math.ceil((deadline - t0) / slot_width)))
    q = bernoulli_projection(graph, n_slots, slot_width, size, m, rng, t0=t0)

    chosen: list[list[StochasticContact]] = []
    used: set[int] = set()
    for _ in range(budget):
        sub = StochasticContactGraph(
            num_nodes=graph.num_nodes,
            contacts=[c for c in graph.contacts if c.cid not in used],
            horizon=graph.horizon,
            latent=graph.latent,
        )
        route, _p = bernoulli_route(
            sub, src, dst, size, t0, deadline, slot_width, rng, m=m,
            q={cid: v for cid, v in q.items() if cid not in used},
        )
        if route is None:
            break
        chosen.append(route)
        used.update(c.cid for c in route)
    return chosen
