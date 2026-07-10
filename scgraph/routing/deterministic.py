"""Deterministic earliest-arrival routing on a realized contact plan.

`cgr_route` is a CGR/SABR-style earliest-arrival Dijkstra over contacts (the
baseline router, and the router run inside the Monte-Carlo loop). Its
correctness anchor is the A-SABR cross-check in routing/asabr.py: on the same
point-mass plan both must return the same delivery time.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

from scgraph.contact import DetContact
from scgraph.graph import Realization


def cgr_route(
    real: Realization,
    src: int,
    dst: int,
    size: float,
    t0: float,
    excluded_cids: frozenset[int] = frozenset(),
) -> tuple[list[DetContact] | None, float]:
    """Earliest-arrival route (SABR distance: arrival, then hop count).

    Returns (route as a contact list, arrival time), or (None, inf).
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
            tau = max(a, c.t_start)              # eq. (2)
            e = tau + size / c.rate              # eq. (3)
            if e > c.t_end:
                continue
            a2, h2 = e + c.owlt, hops + 1        # eq. (4)
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
