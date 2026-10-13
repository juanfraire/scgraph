"""Hop-by-hop execution with local repair: how a deployed CGR node behaves (revision).

A committed source route fails as soon as one of its contacts is missed. A
deployed CGR/SABR implementation (ION, for instance) does not work that way:
every node that holds the bundle recomputes a route over its copy of the plan,
and a node whose planned contact fails to materialise, or closes before the
transfer completes, routes again from where it is. On a two-hop topology the
difference is small; on a deep route it is the difference between losing the
bundle and paying a detour.

`execute_reactive` runs that behaviour against one realization:

  * the node follows its current route while the realized contacts deliver;
  * a contact absent from the realization is discovered when it should have
    opened, a transfer cut off by a closing window when the window closes,
    and a window already shut on arrival at once;
  * the node then excludes that contact and recomputes a route over the
    NOMINAL plan from its own position and the discovery time.

`first_route` lets the source commit to a route computed elsewhere (the
SLA-aware planner's choice), which the forwarding plane then repairs with the
same local rule when it breaks. With `first_route=None` the policy is plain
reactive CGR on the nominal plan.
"""

from __future__ import annotations

import math

from scgraph.graph import Realization
from scgraph.routing.deterministic import cgr_route


def execute_reactive(
    real: Realization,
    nominal: Realization,
    src: int,
    dst: int,
    size: float,
    t0: float,
    first_route: list[int] | None = None,
    max_steps: int = 40,
    replan=None,
) -> tuple[float, int]:
    """(delivery time or inf, number of repairs) for one realization.

    `replan(node, time, excluded) -> list[int] | None` overrides the repair
    rule; the default recomputes CGR over the nominal plan. Passing an
    SLA-aware re-planner gives a forwarding plane that repairs under the laws.
    """
    avail = {c.cid: c for c in real.contacts}
    plan = {c.cid: c for c in nominal.contacts}
    node, a = src, t0
    excluded: set[int] = set()
    route = list(first_route) if first_route else None
    repairs = 0
    for _ in range(max_steps):
        if node == dst:
            return a, repairs
        if not route:
            if replan is not None:
                route = replan(node, a, frozenset(excluded))
                if not route:
                    return math.inf, repairs
            else:
                r, _arr = cgr_route(nominal, node, dst, size, a,
                                    excluded_cids=frozenset(excluded))
                if r is None:
                    return math.inf, repairs
                route = [c.cid for c in r]
        cid = route[0]
        c = avail.get(cid)
        if c is None or c.frm != node:
            # never materialised: noticed when it should have opened
            nom = plan.get(cid)
            a = max(a, nom.t_start if nom is not None else a)
            excluded.add(cid)
            route, repairs = None, repairs + 1
            continue
        tau = max(a, c.t_start)
        e = tau + size / c.rate
        if e > c.t_end:
            # cut off by the closing window (or already closed on arrival)
            a = max(a, c.t_end) if tau < c.t_end else a
            excluded.add(cid)
            route, repairs = None, repairs + 1
            continue
        a = e + c.owlt
        node = c.to
        route = route[1:]
    return (a if node == dst else math.inf), repairs


def sla_replanner(graph, dst: int, size: float, deadline: float, rng,
                  m: int = 300, n_realizations: int = 8):
    """A repair rule that re-plans under the laws: argmax P[deliver by D]
    from the current node and time, over candidates avoiding the contacts
    already found failed. Cached per (node, 5 s bucket, excluded set)."""
    from scgraph.routing.quantile import pick_route

    cache: dict = {}

    def replan(node, a, excluded):
        key = (node, round(a / 5.0), excluded)
        if key in cache:
            return cache[key]
        sub = graph.replaced([c for c in graph.contacts if c.cid not in excluded])
        r, _ = pick_route(sub, node, dst, size, a, deadline, rng, m=m,
                          n_realizations=n_realizations)
        out = [c.cid for c in r] if r else None
        cache[key] = out
        return out

    return replan


def success_under_repair(graph, route_cids, nominal, src, dst, size, t0, deadline,
                         reals) -> float:
    """P[deliver by D] of committing to `route_cids` and repairing with CGR,
    estimated over the given realizations."""
    hits = 0
    for real in reals:
        a, _ = execute_reactive(real, nominal, src, dst, size, t0, first_route=route_cids)
        hits += a <= deadline
    return hits / max(1, len(reals))


def pick_route_under_repair(graph, cands, src, dst, size, t0, deadline, rng,
                            m: int = 300, nominal=None):
    """The planning plane scoring candidates under the forwarding plane's
    actual behaviour: argmax over candidates of P[deliver by D] when the
    route is followed and repaired hop by hop with CGR. All candidates are
    scored on the same m realizations (paired). Returns (route, p_hat)."""
    nominal = nominal if nominal is not None else graph.mean_plan()
    reals = graph.realize_batch(rng, m)
    best, best_p = None, -1.0
    for r in cands:
        p = success_under_repair(graph, [c.cid for c in r], nominal, src, dst, size,
                                 t0, deadline, reals)
        if p > best_p + 1e-12:
            best, best_p = r, p
    return best, best_p
