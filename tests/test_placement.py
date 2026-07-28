"""The placement probe's execution driver (routing/placement.py).

The probe's conclusion rests on the three placements being executed fairly
against the same realizations, so what is pinned here is the driver's
semantics: contingency behaves as advertised, the local policy really is
local, and a placement can only fail for reasons the realization justifies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scgraph.contact import StochasticContact
from scgraph.graph import Realization, StochasticContactGraph
from scgraph.routing.placement import (
    DistributedPlan,
    GroundPlan,
    ProxyPlan,
    execute,
    placement_delivery,
    plan_proxy,
)


def _two_relay_graph(p_exist=1.0):
    """src=0, dst=1, two disjoint relays (2, 3), relay 2 nominally faster."""
    g = StochasticContactGraph(num_nodes=4)
    from scgraph.distributions import Dirac

    def pair(cid, relay, s1):
        g.add(StochasticContact(cid, 0, relay, Dirac(s1), Dirac(100.0), Dirac(10.0),
                                p_exist=p_exist))
        g.add(StochasticContact(cid + 1, relay, 1, Dirac(s1 + 20.0), Dirac(100.0),
                                Dirac(10.0), p_exist=p_exist))

    pair(0, 2, 10.0)
    pair(2, 3, 50.0)
    return g


def test_execute_delivers_on_a_feasible_realization():
    g = _two_relay_graph()
    rng = np.random.default_rng(0)
    real = g.realize(rng)
    plan = GroundPlan([0, 1])
    t = execute(real, 0, 1, size=50.0, t0=0.0, choose=plan.chooser())
    assert math.isfinite(t)


def test_ground_has_no_contingency():
    """With its route's first contact absent, the ground plan is stuck even
    though the other relay is available."""
    g = _two_relay_graph()
    rng = np.random.default_rng(1)
    real = g.realize(rng)
    real = Realization(real.num_nodes, [c for c in real.contacts if c.cid not in (0,)])
    stuck = execute(real, 0, 1, 50.0, 0.0, GroundPlan([0, 1]).chooser())
    assert stuck == math.inf


def test_proxy_falls_through_to_its_second_choice():
    """Same realization, but a ranked table reaches the surviving relay."""
    g = _two_relay_graph()
    rng = np.random.default_rng(1)
    real = g.realize(rng)
    real = Realization(real.num_nodes, [c for c in real.contacts if c.cid not in (0,)])
    table = {(0, 0): [[0, 1], [2, 3]], (3, 0): [[3]], (2, 0): [[1]]}
    proxy = ProxyPlan(size=50.0, t0=0.0, deadline=400.0, n_buckets=1, table=table)
    t = execute(real, 0, 1, 50.0, 0.0, proxy.chooser())
    assert math.isfinite(t)


def test_distributed_is_greedy_and_local():
    """It takes the earliest-arriving outgoing contact, ignoring the future."""
    g = _two_relay_graph()
    rng = np.random.default_rng(2)
    real = g.realize(rng)
    seen = []

    dist = DistributedPlan(size=50.0)
    base = dist.chooser()

    def spy(node, a, avail, used):
        c = base(node, a, avail, used)
        if c is not None:
            seen.append(c.cid)
        return c

    execute(real, 0, 1, 50.0, 0.0, spy)
    assert seen[0] == 0          # relay 2 opens at t=10, before relay 3 at t=50


def test_distributed_walks_into_a_dead_end_it_cannot_foresee():
    """Local ranking has no way to know the fast relay has no onward contact."""
    g = _two_relay_graph()
    rng = np.random.default_rng(3)
    real = g.realize(rng)
    # drop relay 2's onward hop: greedy still takes relay 2 and then strands
    real = Realization(real.num_nodes, [c for c in real.contacts if c.cid != 1])
    t = execute(real, 0, 1, 50.0, 0.0, DistributedPlan(50.0).chooser())
    assert t == math.inf


def test_plan_proxy_builds_entries_for_reachable_nodes():
    g = _two_relay_graph()
    rng = np.random.default_rng(5)
    plan = plan_proxy(g, 0, 1, 50.0, 0.0, 400.0, rng, m=100, n_realizations=4,
                      n_buckets=2)
    assert plan.table, "expected a non-empty table"
    assert all(isinstance(k, tuple) and len(k) == 2 for k in plan.table)
    assert plan.bucket(0.0) == 0 and plan.bucket(399.0) == 1


def test_all_placements_return_probabilities():
    g = _two_relay_graph(p_exist=0.8)
    rng = np.random.default_rng(7)
    out = placement_delivery(g, 0, 1, 50.0, 0.0, 400.0, rng, m=200, plan_m=200)
    assert set(out) == {"ground", "proxy", "dist"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_certain_graph_is_delivered_by_every_placement():
    g = _two_relay_graph(p_exist=1.0)
    rng = np.random.default_rng(11)
    out = placement_delivery(g, 0, 1, 50.0, 0.0, 400.0, rng, m=200, plan_m=200)
    for k, v in out.items():
        assert v == pytest.approx(1.0), f"{k} lost a bundle on a certain graph"
