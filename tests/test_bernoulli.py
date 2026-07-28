"""The Bernoulli-TEG projection and its router (routing/bernoulli.py).

The properties worth pinning are the ones the paper's subsumption claim rests
on: the projection is a correct marginal, it degenerates exactly on point-mass
graphs, and refining the slot grid recovers the SLA-aware choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac, TruncNormal
from scgraph.graph import StochasticContactGraph
from scgraph.routing.bernoulli import (
    bernoulli_projection,
    bernoulli_route,
    bernoulli_select,
    slot_usability,
)
from scgraph.scenarios.uc1_event import make_instance


def test_point_mass_contact_is_certain_while_open():
    """A Dirac contact has q=1 in every slot it can still drain the bundle."""
    c = StochasticContact.deterministic(0, 0, 1, t_start=0.0, t_end=100.0, rate=10.0)
    rng = np.random.default_rng(0)
    # bundle of 50 units at rate 10 needs 5 s, so slots starting at 0..95 work
    q = slot_usability(c, n_slots=10, slot_width=10.0, size=50.0, m=2000, rng=rng)
    assert q[:9] == pytest.approx(1.0)     # slot starts 0..80, all drain by 100
    assert q[9] == pytest.approx(1.0)      # slot start 90, drains at 95 <= 100


def test_point_mass_contact_is_zero_once_closed():
    c = StochasticContact.deterministic(0, 0, 1, t_start=0.0, t_end=40.0, rate=10.0)
    rng = np.random.default_rng(0)
    q = slot_usability(c, n_slots=8, slot_width=10.0, size=50.0, m=2000, rng=rng)
    # needs 5 s: slot starts 0,10,20,30 drain by 40; 40+ do not
    assert q[:4] == pytest.approx(1.0)
    assert q[4:] == pytest.approx(0.0)


def test_existence_probability_passes_through():
    """q is capped by p_exist: the projection keeps the existence marginal."""
    c = StochasticContact(
        0, 0, 1, start=Dirac(0.0), duration=Dirac(100.0), rate=Dirac(10.0),
        p_exist=0.6,
    )
    rng = np.random.default_rng(3)
    q = slot_usability(c, n_slots=5, slot_width=10.0, size=50.0, m=40000, rng=rng)
    assert q[0] == pytest.approx(0.6, abs=0.01)


def test_jitter_lowers_usability_at_the_boundary():
    """A jittered start is not reliably open at a slot boundary near it."""
    c = StochasticContact(
        0, 0, 1, start=TruncNormal(50.0, 20.0, lo=-20.0, hi=120.0),
        duration=Dirac(200.0), rate=Dirac(10.0),
    )
    rng = np.random.default_rng(5)
    q = slot_usability(c, n_slots=12, slot_width=10.0, size=50.0, m=20000, rng=rng)
    assert q[5] == pytest.approx(0.5, abs=0.03)   # slot start 50 = the mean
    assert q[0] < 0.05                            # slot start 0, rarely open yet
    assert q[11] > 0.95                           # slot start 110, almost surely open


def test_projection_covers_every_contact():
    rng = np.random.default_rng(7)
    g, _p = make_instance(rng, sigma=30.0)
    q = bernoulli_projection(g, n_slots=12, slot_width=30.0, size=100.0, m=500, rng=rng)
    assert set(q) == {c.cid for c in g.contacts}
    assert all(v.shape == (12,) for v in q.values())


def test_route_on_point_mass_graph_matches_the_deterministic_choice():
    """With no uncertainty the Bernoulli corner picks a feasible route."""
    g = StochasticContactGraph(num_nodes=3)
    g.add(StochasticContact.deterministic(0, 0, 1, 0.0, 100.0, 10.0))
    g.add(StochasticContact.deterministic(1, 1, 2, 20.0, 200.0, 10.0))
    rng = np.random.default_rng(11)
    route, pred = bernoulli_route(
        g, src=0, dst=2, size=50.0, t0=0.0, deadline=200.0,
        slot_width=10.0, rng=rng, m=500,
    )
    assert route is not None
    assert [c.cid for c in route] == [0, 1]
    assert pred == pytest.approx(1.0)


def test_unreachable_destination_returns_none():
    g = StochasticContactGraph(num_nodes=3)
    g.add(StochasticContact.deterministic(0, 0, 1, 0.0, 100.0, 10.0))
    rng = np.random.default_rng(13)
    route, pred = bernoulli_route(
        g, src=0, dst=2, size=50.0, t0=0.0, deadline=200.0,
        slot_width=10.0, rng=rng, m=200,
    )
    assert route is None and pred == 0.0


def test_refining_slots_recovers_the_safe_route():
    """The subsumption direction: a fine grid finds the timing-certain backbone
    that a coarse grid misses. This is figures/subsumption.pdf in miniature."""
    from scgraph.arrival import sample_route_deliveries

    rng = np.random.default_rng(20260727)
    coarse_viol, fine_viol = [], []
    for _ in range(25):
        g, p = make_instance(rng, sigma=60.0)
        src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
        for slot, acc in ((90.0, coarse_viol), (5.0, fine_viol)):
            r, _ = bernoulli_route(g, src, dst, size, t0, D, slot, rng, m=1500)
            if r is None:
                acc.append(1.0)
                continue
            t = sample_route_deliveries(r, size, t0, 1500, rng)
            acc.append(float(np.mean(~(t <= D))))
    assert np.mean(fine_viol) < np.mean(coarse_viol)
    assert np.mean(fine_viol) < 0.01


def test_select_returns_disjoint_routes():
    rng = np.random.default_rng(31)
    from scgraph.scenarios.uc3_emergency import make_instance as uc3_instance

    g, p = uc3_instance(rng, severity=0.4)
    routes = bernoulli_select(
        g, p["src"], p["dst"], p["size"], p["t0"], p["deadline"],
        slot_width=30.0, budget=2, rng=rng, m=1000,
    )
    assert len(routes) == 2
    a = {c.cid for c in routes[0]}
    b = {c.cid for c in routes[1]}
    assert a.isdisjoint(b)
