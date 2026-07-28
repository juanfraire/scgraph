"""Reading a Contact Plan Designer export as the deterministic skeleton.

What matters here is that the skeleton really is the classical contact plan:
point-mass laws throughout (A1), node ids mapped without losing the operator's
numbering, and the two CPD export formats agreeing with each other.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from scgraph.graph import StochasticContactGraph
from scgraph.io import load_cpd_plan
from scgraph.routing.deterministic import cgr_route
from scgraph.scenarios.uc1_event import CPD_PLAN, EVENT_SITE, GATEWAY, load_skeleton

pytestmark = pytest.mark.skipif(
    not CPD_PLAN.exists(), reason="CPD skeleton export not present"
)


def test_skeleton_loads_and_is_point_mass():
    """A1: the orbital layer carries no timing or existence uncertainty."""
    sk = load_skeleton()
    assert sk.contacts, "expected contacts"
    g = StochasticContactGraph(sk.num_nodes, list(sk.contacts))
    assert g.is_point_mass
    assert all(c.layer == "space" for c in sk.contacts)
    assert all(c.p_exist == 1.0 for c in sk.contacts)


def test_node_mapping_round_trips():
    sk = load_skeleton()
    for cpd_id in (EVENT_SITE, GATEWAY):
        assert sk.label[sk.node(cpd_id)] == cpd_id
    assert set(sk.index.values()) == set(range(sk.num_nodes))
    with pytest.raises(KeyError):
        sk.node(999999)


def test_contacts_are_feasible_and_ordered():
    sk = load_skeleton()
    starts = [c.start.mean() for c in sk.contacts]
    assert starts == sorted(starts), "contacts should load in plan order"
    for c in sk.contacts:
        assert c.duration.mean() >= 0.0          # A5
        assert c.rate.mean() > 0.0
        assert 0.0 <= c.owlt < 1.0               # near-Earth OWLT is milliseconds
        assert c.start.mean() + c.duration.mean() <= sk.horizon + 1e-6


def test_json_and_ion_exports_agree():
    """The two CPD formats describe the same plan, up to ION's integer rounding."""
    j = load_cpd_plan(CPD_PLAN / "plan.json")
    i = load_cpd_plan(CPD_PLAN / "ionFormatContactPlan.txt")
    assert i.num_nodes == j.num_nodes
    assert len(i.contacts) == len(j.contacts)
    for cj, cinit in zip(j.contacts, i.contacts):
        assert (cj.frm, cj.to) == (cinit.frm, cinit.to)
        # ION writes whole seconds, so edges may move by up to 1 s
        assert abs(cj.start.mean() - cinit.start.mean()) <= 1.0
        assert abs(cj.rate.mean() - cinit.rate.mean()) <= 1.0


def test_keep_nodes_restricts_the_plan():
    full = load_skeleton()
    gs = {EVENT_SITE, GATEWAY}
    sats = {k for k in full.index if k not in gs}
    one = sorted(sats)[:1]
    sub = load_cpd_plan(CPD_PLAN, keep_nodes=gs | set(one))
    assert sub.num_nodes <= 3
    assert len(sub.contacts) < len(full.contacts)


def test_a_real_store_carry_forward_route_exists():
    """The skeleton must actually connect the two sites, via a satellite."""
    sk = load_skeleton()
    src, dst = sk.node(EVENT_SITE), sk.node(GATEWAY)
    g = StochasticContactGraph(sk.num_nodes, list(sk.contacts))
    route, arr = cgr_route(g.mean_plan(), src, dst, size=100.0, t0=0.0)
    assert route is not None and math.isfinite(arr)
    assert len(route) >= 2, "expected a relayed path, not a direct link"
    assert route[0].frm == src and route[-1].to == dst
    # every intermediate node is a spacecraft, not the other ground station
    for c in route[:-1]:
        assert sk.label[c.to] not in (EVENT_SITE, GATEWAY)


def test_shifting_the_origin_preserves_structure():
    sk = load_skeleton()
    dt = 100.0
    sh = sk.shifted(dt)
    assert len(sh.contacts) == len(sk.contacts)
    for a, b in zip(sk.contacts, sh.contacts):
        assert b.start.mean() == pytest.approx(a.start.mean() - dt)
        assert b.duration.mean() == pytest.approx(a.duration.mean())


def test_instances_use_real_geometry_and_vary():
    """Each instance is a different release time into the same real plan."""
    from scgraph.scenarios.uc1_event import PASS_BAND, make_instance_cpd

    rng = np.random.default_rng(3)
    t0s, waits = [], []
    for _ in range(10):
        g, p = make_instance_cpd(rng, sigma=20.0)
        t0s.append(p["t0"])
        waits.append(p["pass_wait"])
        assert PASS_BAND[0] <= p["pass_wait"] <= PASS_BAND[1]
        assert p["deadline"] > p["t0"]
        # air relays were injected on top of the orbital skeleton
        assert any(c.layer == "air" for c in g.contacts)
        assert any(c.layer == "space" for c in g.contacts)
    assert len(set(t0s)) == len(t0s), "release times should differ per instance"
    assert np.std(waits) > 1.0, "pass geometry should vary across instances"


def test_empty_selection_is_an_error_not_an_empty_plan():
    with pytest.raises(ValueError):
        load_cpd_plan(CPD_PLAN, keep_nodes={EVENT_SITE})
