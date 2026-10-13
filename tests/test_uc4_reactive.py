"""Revision: reactive (hop-by-hop repair) execution, batched realizations, UC4."""

import math

import numpy as np
import pytest

from scgraph.contact import DetContact, StochasticContact
from scgraph.distributions import Dirac, TruncNormal
from scgraph.graph import Realization, StochasticContactGraph
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.reactive import execute_reactive, pick_route_under_repair


def _plan():
    """0 -> 1 -> 3 is fast; 0 -> 2 -> 3 is the slower alternative."""
    cs = [
        DetContact(0, 0, 1, 0.0, 50.0, 10.0),
        DetContact(1, 1, 3, 20.0, 70.0, 10.0),
        DetContact(2, 0, 2, 0.0, 100.0, 10.0),
        DetContact(3, 2, 3, 100.0, 200.0, 10.0),
    ]
    return Realization(4, cs)


def test_reactive_equals_cgr_when_nothing_fails():
    plan = _plan()
    _, arr = cgr_route(plan, 0, 3, 100.0, 0.0)
    got, repairs = execute_reactive(plan, plan, 0, 3, 100.0, 0.0)
    assert got == pytest.approx(arr)
    assert repairs == 0


def test_reactive_repairs_a_missing_contact():
    plan = _plan()
    real = Realization(4, [c for c in plan.contacts if c.cid != 1])   # 1->3 absent
    got, repairs = execute_reactive(real, plan, 0, 3, 100.0, 0.0)
    # the bundle reached node 1, found 1->3 missing at its nominal start (20 s),
    # and node 1 has no other way out: lost
    assert math.isinf(got) and repairs == 1
    # had the source committed to the alternative, it would have delivered
    got2, _ = execute_reactive(real, plan, 0, 3, 100.0, 0.0, first_route=[2, 3])
    assert got2 == pytest.approx(110.0)


def test_reactive_window_cut_discovered_at_window_end():
    plan = _plan()
    # the source's first choice 0->1 closes early in reality: transfer cut at t=5
    real = Realization(4, [DetContact(0, 0, 1, 0.0, 5.0, 10.0)] + plan.contacts[1:])
    got, repairs = execute_reactive(real, plan, 0, 3, 100.0, 0.0)
    # repair at t=5 onto 0->2 (until 15) then 2->3 at 100 -> 110
    assert got == pytest.approx(110.0)
    assert repairs == 1


def test_realize_batch_matches_realize_in_law():
    g = StochasticContactGraph(2, [
        StochasticContact(0, 0, 1, TruncNormal(100.0, 10.0, 60.0, 140.0), Dirac(30.0),
                          Dirac(1.0), p_exist=0.7),
    ])
    rb = g.realize_batch(np.random.default_rng(0), 20_000)
    rr = [g.realize(np.random.default_rng(i)) for i in range(3000)]
    pb = np.mean([len(r.contacts) for r in rb])
    pr = np.mean([len(r.contacts) for r in rr])
    assert pb == pytest.approx(0.7, abs=0.01) and pr == pytest.approx(0.7, abs=0.03)
    sb = np.mean([r.contacts[0].t_start for r in rb if r.contacts])
    assert sb == pytest.approx(100.0, abs=0.3)


def test_uc4_instance_structure():
    from scgraph.scenarios.uc4_continuum import make_instance

    g, p = make_instance(np.random.default_rng(3), 40.0)
    kinds = {c.kind for c in g.contacts}
    assert {"uav", "direct", "isl_in", "isl_x", "feeder", "haps_up"} <= kinds
    uav = [c for c in g.contacts if c.kind == "uav"]
    assert all(c.corr is not None for c in uav)
    feeders = [c for c in g.contacts if c.kind == "feeder"]
    assert all(c.factor == "wx_gw" and c.factor_rho > 0 for c in feeders)
    assert p["deadline"] >= p["reliable_arrival"]
    assert p["nominal_hops"] >= 4
    # the deterministic nominal route meets its own nominal promise
    r, arr = cgr_route(g.mean_plan(), p["src"], p["dst"], p["size"], p["t0"])
    assert arr == pytest.approx(p["nominal_arrival"])


def test_pick_route_under_repair_returns_a_candidate():
    from scgraph.routing.quantile import candidate_routes
    from scgraph.scenarios.uc4_continuum import make_instance

    rng = np.random.default_rng(4)
    g, p = make_instance(rng, 40.0)
    cands = candidate_routes(g, p["src"], p["dst"], p["size"], p["t0"], rng, 5)
    r, ph = pick_route_under_repair(g, cands, p["src"], p["dst"], p["size"], p["t0"],
                                    p["deadline"], rng, m=50)
    assert r in cands and 0.0 <= ph <= 1.0
