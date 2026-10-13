"""Multi-bundle executor: reduction to eq. (4), FIFO queueing, storage, windows."""

import math

import numpy as np
import pytest

from scgraph.arrival import delivery_time
from scgraph.contact import DetContact
from scgraph.flows import (
    Bundle, assign_contention_aware, book_routes, cgr_route_booked, contact_map,
    run_flows,
)
from scgraph.graph import Realization
from scgraph.routing.deterministic import cgr_route


def det(cid, frm, to, ts, te, rate, owlt=0.0):
    return DetContact(cid, frm, to, ts, te, rate, owlt)


def plan(*contacts, nodes=8):
    return Realization(nodes, list(contacts))


# ------------------------------------------------- single bundle = eq. (4)

def test_single_bundle_matches_delivery_time_random_chains():
    rng = np.random.default_rng(7)
    infeasible = 0
    for trial in range(400):
        hops = int(rng.integers(1, 6))
        route, t = [], 0.0
        for h in range(hops):
            ts = t + float(rng.uniform(-20, 40))
            te = ts + float(rng.uniform(1, 60))
            route.append(det(h, h, h + 1, ts, te, float(rng.uniform(0.5, 5)),
                             float(rng.uniform(0, 3))))
            t = ts
        size, t0 = float(rng.uniform(1, 40)), float(rng.uniform(-10, 10))
        ref = delivery_time(route, size, t0)
        b = Bundle(0, hops, size, t0, math.inf, tuple(c.cid for c in route))
        for buf in (math.inf, 1):
            res = run_flows(plan(*route, nodes=hops + 1), [b], buffer=buf)
            if math.isfinite(ref):
                assert res.delivery[0] == pytest.approx(ref, abs=1e-9)
                assert res.cause == ["ok"]
            else:
                assert res.delivery[0] == math.inf
                assert res.cause == ["window"]
        infeasible += not math.isfinite(ref)
    assert 20 < infeasible < 380        # both branches exercised


def test_single_bundle_matches_on_uc1_realizations():
    from scgraph.routing.quantile import candidate_routes
    from scgraph.scenarios.uc1_event import make_instance_cpd

    rng = np.random.default_rng(3)
    g, p = make_instance_cpd(rng, sigma=40.0)
    cands = candidate_routes(g, p["src"], p["dst"], p["size"], p["t0"], rng, 10)
    for _ in range(20):
        real = g.realize(rng)
        cmap = contact_map(real)
        for r in cands:
            ref = delivery_time([cmap[c.cid] for c in r], p["size"], p["t0"])
            b = Bundle(p["src"], p["dst"], p["size"], p["t0"], p["deadline"]).on(r)
            assert run_flows(cmap, [b]).delivery[0] == pytest.approx(ref)


# ------------------------------------------------------------- FIFO queues

def test_fifo_queueing_on_shared_contact():
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0, owlt=2.0))
    bs = [Bundle(0, 1, 10.0, 0.0, 100.0, (0,)) for _ in range(3)]
    res = run_flows(pl, bs)
    assert list(res.delivery) == pytest.approx([12.0, 22.0, 32.0])


def test_fifo_is_by_arrival_not_by_index():
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0))
    late_first = Bundle(0, 1, 10.0, 5.0, 100.0, (0,))     # index 0, released at 5
    early = Bundle(0, 1, 10.0, 0.0, 100.0, (0,))          # index 1, released at 0
    res = run_flows(pl, [late_first, early])
    assert list(res.delivery) == pytest.approx([20.0, 10.0])


def test_queue_waits_for_window_and_relay_pipelines():
    # two hops; the second window opens late, so both bundles wait at the relay
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0, owlt=1.0),
              det(1, 1, 2, 50.0, 100.0, 1.0, owlt=1.0))
    bs = [Bundle(0, 2, 10.0, 0.0, 200.0, (0, 1)) for _ in range(2)]
    res = run_flows(pl, bs)
    assert list(res.delivery) == pytest.approx([61.0, 71.0])
    assert res.peak_occupancy[1] == 2


# ---------------------------------------------------------- finite storage

def test_buffer_drop_at_relay():
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0), det(1, 1, 2, 50.0, 100.0, 1.0))
    bs = [Bundle(0, 2, 10.0, 0.0, 200.0, (0, 1)) for _ in range(3)]
    res = run_flows(pl, bs, buffer=2)
    assert res.cause == ["ok", "ok", "buffer"]
    assert res.lost_node[2] == 1
    assert list(res.delivery[:2]) == pytest.approx([60.0, 70.0])
    assert run_flows(pl, bs, buffer=3).cause == ["ok"] * 3


def test_slot_freed_at_departure_is_reusable():
    # the relay forwards at the rate it receives, so each departure from the
    # relay coincides with the next arrival: departures go first at equal
    # times, and B = 1 is enough
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0), det(1, 1, 2, 0.0, 100.0, 1.0))
    bs = [Bundle(0, 2, 10.0, 0.0, 200.0, (0, 1)) for _ in range(3)]
    res = run_flows(pl, bs, buffer=1)
    assert res.cause == ["ok"] * 3
    assert list(res.delivery) == pytest.approx([20.0, 30.0, 40.0])
    assert res.peak_occupancy[1] == 1


def test_source_and_destination_are_unbounded():
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0))
    bs = [Bundle(0, 1, 10.0, 0.0, 200.0, (0,)) for _ in range(5)]
    res = run_flows(pl, bs, buffer=0)
    assert res.cause == ["ok"] * 5


# ------------------------------------------------------------- window cut

def test_window_cut_off():
    pl = plan(det(0, 0, 1, 0.0, 25.0, 1.0))
    bs = [Bundle(0, 1, 10.0, 0.0, 100.0, (0,)) for _ in range(4)]
    res = run_flows(pl, bs)
    assert res.cause == ["ok", "ok", "window", "window"]
    assert list(res.delivery[:2]) == pytest.approx([10.0, 20.0])


def test_cut_transfer_holds_contact_until_window_end():
    # a big bundle starts, is cut at 25; the small one behind it cannot use
    # the remaining 5 s (an atomic sender does not know the window will close)
    pl = plan(det(0, 0, 1, 0.0, 25.0, 1.0))
    bs = [Bundle(0, 1, 18.0, 0.0, 100.0, (0,)), Bundle(0, 1, 10.0, 0.0, 100.0, (0,)),
          Bundle(0, 1, 1.0, 0.0, 100.0, (0,))]
    res = run_flows(pl, bs)
    assert res.cause == ["ok", "window", "window"]


def test_missing_contact_and_noroute():
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0))
    bs = [Bundle(0, 2, 10.0, 0.0, 100.0, (0, 9)), Bundle(0, 2, 10.0, 0.0, 100.0, ())]
    res = run_flows(pl, bs, buffer=1)
    assert res.cause == ["missing", "noroute"]
    assert res.lost_node[0] == 1


def test_route_must_chain():
    pl = plan(det(0, 0, 1, 0.0, 100.0, 1.0), det(1, 3, 2, 0.0, 100.0, 1.0))
    with pytest.raises(ValueError):
        run_flows(pl, [Bundle(0, 2, 1.0, 0.0, 100.0, (0, 1))])


# ------------------------------------------------------ policy helpers

def test_booked_cgr_matches_cgr_without_bookings_and_spreads_with():
    fast = det(0, 0, 1, 0.0, 20.0, 1.0)       # room for two bundles of 10
    slow = det(1, 0, 1, 30.0, 200.0, 1.0)
    pl = plan(fast, slow)
    r0, a0 = cgr_route_booked(pl, 0, 1, 10.0, 0.0, {})
    r1, a1 = cgr_route(pl, 0, 1, 10.0, 0.0)
    assert [c.cid for c in r0] == [c.cid for c in r1] and a0 == a1
    tmpl = [Bundle(0, 1, 10.0, 0.0, 100.0) for _ in range(4)]
    routes, booked = book_routes(pl, tmpl)
    assert routes == [(0,), (0,), (1,), (1,)]
    assert booked == {0: 20.0, 1: 20.0}


def test_contention_aware_assignment_splits_load():
    # route A delivers one bundle early; route B is slower but still on time
    a = det(0, 0, 1, 0.0, 10.0, 1.0)
    b = det(1, 0, 1, 50.0, 200.0, 1.0)
    cmap = contact_map(plan(a, b))
    tmpl = [Bundle(0, 1, 10.0, 0.0, 100.0) for _ in range(2)]
    routes, log = assign_contention_aware([(0,), (1,)], tmpl, [cmap])
    assert routes == [(0,), (1,)]
    assert log[1]["p_own"] == 1.0 and log[1]["exp_ontime"] == 2.0
    # a contention-blind router puts both on A and loses one
    res = run_flows(cmap, [t.on((0,)) for t in tmpl])
    assert res.cause == ["ok", "window"]
