import math

import numpy as np
import pytest

from scgraph.arrival import delivery_time, sample_route_deliveries
from scgraph.contact import DetContact, StochasticContact
from scgraph.distributions import Dirac


def det(cid, frm, to, ts, te, rate, owlt=0.0):
    return DetContact(cid, frm, to, ts, te, rate, owlt)


def test_simple_chain():
    # matches the hand-checked A-SABR example: start 60, rate 1e4, owlt 10
    route = [det(0, 0, 1, 60, 7260, 10000, 10), det(1, 1, 4, 60, 7260, 10000, 10)]
    t = delivery_time(route, size=1.0, t0=0.0)
    assert t == pytest.approx(60 + 1e-4 + 10 + 1e-4 + 10)


def test_wait_for_window():
    route = [det(0, 0, 1, 100, 200, 1.0)]
    assert delivery_time(route, size=10.0, t0=0.0) == pytest.approx(110.0)


def test_window_closes_mid_transfer():
    route = [det(0, 0, 1, 100, 105, 1.0)]  # needs 10 s, window has 5
    assert delivery_time(route, size=10.0, t0=0.0) == math.inf


def test_arrives_after_window():
    route = [det(0, 0, 1, 0, 50, 1.0)]
    assert delivery_time(route, size=10.0, t0=45.0) == math.inf


def test_owlt_added_after_transmission():
    route = [det(0, 0, 1, 0, 100, 2.0, owlt=7.0)]
    assert delivery_time(route, size=10.0, t0=0.0) == pytest.approx(5.0 + 7.0)


def test_point_mass_sampler_matches_scalar():
    c1 = StochasticContact(0, 0, 1, Dirac(60.0), Dirac(7200.0), Dirac(10000.0), owlt=10.0)
    c2 = StochasticContact(1, 1, 2, Dirac(60.0), Dirac(7200.0), Dirac(10000.0), owlt=10.0)
    t = sample_route_deliveries([c1, c2], size=1.0, t0=0.0, m=64, rng=np.random.default_rng(1))
    scalar = delivery_time([c1.mean_det(), c2.mean_det()], size=1.0, t0=0.0)
    assert np.allclose(t, scalar)
