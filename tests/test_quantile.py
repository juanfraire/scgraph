"""M1 consistency: analytic CDF propagation vs Monte Carlo, and eq. (8)'s
quantile/chance-constraint equivalence on a hand-computable case."""

import numpy as np
import pytest

from scgraph.analytic import route_delivery_cdf, route_quantile
from scgraph.arrival import sample_route_deliveries
from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac, TruncNormal


def jittered_route():
    # hop 1: start ~ TruncNormal(100, 20) on [40, 160], window 60s, rate 10, size 100
    c1 = StochasticContact(
        cid=0, frm=0, to=1,
        start=TruncNormal(100.0, 20.0, lo=40.0, hi=160.0),
        duration=Dirac(60.0), rate=Dirac(10.0), owlt=1.0,
    )
    # hop 2: start ~ TruncNormal(150, 10) on [120, 180], window 40s
    c2 = StochasticContact(
        cid=1, frm=1, to=2,
        start=TruncNormal(150.0, 10.0, lo=120.0, hi=180.0),
        duration=Dirac(40.0), rate=Dirac(10.0), owlt=1.0,
    )
    return [c1, c2]


def test_analytic_matches_mc():
    route = jittered_route()
    size, t0 = 100.0, 0.0
    t_grid = np.linspace(0.0, 400.0, 4001)
    F = route_delivery_cdf(route, size, t0, t_grid)

    rng = np.random.default_rng(11)
    t = sample_route_deliveries(route, size, t0, 400_000, rng)

    for x in (150.0, 180.0, 200.0, 250.0):
        mc = float(np.mean(t <= x))
        an = float(np.interp(x, t_grid, F))
        assert an == pytest.approx(mc, abs=0.006), f"t={x}: analytic {an} vs MC {mc}"

    # success mass (t -> inf) matches MC feasibility rate
    assert F[-1] == pytest.approx(np.isfinite(t).mean(), abs=0.006)


def test_quantile_equivalence_eq8():
    """Q_{1-eps}(pi) <= D  <=>  P[T <= D] >= 1-eps (eq. 8)."""
    route = jittered_route()
    size, t0, eps = 100.0, 0.0, 0.25
    t_grid = np.linspace(0.0, 400.0, 4001)
    F = route_delivery_cdf(route, size, t0, t_grid)
    q = route_quantile(route, size, t0, 1 - eps, t_max=400.0)
    if np.isfinite(q):
        assert float(np.interp(q, t_grid, F)) >= 1 - eps - 1e-2
        # D well below the quantile fails the chance constraint
        assert float(np.interp(q - 5.0, t_grid, F)) < 1 - eps


def test_pointmass_cdf_is_step():
    c = StochasticContact.deterministic(0, 0, 1, 60.0, 120.0, 10.0, owlt=5.0)
    t_grid = np.linspace(0.0, 200.0, 2001)
    F = route_delivery_cdf([c], 100.0, 0.0, t_grid)
    # delivery at exactly 75 (60 + 10 + 5): step CDF
    assert float(np.interp(74.0, t_grid, F)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.interp(76.0, t_grid, F)) == pytest.approx(1.0, abs=1e-9)
