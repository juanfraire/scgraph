"""THE correctness anchor (PLAN.md 7.1): on point-mass graphs, scgraph's
router must reproduce A-SABR's delivery times exactly.

Random deterministic graphs are generated, routed by both engines, and the
arrival times compared. This test gates everything; if asabr-cli is not
built the test fails loudly rather than skipping silently.
"""

import math

import numpy as np
import pytest

from scgraph.contact import StochasticContact
from scgraph.graph import StochasticContactGraph
from scgraph.routing import asabr
from scgraph.routing.deterministic import cgr_route


def random_pointmass_graph(rng, n_nodes=8, n_contacts=40, horizon=1000):
    # integer-valued attributes: A-SABR's plan lexer only accepts i64 tokens
    g = StochasticContactGraph(num_nodes=n_nodes)
    cid = 0
    for _ in range(n_contacts):
        frm, to = rng.choice(n_nodes, size=2, replace=False)
        t_s = int(rng.integers(0, int(horizon * 0.8)))
        dur = int(rng.integers(20, 200))
        rate = int(rng.integers(1, 100))
        owlt = int(rng.integers(0, 6))
        g.add(
            StochasticContact.deterministic(
                cid, int(frm), int(to), t_s, t_s + dur, rate, owlt=owlt
            )
        )
        cid += 1
    return g


@pytest.mark.skipif(not asabr.available(), reason="asabr-cli binary not built")
@pytest.mark.parametrize("seed", range(20))
def test_pointmass_anchor(seed):
    rng = np.random.default_rng(seed)
    g = random_pointmass_graph(rng)
    real = g.mean_plan()
    size = float(rng.uniform(0.5, 50.0))
    src, dst = 0, g.num_nodes - 1

    route, arrival_py = cgr_route(real, src, dst, size, t0=0.0)
    arrival_rs, path_rs = asabr.route_arrival(real, src, dst, size, t0=0.0)

    if route is None:
        assert math.isinf(arrival_rs)
    else:
        assert arrival_py == pytest.approx(arrival_rs, rel=1e-9, abs=1e-6)


def test_anchor_binary_is_built():
    """The anchor must actually run: a missing binary should be fixed, not skipped."""
    assert asabr.available(), (
        "asabr-cli not built; run `cargo build --release` in rust/asabr-cli"
    )
