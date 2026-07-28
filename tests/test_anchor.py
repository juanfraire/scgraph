"""THE correctness anchor: on point-mass graphs, scgraph's router must
reproduce A-SABR's delivery times exactly.

This is the equality half of the paper's subsumption claim (Sec. III-A): the
deterministic contact plan is not merely *described* as a corner of the
stochastic contact graph, it is checked to be one, against an independent
implementation. Random deterministic graphs are generated, routed by both
engines, and the arrival times compared.

The anchor needs the Rust CLI (scripts/build_anchor.sh, or `make anchor`).
Because that pulls in a Rust toolchain and a network clone, a clone without it
SKIPS these tests rather than failing: a reader reproducing the paper's numbers
needs none of it. Set SCGRAPH_REQUIRE_ANCHOR=1 to turn the skip into a failure,
which is what CI and the maintainer's machine do, so the anchor can never be
silently lost.
"""

import math
import os

import numpy as np
import pytest

from scgraph.contact import StochasticContact
from scgraph.graph import StochasticContactGraph
from scgraph.routing import asabr
from scgraph.routing.deterministic import cgr_route

REQUIRED = os.environ.get("SCGRAPH_REQUIRE_ANCHOR") == "1"

_MISSING = (
    "asabr-cli not built, so the A-SABR cross-check cannot run.\n"
    "  Build it with:  scripts/build_anchor.sh   (or: make anchor)\n"
    "  It needs a Rust toolchain and clones A-SABR at a pinned revision.\n"
    "  Nothing in results/ or figures/ depends on it: `make reproduce`\n"
    "  regenerates every number in the paper without it.\n"
    "  Set SCGRAPH_REQUIRE_ANCHOR=1 to make this a hard failure instead."
)

# One decision for the whole module, so the skip reason is stated once and the
# opt-in flag cannot disagree between the two tests below.
anchor_required = pytest.mark.skipif(
    not asabr.available() and not REQUIRED, reason=_MISSING
)


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


@anchor_required
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


@anchor_required
def test_anchor_binary_is_built():
    """Under SCGRAPH_REQUIRE_ANCHOR=1 a missing binary is a failure, not a skip."""
    assert asabr.available(), _MISSING
