"""UC1 - large-scale mobile event: timing uncertainty (H1).

Topology per instance (randomized):
  node 0 = source terminal, nodes 1..K = air relays (HAPS/UAV),
  node K+1 = LEO satellite (backbone), node K+2 = destination gateway.

Air routes are nominally FAST but their contact windows jitter (TruncNormal
start, rigid window: A2); the space route is slower but timing-certain (A1).
A deterministic operator routes on the nominal (mean) plan and inherits the
air route's tight transfer; the SLA-aware router maximizes P[T <= D] (eq. 6)
and switches to slack or backbone routes as jitter grows. Hypothesis H1: the
SLA-violation gap widens with jitter.
"""

from __future__ import annotations

import numpy as np

from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac, TruncNormal
from scgraph.graph import StochasticContactGraph


def make_instance(
    rng: np.random.Generator, sigma: float, n_air: int = 3
) -> tuple[StochasticContactGraph, dict]:
    """One random UC1 instance at timing jitter level sigma (seconds)."""
    src = 0
    sat = n_air + 1
    dst = n_air + 2
    g = StochasticContactGraph(num_nodes=n_air + 3)
    size = 100.0  # bundle volume [Mb]
    cid = 0

    def timing(nominal: float) -> "TruncNormal | Dirac":
        if sigma <= 0.0:
            return Dirac(nominal)
        return TruncNormal(nominal, sigma, lo=nominal - 3.5 * sigma, hi=nominal + 3.5 * sigma)

    # --- air routes: nominally early, tight relay hand-offs, jittered ---
    for k in range(1, n_air + 1):
        s1 = float(rng.uniform(50.0, 150.0))
        dur1 = float(rng.uniform(50.0, 80.0))
        rate = 10.0  # 10 s transfer for a 100-unit bundle
        arr1 = s1 + size / rate  # nominal arrival at the relay
        slack = float(rng.uniform(5.0, 25.0))
        s2 = arr1 + slack
        dur2 = float(rng.uniform(50.0, 80.0))
        g.add(StochasticContact(cid, src, k, timing(s1), Dirac(dur1), Dirac(rate),
                                owlt=0.5, layer="air"))
        cid += 1
        g.add(StochasticContact(cid, k, dst, timing(s2), Dirac(dur2), Dirac(rate),
                                owlt=0.5, layer="air"))
        cid += 1

    # --- space route: later, wide windows, timing-certain (A1) ---
    s_up = float(rng.uniform(250.0, 300.0))
    g.add(StochasticContact(cid, src, sat, Dirac(s_up), Dirac(120.0), Dirac(10.0),
                            owlt=3.0, layer="space"))
    cid += 1
    s_dn = s_up + size / 10.0 + float(rng.uniform(20.0, 40.0))
    g.add(StochasticContact(cid, sat, dst, Dirac(s_dn), Dirac(200.0), Dirac(10.0),
                            owlt=3.0, layer="space"))
    cid += 1

    # deadline: the backbone route meets it with margin; nominal air is much earlier
    space_arrival = s_dn + size / 10.0 + 3.0
    deadline = space_arrival + 40.0

    params = {"src": src, "dst": dst, "size": size, "t0": 0.0, "deadline": deadline}
    return g, params
