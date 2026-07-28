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

import functools
from pathlib import Path

import numpy as np

from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac, TruncNormal
from scgraph.graph import StochasticContactGraph
from scgraph.routing.deterministic import cgr_route

# CPD-generated orbital skeleton (see scenarios/uc1_skeleton.yaml for the
# scenario and the regeneration command). Checked in so the experiments
# reproduce without a CPD install.
CPD_PLAN = Path(__file__).resolve().parents[2] / "scenarios" / "uc1_cpd"
EVENT_SITE, GATEWAY = 101, 102

# Release times are drawn where the next favourable pass is this far off. Below
# the band the backbone is already fast and a local relay buys nothing; above
# it the deadline becomes so loose that every policy meets it. The band is the
# regime in which the trade H1 is about actually exists.
PASS_BAND = (200.0, 900.0)


def make_instance(
    rng: np.random.Generator,
    sigma: float,
    n_air: int = 3,
    margin: float = 40.0,
) -> tuple[StochasticContactGraph, dict]:
    """One random UC1 instance at timing jitter level sigma (seconds).

    `margin` is the slack between the deadline and the timing-certain backbone's
    arrival. At the default the backbone always meets the deadline, so a router
    that recognises the risk can always escape to it and the achievable
    violation rate is zero. Reducing the margin below zero makes the backbone
    itself infeasible and forces every router to gamble on the jittered air
    relays, which is the regime where the SLA is genuinely contested.
    """
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

    # deadline, placed relative to the backbone's (certain) arrival
    space_arrival = s_dn + size / 10.0 + 3.0
    deadline = space_arrival + margin

    params = {"src": src, "dst": dst, "size": size, "t0": 0.0, "deadline": deadline}
    return g, params


# ---------------------------------------------------------------------------
# CPD-backed variant: the same experiment on a real orbital backbone
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def load_skeleton(path: str = None):
    """The CPD orbital skeleton, loaded once and shared across instances."""
    from scgraph.io import load_cpd_plan

    return load_cpd_plan(path or CPD_PLAN, layer="space")


def make_instance_cpd(
    rng: np.random.Generator,
    sigma: float,
    n_air: int = 3,
    margin: float = 40.0,
    skeleton=None,
) -> tuple[StochasticContactGraph, dict]:
    """A UC1 instance whose space layer comes from orbital mechanics.

    The backbone is no longer invented: it is the store-carry-forward path the
    Contact Plan Designer computes between the event site and the gateway for
    this constellation, with point-mass timing (A1). What each instance varies
    is the *release time* into the plan, so the geometry differs from instance
    to instance while remaining real throughout. On top of that skeleton we
    inject the air relays, which are nominally faster and whose hand-offs
    jitter with standard deviation `sigma` (A2).
    """
    sk = skeleton if skeleton is not None else load_skeleton()
    src, dst = sk.node(EVENT_SITE), sk.node(GATEWAY)
    size = 100.0
    rate = 10.0

    space = StochasticContactGraph(
        num_nodes=sk.num_nodes, contacts=list(sk.contacts), horizon=sk.horizon
    )
    mean_plan = space.mean_plan()

    # find a release time whose next favourable pass sits inside the band
    lo, hi = PASS_BAND
    t0 = 0.0
    space_arrival = None
    for _ in range(200):
        t0 = float(rng.uniform(0.0, sk.horizon - 2.0 * hi))
        route, arr = cgr_route(mean_plan, src, dst, size, t0)
        if route is not None and lo <= (arr - t0) <= hi:
            space_arrival = arr
            break
    if space_arrival is None:  # pragma: no cover - band is wide enough in practice
        raise RuntimeError("no release time with a pass inside PASS_BAND")

    g = StochasticContactGraph(
        num_nodes=sk.num_nodes + n_air, contacts=list(sk.contacts),
        horizon=sk.horizon,
    )
    cid = max(c.cid for c in sk.contacts) + 1

    def timing(nominal: float):
        if sigma <= 0.0:
            return Dirac(nominal)
        return TruncNormal(nominal, sigma, lo=nominal - 3.5 * sigma,
                           hi=nominal + 3.5 * sigma)

    # air relays: nominally early, tight hand-offs, jittered (as in the
    # synthetic variant, but now competing against a real pass schedule)
    for k in range(n_air):
        relay = sk.num_nodes + k
        s1 = t0 + float(rng.uniform(20.0, 80.0))
        dur1 = float(rng.uniform(50.0, 80.0))
        arr1 = s1 + size / rate
        slack = float(rng.uniform(5.0, 25.0))
        s2 = arr1 + slack
        dur2 = float(rng.uniform(50.0, 80.0))
        g.add(StochasticContact(cid, src, relay, timing(s1), Dirac(dur1),
                                Dirac(rate), owlt=0.5, layer="air"))
        cid += 1
        g.add(StochasticContact(cid, relay, dst, timing(s2), Dirac(dur2),
                                Dirac(rate), owlt=0.5, layer="air"))
        cid += 1

    deadline = space_arrival + margin
    params = {
        "src": src, "dst": dst, "size": size, "t0": t0, "deadline": deadline,
        "space_arrival": space_arrival, "pass_wait": space_arrival - t0,
    }
    return g, params
