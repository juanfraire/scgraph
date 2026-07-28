"""UC3 - emergency D2D under partial collapse: existence + correlation (H3).

Topology per instance: source 0, destination 1, two regions of D2D relays.
Region A relays are nominally faster; all relay contacts are opportunistic
(p_exist < 1) and share their region's latent collapse state: when a region
collapses, every contact in it drops to a small existence probability
TOGETHER (eq. 13's regime).

Policies at equal copy budget N:
  stale   - the single nominal-best route from the pre-collapse plan, blind
  naive-N - the N nominally fastest routes (ignores probabilities/correlation)
  scg-N   - greedy joint-success selection on the stochastic contact graph
  epidemic - every candidate route (unbounded overhead; reference bound)
"""

from __future__ import annotations

import numpy as np

from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac
from scgraph.graph import LatentModel, StochasticContactGraph


def make_instance(
    rng: np.random.Generator,
    severity: float,
    m_per_region: int = 3,
    p_base: float = 0.9,
    collapse_scale: float = 0.1,
) -> tuple[StochasticContactGraph, dict]:
    """One UC3 instance; `severity` = per-region collapse probability."""
    src, dst = 0, 1
    n_nodes = 2 + 2 * m_per_region
    g = StochasticContactGraph(num_nodes=n_nodes)
    size = 50.0
    cid = 0

    def relay_pair(node: int, region: str, s1: float):
        nonlocal cid
        rate = 10.0
        dur = float(rng.uniform(80.0, 140.0))
        arr1 = s1 + size / rate
        s2 = arr1 + float(rng.uniform(10.0, 30.0))
        c1 = StochasticContact(cid, src, node, Dirac(s1), Dirac(dur), Dirac(rate),
                               owlt=0.2, layer="ground", p_exist=p_base, group=region)
        cid += 1
        c2 = StochasticContact(cid, node, dst, Dirac(s2), Dirac(dur), Dirac(rate),
                               owlt=0.2, layer="ground", p_exist=p_base, group=region)
        cid += 1
        g.add(c1)
        g.add(c2)

    # region A: nominally faster relays; region B: slower
    for i in range(m_per_region):
        relay_pair(2 + i, "A", s1=float(rng.uniform(20.0, 60.0)))
    for i in range(m_per_region):
        relay_pair(2 + m_per_region + i, "B", s1=float(rng.uniform(80.0, 140.0)))

    # latent state: each region collapses independently with prob = severity
    rho = severity
    g.latent = LatentModel(
        states=[
            ((1 - rho) * (1 - rho), {"A": 1.0, "B": 1.0}),
            (rho * (1 - rho), {"A": collapse_scale, "B": 1.0}),
            ((1 - rho) * rho, {"A": 1.0, "B": collapse_scale}),
            (rho * rho, {"A": collapse_scale, "B": collapse_scale}),
        ]
    )

    deadline = 400.0
    params = {"src": src, "dst": dst, "size": size, "t0": 0.0, "deadline": deadline}
    return g, params
