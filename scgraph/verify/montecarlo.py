"""Monte-Carlo SLA verification (Sec. V-F strategy b): P_hat with confidence.

Full-graph verification realizes the whole stochastic contact graph and lets
the deterministic router adapt per realization; this is the "oracle rerouting"
upper reference, while fixed-route evaluation measures a committed policy.
"""

from __future__ import annotations

import math

import numpy as np

from scgraph.graph import StochasticContactGraph
from scgraph.routing.deterministic import cgr_route


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def verify_fullgraph(
    graph: StochasticContactGraph,
    src: int,
    dst: int,
    size: float,
    t0: float,
    deadline: float,
    m: int,
    rng: np.random.Generator,
) -> dict:
    """P[deliver by D] when a deterministic router reroutes on each realization.

    This is the omniscient-per-realization reference (the router sees the
    realized plan), an upper bound no committed policy can beat.
    """
    ok = 0
    for _ in range(m):
        real = graph.realize(rng)
        _, arrival = cgr_route(real, src, dst, size, t0)
        if arrival <= deadline:
            ok += 1
    lo, hi = wilson_ci(ok, m)
    return {"p_success": ok / m, "ci": (lo, hi), "n": m}
