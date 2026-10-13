"""Monte-Carlo SLA verification (Sec. V-F strategy b): P_hat with confidence.

Full-graph verification realizes the whole stochastic contact graph and lets
the deterministic router adapt per realization; this is the "oracle rerouting"
upper reference, while fixed-route evaluation measures a committed policy.
"""

from __future__ import annotations

import math

import numpy as np

from scgraph.arrival import joint_route_set_deliveries, sample_route_deliveries
from scgraph.graph import StochasticContactGraph
from scgraph.routing.deterministic import cgr_route
from scgraph.stats import clopper_pearson, cp_upper


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


def evaluate_route(
    route,
    size: float,
    t0: float,
    deadline: float,
    m: int,
    rng: np.random.Generator,
    latent=None,
    delta: float = 0.05,
) -> dict:
    """Held-out evaluation of a committed route (revision).

    `rng` must be a stream the router never saw. A route scored on the draws
    that selected it inherits the selection's optimism; scored here, it is
    measured like any baseline. Returns the violation estimate, the failure
    count, the exact two-sided 95% interval on the violation probability, and
    its one-sided (1 - delta) upper bound.
    """
    if not route:
        return {"viol": 1.0, "fails": m, "n": m, "ci": (1.0, 1.0), "viol_ub": 1.0,
                "mean_delivery": math.inf, "mean_ontime": math.inf,
                "samples": np.full(m, np.inf)}
    t = sample_route_deliveries(route, size, t0, m, rng, latent=latent)
    fails = int(np.count_nonzero(~(t <= deadline)))
    fin = t[np.isfinite(t)]                 # delivered at all (the paper's latency)
    ontime = t[t <= deadline]
    return {
        "viol": fails / m,
        "fails": fails,
        "n": int(m),
        "ci": clopper_pearson(fails, m),
        "viol_ub": cp_upper(fails, m, delta),
        "mean_delivery": float(np.mean(fin)) if fin.size else math.inf,
        "mean_ontime": float(np.mean(ontime)) if ontime.size else math.inf,
        "samples": t,
    }


def evaluate_route_set(routes, size, t0, deadline, m, rng, latent=None) -> dict:
    """Held-out joint success of a replicated route set (any copy by D)."""
    if not routes:
        return {"p_success": 0.0, "k": 0, "n": int(m)}
    t = joint_route_set_deliveries(routes, size, t0, m, rng, latent=latent)
    k = int(np.count_nonzero((t <= deadline).any(axis=0)))
    return {"p_success": k / m, "k": k, "n": int(m)}
