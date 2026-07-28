"""Replication as a reliability lever (Sec. V-E, eqs. 12-13).

Joint success of a route set is estimated under JOINT sampling (shared
contact draws + shared latent state), which captures eq. (13)'s correlation
penalty; assuming independence instead gives eq. (12)'s optimistic value.
`greedy_select` budgets N copies by marginal joint-success gain, which
naturally diversifies across latent-state groups.
"""

from __future__ import annotations

import numpy as np

from scgraph.arrival import joint_route_set_deliveries
from scgraph.contact import StochasticContact


def joint_success(
    routes: list[list[StochasticContact]],
    size: float,
    t0: float,
    deadline: float,
    m: int,
    rng: np.random.Generator,
    latent=None,
) -> float:
    """P[any copy delivered by D] under joint (correlated) sampling: eq. (13)."""
    if not routes:
        return 0.0
    t = joint_route_set_deliveries(routes, size, t0, m, rng, latent=latent)
    return float(np.mean((t <= deadline).any(axis=0)))


def independent_success(p_singles: list[float]) -> float:
    """Eq. (12): the independence approximation, for comparison only."""
    out = 1.0
    for p in p_singles:
        out *= 1.0 - p
    return 1.0 - out


def greedy_select(
    candidates: list[list[StochasticContact]],
    budget: int,
    size: float,
    t0: float,
    deadline: float,
    m: int,
    rng: np.random.Generator,
    latent=None,
) -> list[list[StochasticContact]]:
    """Pick up to `budget` routes maximizing joint success (correlation-aware)."""
    chosen: list[list[StochasticContact]] = []
    remaining = list(candidates)
    current = 0.0
    for _ in range(min(budget, len(candidates))):
        best_gain, best_route = -1.0, None
        for r in remaining:
            p = joint_success(chosen + [r], size, t0, deadline, m, rng, latent=latent)
            if p - current > best_gain:
                best_gain, best_route = p - current, r
        chosen.append(best_route)
        remaining.remove(best_route)
        current += best_gain
    return chosen
