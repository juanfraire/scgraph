"""Analytic delivery-time CDF propagation (Sec. V-F strategy a).

Grid-based numerical propagation of the arrival CDF along a route, under the
v1 restrictions that make the per-hop law closed-form-composable:
independent contacts (A4 baseline), Dirac rate and Dirac duration per contact
(timing uncertainty in the start; the window shifts rigidly with it).

Per hop, with A the (defective) arrival CDF at the transmitting node,
S the start law, d the duration, w = size/rate, ell the OWLT, p the
existence probability:

  feasible given S=s  iff  A <= s + d - w        (and w <= d)
  arrival             =    max(A, s) + w + ell
  F_next(t) = p * E_S[ 1{S <= t-w-ell} * F_A( min(t-w-ell, S+d-w) ) ]

which is evaluated by quadrature over the start-law grid. Used to validate
the Monte-Carlo evaluator (M1's consistency test) and as the fast cost inside
an analytic route search.
"""

from __future__ import annotations

import math

import numpy as np

from scgraph.contact import StochasticContact


def route_delivery_cdf(
    route: list[StochasticContact],
    size: float,
    t0: float,
    t_grid: np.ndarray,
    s_points: int = 2001,
) -> np.ndarray:
    """Defective CDF of T_pi on t_grid (its limit is P[route ever delivers])."""
    F = (t_grid >= t0).astype(float)  # arrival at source: Dirac(t0)

    for c in route:
        if not (c.duration.is_point_mass and c.rate.is_point_mass):
            raise ValueError(
                "analytic propagation (v1) needs Dirac duration and rate; "
                f"contact {c.cid} violates this"
            )
        d = c.duration.mean()
        w = size / c.rate.mean()
        ell = c.owlt
        p = c.p_exist

        if w > d:
            return np.zeros_like(t_grid)  # transfer can never fit the window

        def F_A(x, F=F):
            return np.interp(x, t_grid, F, left=0.0, right=F[-1])

        if c.start.is_point_mass:
            s0 = c.start.mean()
            shifted = t_grid - w - ell
            F = p * np.where(shifted >= s0, F_A(np.minimum(shifted, s0 + d - w)), 0.0)
        else:
            # quadrature over the start law
            qs = np.linspace(1e-6, 1 - 1e-6, s_points)
            s_vals = np.array([c.start.quantile(q) for q in qs]) \
                if not hasattr(c.start, "_frozen") else c.start._frozen.ppf(qs)
            weight = 1.0 / s_points  # equal probability mass per quadrature point
            shifted = t_grid[:, None] - w - ell           # (T, 1)
            cap = s_vals[None, :] + d - w                  # (1, S)
            arg = np.minimum(shifted, cap)
            mask = s_vals[None, :] <= shifted
            F = p * (F_A(arg) * mask).sum(axis=1) * weight

    return F


def route_quantile(
    route: list[StochasticContact],
    size: float,
    t0: float,
    one_minus_eps: float,
    t_max: float,
    n_grid: int = 4000,
) -> float:
    """Q_{1-eps}(pi) (eq. 8) from the analytic CDF; inf if the mass never reaches it."""
    t_grid = np.linspace(t0, t_max, n_grid)
    F = route_delivery_cdf(route, size, t0, t_grid)
    idx = np.searchsorted(F, one_minus_eps)
    if idx >= len(t_grid):
        return math.inf
    return float(t_grid[idx])


def route_p_success(
    route: list[StochasticContact],
    size: float,
    t0: float,
    deadline: float,
    n_grid: int = 4000,
) -> float:
    """P[T_pi <= D] (eq. 5) from the analytic CDF."""
    t_grid = np.linspace(t0, deadline, n_grid)
    F = route_delivery_cdf(route, size, t0, t_grid)
    return float(F[-1])
