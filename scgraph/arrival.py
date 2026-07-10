"""The store-carry-forward arrival recursion, eqs. (2)-(4) of the paper.

Scalar form for realized routes, and a vectorized form that evaluates a route
of stochastic contacts under M joint attribute draws (the per-route Monte
Carlo of Sec. V-F, strategy b).
"""

from __future__ import annotations

import math

import numpy as np

from scgraph.contact import DetContact, StochasticContact


def delivery_time(route: list[DetContact], size: float, t0: float) -> float:
    """T_pi for one realized route; inf on infeasibility (eq. 4)."""
    a = t0
    for c in route:
        tau = max(a, c.t_start)          # eq. (2)
        e = tau + size / c.rate          # eq. (3), constant-rate form
        if e > c.t_end:                  # feasibility: transfer fits the window
            return math.inf
        a = e + c.owlt                   # eq. (4)
    return a


def sample_route_deliveries(
    route: list[StochasticContact],
    size: float,
    t0: float,
    m: int,
    rng: np.random.Generator,
    latent=None,
) -> np.ndarray:
    """(m,) array of delivery times for a stochastic route; inf = failure.

    Contacts are sampled independently given the latent state (A3/A4); the
    latent state, when present, is shared across the route's contacts.
    """
    scale_by_group: dict = {}
    if latent is not None:
        groups = {c.group for c in route if c.group is not None}
        for g in groups:
            scale_by_group[g] = None  # filled from one shared draw below
        # one latent draw per MC sample, shared across all contacts
        probs = [p for p, _ in latent.states]
        idx = rng.choice(len(latent.states), size=m, p=probs)
        for g in list(scale_by_group):
            scale_by_group[g] = np.array([latent.states[i][1].get(g, 1.0) for i in idx])

    a = np.full(m, t0, dtype=float)
    for c in route:
        p_scale = scale_by_group.get(c.group, 1.0) if c.group is not None else 1.0
        exists, t_s, t_e, rate = c.sample_attrs(rng, m, p_scale=p_scale)
        tau = np.maximum(a, t_s)
        e = tau + size / rate
        feasible = exists & (e <= t_e) & np.isfinite(a)
        a = np.where(feasible, e + c.owlt, np.inf)
    return a


def joint_route_set_deliveries(
    routes: list[list[StochasticContact]],
    size: float,
    t0: float,
    m: int,
    rng: np.random.Generator,
    latent=None,
) -> np.ndarray:
    """(n_routes, m) delivery times under JOINT sampling.

    Contacts shared by several routes get one draw per MC sample, and the
    latent state is shared by everything: this is what makes replication
    correlation-aware (eq. 11) instead of optimistic (eq. 10).
    """
    # one latent draw per sample
    if latent is not None:
        probs = [p for p, _ in latent.states]
        idx = rng.choice(len(latent.states), size=m, p=probs)
    else:
        idx = None

    # one attribute draw per distinct contact
    draws: dict[int, tuple] = {}
    for route in routes:
        for c in route:
            if c.cid in draws:
                continue
            if c.group is not None and idx is not None:
                p_scale = np.array(
                    [latent.states[i][1].get(c.group, 1.0) for i in idx]
                )
            else:
                p_scale = 1.0
            draws[c.cid] = c.sample_attrs(rng, m, p_scale=p_scale)

    out = np.empty((len(routes), m), dtype=float)
    for k, route in enumerate(routes):
        a = np.full(m, t0, dtype=float)
        for c in route:
            exists, t_s, t_e, rate = draws[c.cid]
            tau = np.maximum(a, t_s)
            e = tau + size / rate
            feasible = exists & (e <= t_e) & np.isfinite(a)
            a = np.where(feasible, e + c.owlt, np.inf)
        out[k] = a
    return out
