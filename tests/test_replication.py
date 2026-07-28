"""Eqs. (12)-(13): independence is optimistic under a shared latent state."""

import numpy as np
import pytest

from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac
from scgraph.graph import LatentModel
from scgraph.routing.replication import greedy_select, independent_success, joint_success


def relay_route(cid0, relay_group):
    c1 = StochasticContact.deterministic(cid0, 0, cid0 + 10, 0.0, 500.0, 10.0)
    c1.p_exist, c1.group = 0.9, relay_group
    c2 = StochasticContact.deterministic(cid0 + 1, cid0 + 10, 99, 100.0, 600.0, 10.0)
    c2.p_exist, c2.group = 0.9, relay_group
    return [c1, c2]


def test_correlated_copies_fail_together():
    # latent state: region A collapses with prob 0.5 (contacts drop to 10%)
    latent = LatentModel(states=[(0.5, {"A": 1.0}), (0.5, {"A": 0.1})])
    r1, r2 = relay_route(0, "A"), relay_route(2, "A")
    rng = np.random.default_rng(5)
    size, t0, D, m = 10.0, 0.0, 400.0, 60_000

    p1 = joint_success([r1], size, t0, D, m, rng, latent=latent)
    p_joint = joint_success([r1, r2], size, t0, D, m, rng, latent=latent)
    p_indep = independent_success([p1, p1])

    # eq. (13) < eq. (12) under positive correlation
    assert p_joint < p_indep - 0.02


def test_greedy_diversifies_across_groups():
    latent = LatentModel(
        states=[
            (0.25, {"A": 1.0, "B": 1.0}),
            (0.25, {"A": 0.05, "B": 1.0}),
            (0.25, {"A": 1.0, "B": 0.05}),
            (0.25, {"A": 0.05, "B": 0.05}),
        ]
    )
    cands = [relay_route(0, "A"), relay_route(2, "A"), relay_route(4, "B")]
    rng = np.random.default_rng(9)
    chosen = greedy_select(cands, 2, 10.0, 0.0, 400.0, 30_000, rng, latent=latent)
    groups = {r[0].group for r in chosen}
    assert groups == {"A", "B"}, "greedy should pick one route per region"
