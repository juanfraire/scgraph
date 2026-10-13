"""Exhaustive enumeration under the support envelope (routing/enumerate.py).

Completeness is checked against brute force on small hand-built graphs: every
node-simple contact sequence is listed without any pruning, then filtered.
Soundness of the envelope is checked against realizations: a route that some
sampled realization delivers by the deadline must be in the enumerated set.
"""

import itertools
import math

import numpy as np
import pytest

from scgraph.arrival import delivery_time, joint_route_set_deliveries
from scgraph.contact import DetContact, StochasticContact
from scgraph.distributions import Dirac, LogNormal, TruncNormal, Uniform
from scgraph.graph import LatentModel, StochasticContactGraph
from scgraph.routing.enumerate import (
    enumerate_routes,
    envelope_arrival,
    screen_routes,
)
from scgraph.routing.quantile import candidate_routes


PM_DEADLINES = (80.0, 150.0, 300.0)
ST_DEADLINE = 200.0


def all_simple_routes(graph, src, dst):
    """Every node-simple contact sequence src -> dst, no time check at all."""
    out = []
    by_frm = {}
    for c in graph.contacts:
        by_frm.setdefault(c.frm, []).append(c)

    def rec(v, path, visited):
        for c in by_frm.get(v, []):
            if c.to in visited:
                continue
            if c.to == dst:
                out.append(tuple(path + [c.cid]))
                continue
            rec(c.to, path + [c.cid], visited | {c.to})

    rec(src, [], {src})
    return out


def random_pointmass_graph(rng, n_nodes=6, n_contacts=30, horizon=150.0):
    g = StochasticContactGraph(num_nodes=n_nodes)
    for cid in range(n_contacts):
        a, b = rng.choice(n_nodes, size=2, replace=False)
        s = float(rng.integers(0, int(horizon)))
        d = float(rng.integers(5, 60))
        r = float(rng.choice([2.0, 5.0, 10.0]))
        g.add(StochasticContact.deterministic(cid, int(a), int(b), s, s + d, r,
                                              owlt=float(rng.choice([0.0, 0.5, 1.0]))))
    return g


def random_stochastic_graph(rng, n_nodes=6, n_contacts=30, horizon=150.0):
    """Mixed laws: jittered starts, random durations, lognormal and uniform
    rates, uncertain existence, a copula-coupled contact and a shared factor."""
    g = StochasticContactGraph(num_nodes=n_nodes)
    for cid in range(n_contacts):
        a, b = rng.choice(n_nodes, size=2, replace=False)
        s = float(rng.uniform(0.0, horizon))
        d = float(rng.uniform(10.0, 60.0))
        kind = cid % 5
        if kind == 0:
            start, dur, rate = Dirac(s), Dirac(d), Dirac(5.0)
        elif kind == 1:
            start = TruncNormal(s, 10.0, s - 30.0, s + 30.0)
            dur, rate = Dirac(d), Dirac(5.0)
        elif kind == 2:
            start, dur = Dirac(s), TruncNormal(d, 8.0, max(0.0, d - 20.0), d + 20.0)
            rate = LogNormal.from_mean_cv(5.0, 0.5)
        elif kind == 3:
            start, dur, rate = Uniform(s - 15.0, s + 15.0), Dirac(d), Uniform(2.0, 8.0)
        else:
            start = TruncNormal(s, 10.0, s - 30.0, s + 30.0)
            dur = TruncNormal(d, 8.0, max(0.0, d - 20.0), d + 20.0)
            rate = LogNormal.from_mean_cv(5.0, 0.3)
        c = StochasticContact(cid, int(a), int(b), start, dur, rate, owlt=0.5,
                              p_exist=0.7 if cid % 3 == 0 else 1.0)
        if kind == 4:
            c = c.with_laws(corr=((1.0, -0.6, -0.3), (-0.6, 1.0, 0.2), (-0.3, 0.2, 1.0)))
        if kind == 2:
            c = c.with_laws(factor="wx", factor_attr="rate", factor_rho=0.5)
        g.add(c)
    return g


def test_fixtures_are_not_vacuous():
    """The random graphs below must actually hold on-time routes, and the
    envelope must be strictly looser than what a sample of draws exhibits."""
    n_pm = sum(
        len(enumerate_routes(random_pointmass_graph(np.random.default_rng(s)),
                             0, 1, 50.0, 0.0, 150.0, hop_slack=99).routes)
        for s in range(12))
    assert n_pm >= 30
    n_env, n_seen = 0, 0
    for s in range(8):
        rng = np.random.default_rng(200 + s)
        g = random_stochastic_graph(rng)
        en = enumerate_routes(g, 0, 1, 40.0, 0.0, ST_DEADLINE, hop_slack=99, rate_q=1.0)
        n_env += len(en.routes)
        if en.routes:
            t = joint_route_set_deliveries(en.contacts(g), 40.0, 0.0, 2000, rng)
            n_seen += int((t <= ST_DEADLINE).any(axis=1).sum())
    assert n_seen >= 50 and n_env > n_seen


@pytest.mark.parametrize("seed", range(12))
def test_pointmass_equals_bruteforce(seed):
    """On a point-mass graph the envelope is the plan itself: the enumeration is
    exactly the set of simple routes that deliver by the deadline."""
    rng = np.random.default_rng(seed)
    g = random_pointmass_graph(rng)
    src, dst, size, t0 = 0, 1, 50.0, 0.0
    det = {c.cid: DetContact(c.cid, c.frm, c.to, c.start.value,
                             c.start.value + c.duration.value, c.rate.value, c.owlt)
           for c in g.contacts}
    every = all_simple_routes(g, src, dst)
    for D in PM_DEADLINES:
        want = {r for r in every if delivery_time([det[i] for i in r], size, t0) <= D}
        en = enumerate_routes(g, src, dst, size, t0, D, hop_slack=99)
        assert en.complete and en.hop_bound_free
        assert set(en.routes) == want
        assert len(en.routes) == len(set(en.routes))
        if want:
            assert en.min_hops == min(len(r) for r in want)
        else:
            assert en.routes == []


@pytest.mark.parametrize("seed", range(10))
def test_envelope_bruteforce_and_pruning(seed):
    """Stochastic laws: the enumeration equals brute force under the envelope,
    with and without the latest-arrival prune, and the hop bound is respected."""
    rng = np.random.default_rng(100 + seed)
    g = random_stochastic_graph(rng)
    src, dst, size, t0, D = 0, 1, 40.0, 0.0, ST_DEADLINE
    every = all_simple_routes(g, src, dst)
    want = {r for r in every if envelope_arrival(r, g, size, t0) <= D}
    full = enumerate_routes(g, src, dst, size, t0, D, hop_slack=99)
    assert set(full.routes) == want
    plain = enumerate_routes(g, src, dst, size, t0, D, hop_slack=99, time_prune=False)
    assert set(plain.routes) == want
    if want:
        for slack in (0, 1, 2):
            en = enumerate_routes(g, src, dst, size, t0, D, hop_slack=slack)
            assert set(en.routes) == {r for r in want if len(r) <= en.hop_bound}
            assert en.hop_bound == min(full.min_hops + slack, en.n_nodes - 1)


@pytest.mark.parametrize("seed", range(8))
def test_envelope_never_excludes_an_ontime_realization(seed):
    """Soundness: any simple route some realization delivers by D is enumerated
    (rate_q = 1 keeps the exact rate supremum)."""
    rng = np.random.default_rng(200 + seed)
    g = random_stochastic_graph(rng)
    src, dst, size, t0, D = 0, 1, 40.0, 0.0, ST_DEADLINE
    en = enumerate_routes(g, src, dst, size, t0, D, hop_slack=99, rate_q=1.0)
    got = set(en.routes)
    every = all_simple_routes(g, src, dst)
    hit = set()
    for _ in range(300):
        real = {c.cid: c for c in g.realize(rng).contacts}
        for r in every:
            if all(i in real for i in r) and delivery_time([real[i] for i in r], size, t0) <= D:
                hit.add(r)
    assert hit <= got
    # and through the vectorised sampler the route MC uses
    if every:
        t = joint_route_set_deliveries([[g.by_id(i) for i in r] for r in every], size, t0,
                                       2000, rng)
        ontime = {r for r, row in zip(every, t) if (row <= D).any()}
        assert ontime <= got


def test_rate_quantile_is_looser_than_nothing_and_tighter_than_supremum():
    rng = np.random.default_rng(7)
    g = random_stochastic_graph(rng, n_contacts=26)
    args = (g, 0, 1, 40.0, 0.0, ST_DEADLINE)
    exact = set(enumerate_routes(*args, hop_slack=99, rate_q=1.0).routes)
    trunc = set(enumerate_routes(*args, hop_slack=99, rate_q=0.999).routes)
    assert trunc <= exact


def test_route_cap_is_reported():
    # a complete digraph on 6 nodes with wide windows: 65 simple routes 0 -> 1
    g = StochasticContactGraph(num_nodes=6)
    for cid, (a, b) in enumerate(itertools.permutations(range(6), 2)):
        g.add(StochasticContact.deterministic(cid, a, b, 0.0, 1000.0, 10.0))
    args = (g, 0, 1, 10.0, 0.0, 400.0)
    full = enumerate_routes(*args, hop_slack=99)
    assert full.complete and full.hop_bound_free and len(full.routes) == 65
    capped = enumerate_routes(*args, hop_slack=99, max_routes=5)
    assert capped.capped and not capped.complete
    assert len(capped.routes) == 5
    assert set(capped.routes) <= set(full.routes)
    bounded = enumerate_routes(*args, hop_slack=1)
    assert bounded.hop_bound == 2 and len(bounded.routes) == 1 + 4


def test_unreachable_and_latent_existence():
    g = StochasticContactGraph(num_nodes=3, latent=LatentModel([(0.5, {"g": 0.0}), (0.5, {"g": 0.0})]))
    g.add(StochasticContact(0, 0, 2, Dirac(0.0), Dirac(100.0), Dirac(10.0), group="g"))
    g.add(StochasticContact(1, 0, 1, Dirac(0.0), Dirac(100.0), Dirac(10.0)))
    en = enumerate_routes(g, 0, 2, 10.0, 0.0, 50.0)
    assert en.routes == [] and en.min_hops is None
    g.add(StochasticContact(2, 1, 2, Dirac(5.0), Dirac(100.0), Dirac(10.0), p_exist=0.1))
    en = enumerate_routes(g, 0, 2, 10.0, 0.0, 50.0)
    assert en.routes == [(1, 2)] and en.min_hops == 2


def test_screen_matches_joint_route_set_deliveries():
    """Common random numbers along the prefix tree reproduce the library's joint
    sampler draw for draw, including copulas, factors and a latent state."""
    rng = np.random.default_rng(42)
    g = random_stochastic_graph(rng, n_contacts=26)
    for i, c in enumerate(g.contacts):
        if i % 4 == 1:
            g.contacts[i] = c.with_laws(group="z")
    g.latent = LatentModel([(0.3, {"z": 0.2}), (0.7, {"z": 1.0})])
    src, dst, size, t0, D = 0, 1, 40.0, 0.0, ST_DEADLINE
    routes = enumerate_routes(g, src, dst, size, t0, D, hop_slack=99).routes
    assert len(routes) >= 3
    routes = list(reversed(routes))       # any order; the scorer sorts internally
    sc = screen_routes(g, routes, size, t0, D, 3000, np.random.default_rng(9), return_ontime=True)
    t = joint_route_set_deliveries([[g.by_id(i) for i in r] for r in routes], size, t0, 3000,
                                   np.random.default_rng(9), latent=g.latent)
    np.testing.assert_array_equal(sc["ontime"], t <= D)
    np.testing.assert_array_equal(sc["k"], (t <= D).sum(axis=1))
    for j in range(len(routes)):
        fin = np.isfinite(t[j])
        want = float(np.mean(t[j][fin])) if fin.any() else math.inf
        assert sc["mean_delivery"][j] == pytest.approx(want)


def test_candidates_that_ever_deliver_are_enumerated():
    """On a UC1 instance, every candidate route that delivers on time in some
    draw is in the enumerated set (the generator never sees a route the
    reference misses)."""
    from scgraph.scenarios.uc1_event import make_instance

    rng = np.random.default_rng(5)
    for sigma in (20.0, 60.0):
        g, p = make_instance(rng, sigma)
        src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
        en = enumerate_routes(g, src, dst, size, t0, D, hop_slack=99)
        assert en.complete and en.hop_bound_free
        cands = candidate_routes(g, src, dst, size, t0, rng, 40)
        t = joint_route_set_deliveries(cands, size, t0, 4000, rng)
        for r, row in zip(cands, t):
            if (row <= D).any():
                assert tuple(c.cid for c in r) in set(en.routes)
