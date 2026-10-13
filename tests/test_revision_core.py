"""Revision core: exact bounds, joint laws, common factors, held-out evaluation."""

import math

import numpy as np
import pytest
from scipy import stats as st

from scgraph.arrival import sample_route_deliveries
from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac, LogNormal, TruncNormal, TruncStudentT, Uniform
from scgraph.graph import StochasticContactGraph
from scgraph.routing.quantile import pick_route_chance, route_stats
from scgraph.stats import (
    admission_draws, clopper_pearson, cp_lower, cp_upper, zero_failure_n,
)
from scgraph.verify.montecarlo import evaluate_route


# ----------------------------------------------------------------- bounds

def test_rule_of_three():
    # 0 failures in n draws -> upper bound ~ 3/n at 95%
    n = 2000
    ub = cp_upper(0, n, 0.05)
    assert ub == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-9)
    assert ub == pytest.approx(3.0 / n, rel=0.01)


def test_zero_failure_n_matches_bound():
    for eps in (1e-1, 1e-2, 1e-3):
        n = zero_failure_n(eps, 0.05)
        assert cp_upper(0, n, 0.05) <= eps
        assert cp_upper(0, n - 1, 0.05) > eps


def test_cp_lower_upper_bracket_estimate():
    k, n = 37, 400
    lo, hi = cp_lower(k, n, 0.05), cp_upper(k, n, 0.05)
    assert lo < k / n < hi
    a, b = clopper_pearson(k, n, 0.95)
    assert a < lo and hi < b       # two-sided 95% is wider than one-sided 95%


def test_admission_draws_monotone():
    assert admission_draws(1e-3, 10) > admission_draws(1e-2, 10) >= 2000


# ------------------------------------------------------- distributions

@pytest.mark.parametrize("law", [
    TruncNormal(10.0, 3.0, 0.0, 20.0), LogNormal.from_mean_cv(5.0, 0.4),
    Uniform(1.0, 4.0), TruncStudentT.matched_std(0.0, 2.0, 4.0, -10.0, 10.0),
])
def test_ppf_inverts_cdf(law):
    u = np.array([0.05, 0.3, 0.5, 0.9])
    x = law.ppf(u)
    assert np.allclose(law.cdf(x), u, atol=1e-6)


def test_student_matched_std_has_heavier_tails():
    sd, nu = 10.0, 3.0
    t = TruncStudentT.matched_std(0.0, sd, nu)
    rng = np.random.default_rng(0)
    x = t.sample(rng, 400_000)
    assert np.std(x) == pytest.approx(sd, rel=0.1)
    # P[|X| > 3 sd] is far larger than the Gaussian's 0.0027
    assert np.mean(np.abs(x) > 3 * sd) > 0.005


# ------------------------------------------------------------ joint laws

def _coupled(rho_sd=-0.8, rho_sr=-0.6, rho_dr=0.4):
    corr = ((1.0, rho_sd, rho_sr), (rho_sd, 1.0, rho_dr), (rho_sr, rho_dr, 1.0))
    return StochasticContact(
        0, 0, 1, TruncNormal(100.0, 20.0, 30.0, 170.0),
        TruncNormal(60.0, 10.0, 0.0, 120.0), LogNormal.from_mean_cv(10.0, 0.3),
        corr=corr,
    )


def test_copula_preserves_marginals_and_couples():
    c = _coupled()
    rng = np.random.default_rng(1)
    _, ts, te, rate = c.sample_attrs(rng, 100_000)
    dur = te - ts
    # marginals unchanged (KS against the declared laws)
    assert st.kstest(ts, c.start.cdf).pvalue > 1e-3
    assert st.kstest(dur, c.duration.cdf).pvalue > 1e-3
    assert st.kstest(rate, c.rate.cdf).pvalue > 1e-3
    # and they move together with the declared signs
    assert np.corrcoef(ts, dur)[0, 1] < -0.6
    assert np.corrcoef(ts, rate)[0, 1] < -0.4
    # feasibility by construction
    assert np.all(te >= ts)


def test_independent_twin_has_same_marginals_no_coupling():
    c = _coupled().independent_twin()
    assert c.corr is None
    rng = np.random.default_rng(2)
    _, ts, te, _ = c.sample_attrs(rng, 50_000)
    assert abs(np.corrcoef(ts, te - ts)[0, 1]) < 0.03


def test_common_factor_correlates_contacts_at_rho():
    rho = 0.4
    law = LogNormal.from_mean_cv(10.0, 0.5)
    a = StochasticContact(0, 0, 1, Dirac(0.0), Dirac(100.0), law,
                          factor="wx", factor_attr="rate", factor_rho=rho)
    b = StochasticContact(1, 1, 2, Dirac(200.0), Dirac(100.0), law,
                          factor="wx", factor_attr="rate", factor_rho=rho)
    rng = np.random.default_rng(3)
    zf = {"wx": rng.standard_normal(200_000)}
    _, _, _, ra = a.sample_attrs(rng, 200_000, zf=zf)
    _, _, _, rb = b.sample_attrs(rng, 200_000, zf=zf)
    za, zb = np.log(ra), np.log(rb)            # normal scores up to affine maps
    assert np.corrcoef(za, zb)[0, 1] == pytest.approx(rho, abs=0.01)


def test_factor_shared_inside_route_evaluation():
    """A bundle needing BOTH passes to be good fails more when fades co-move."""
    law = LogNormal.from_mean_cv(10.0, 0.6)

    def route(rho):
        return [
            StochasticContact(0, 0, 1, Dirac(0.0), Dirac(12.0), law,
                              factor="wx", factor_rho=rho),
            StochasticContact(1, 1, 2, Dirac(20.0), Dirac(12.0), law,
                              factor="wx", factor_rho=rho),
        ]

    def p(rho):
        t = sample_route_deliveries(route(rho), 100.0, 0.0, 200_000,
                                    np.random.default_rng(4))
        return np.mean(np.isfinite(t))

    # P[both fit] = P[A]P[B] under independence, larger when they co-move
    assert p(0.8) > p(0.0) + 0.02


def test_realize_uses_one_factor_draw_per_realization():
    law = LogNormal.from_mean_cv(10.0, 0.5)
    g = StochasticContactGraph(3, [
        StochasticContact(0, 0, 1, Dirac(0.0), Dirac(50.0), law,
                          factor="wx", factor_rho=0.999),
        StochasticContact(1, 1, 2, Dirac(60.0), Dirac(50.0), law,
                          factor="wx", factor_rho=0.999),
    ])
    rng = np.random.default_rng(5)
    r = [g.realize(rng) for _ in range(2000)]
    ra = np.array([x.contacts[0].rate for x in r])
    rb = np.array([x.contacts[1].rate for x in r])
    assert np.corrcoef(np.log(ra), np.log(rb))[0, 1] > 0.95


def test_plain_contacts_unchanged_by_zf():
    """A contact without coupling ignores zf and draws exactly as before."""
    c = StochasticContact(0, 0, 1, TruncNormal(5.0, 1.0, 0.0, 10.0),
                          Dirac(5.0), Dirac(1.0))
    a = c.sample_attrs(np.random.default_rng(9), 100)
    b = c.sample_attrs(np.random.default_rng(9), 100, zf={"x": np.zeros(100)})
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# ------------------------------------------------------- admission / holdout

def _two_route_graph():
    """A safe slow route and a fast route that fails ~3% of the time."""
    g = StochasticContactGraph(3)
    g.add(StochasticContact(0, 0, 2, Dirac(500.0), Dirac(100.0), Dirac(10.0)))
    # fast: opens late with prob ~3% -> misses deadline
    g.add(StochasticContact(1, 0, 2, TruncNormal(10.0, 5.0, 0.0, 400.0),
                            Dirac(50.0), Dirac(10.0), p_exist=0.97))
    return g


def test_lcb_admission_is_more_conservative_than_point():
    g = _two_route_graph()
    D, eps = 600.0, 0.05
    picks = {"point": 0, "lcb": 0}
    for seed in range(30):
        for rule in picks:
            r, s = pick_route_chance(g, 0, 2, 10.0, 0.0, D, np.random.default_rng(seed),
                                     m=400, eps=eps, admission=rule, n_realizations=5)
            picks[rule] += int(r[0].cid == 1)
    # the point rule admits the risky fast route at least as often
    assert picks["point"] >= picks["lcb"]


def test_evaluate_route_reports_bounds():
    g = _two_route_graph()
    out = evaluate_route([g.by_id(1)], 10.0, 0.0, 600.0, 20_000, np.random.default_rng(0))
    assert out["viol"] == pytest.approx(0.03, abs=0.006)
    assert out["ci"][0] < out["viol"] < out["ci"][1]
    assert out["viol_ub"] > out["viol"]


def test_window_keeps_only_reachable_contacts():
    g = StochasticContactGraph(2, [
        StochasticContact(0, 0, 1, Dirac(0.0), Dirac(10.0), Dirac(1.0)),
        StochasticContact(1, 0, 1, Dirac(100.0), Dirac(10.0), Dirac(1.0)),
        StochasticContact(2, 0, 1, TruncNormal(200.0, 10.0, 150.0, 250.0),
                          Dirac(10.0), Dirac(1.0)),
    ])
    w = g.window(95.0, 160.0)
    assert sorted(c.cid for c in w.contacts) == [1, 2]
