import numpy as np
import pytest

from scgraph.distributions import Dirac, LogNormal, TruncNormal, Uniform


RNG = np.random.default_rng(7)


def test_dirac():
    d = Dirac(5.0)
    assert d.sample(RNG) == 5.0
    assert d.quantile(0.01) == 5.0 and d.quantile(0.99) == 5.0
    assert d.cdf(4.999) == 0.0 and d.cdf(5.0) == 1.0
    assert d.mean() == 5.0
    assert d.is_point_mass


@pytest.mark.parametrize(
    "dist",
    [
        Uniform(2.0, 10.0),
        TruncNormal(mu=100.0, sigma=15.0, lo=60.0, hi=140.0),
        LogNormal.from_mean_cv(mean=50.0, cv=0.4),
    ],
)
def test_quantile_cdf_roundtrip(dist):
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert dist.cdf(dist.quantile(q)) == pytest.approx(q, abs=1e-6)


@pytest.mark.parametrize(
    "dist",
    [
        Uniform(2.0, 10.0),
        TruncNormal(mu=100.0, sigma=15.0, lo=60.0, hi=140.0),
        LogNormal.from_mean_cv(mean=50.0, cv=0.4),
    ],
)
def test_sample_mean_matches(dist):
    x = dist.sample(np.random.default_rng(0), size=200_000)
    assert np.mean(x) == pytest.approx(dist.mean(), rel=0.01)


def test_lognormal_from_mean_cv():
    ln = LogNormal.from_mean_cv(mean=80.0, cv=0.5)
    assert ln.mean() == pytest.approx(80.0, rel=1e-9)
