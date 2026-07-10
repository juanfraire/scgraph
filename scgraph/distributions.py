"""Distribution objects for stochastic contact attributes (assumption A2).

Every law exposes sample/cdf/quantile/mean. A Dirac point mass makes any
stochastic object collapse to its deterministic counterpart, which is what
gives the subsumption property (deterministic contact plan = all-Dirac graph)
for free.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats


class Distribution(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, size=None):
        """Draw one value (size=None) or an ndarray of values."""

    @abstractmethod
    def cdf(self, t):
        ...

    @abstractmethod
    def quantile(self, q):
        ...

    @abstractmethod
    def mean(self) -> float:
        ...

    @property
    def is_point_mass(self) -> bool:
        return False


class Dirac(Distribution):
    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, rng, size=None):
        if size is None:
            return self.value
        return np.full(size, self.value)

    def cdf(self, t):
        return np.where(np.asarray(t) >= self.value, 1.0, 0.0)

    def quantile(self, q):
        return self.value

    def mean(self):
        return self.value

    @property
    def is_point_mass(self):
        return True

    def __repr__(self):
        return f"Dirac({self.value})"


class Uniform(Distribution):
    def __init__(self, lo: float, hi: float):
        assert hi > lo
        self.lo, self.hi = float(lo), float(hi)

    def sample(self, rng, size=None):
        return rng.uniform(self.lo, self.hi, size)

    def cdf(self, t):
        return np.clip((np.asarray(t) - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def quantile(self, q):
        return self.lo + q * (self.hi - self.lo)

    def mean(self):
        return 0.5 * (self.lo + self.hi)

    def __repr__(self):
        return f"Uniform({self.lo}, {self.hi})"


class TruncNormal(Distribution):
    """Normal(mu, sigma) truncated to [lo, hi]; the paper's default timing law."""

    def __init__(self, mu: float, sigma: float, lo: float = -math.inf, hi: float = math.inf):
        assert sigma > 0
        self.mu, self.sigma = float(mu), float(sigma)
        self.lo, self.hi = float(lo), float(hi)
        self._a = (self.lo - self.mu) / self.sigma
        self._b = (self.hi - self.mu) / self.sigma
        self._frozen = stats.truncnorm(self._a, self._b, loc=self.mu, scale=self.sigma)

    def sample(self, rng, size=None):
        return self._frozen.rvs(size=size if size is not None else 1, random_state=rng)[
            0
        ] if size is None else self._frozen.rvs(size=size, random_state=rng)

    def cdf(self, t):
        return self._frozen.cdf(t)

    def quantile(self, q):
        return float(self._frozen.ppf(q))

    def mean(self):
        return float(self._frozen.mean())

    def __repr__(self):
        return f"TruncNormal(mu={self.mu}, sigma={self.sigma}, lo={self.lo}, hi={self.hi})"


class LogNormal(Distribution):
    """Lognormal law parameterised by the underlying normal (mu, sigma).

    The paper's default capacity/volume law (A2). mean = exp(mu + sigma^2/2).
    """

    def __init__(self, mu: float, sigma: float):
        assert sigma > 0
        self.mu, self.sigma = float(mu), float(sigma)

    @classmethod
    def from_mean_cv(cls, mean: float, cv: float) -> "LogNormal":
        """Build from an arithmetic mean and coefficient of variation."""
        sigma2 = math.log(1.0 + cv * cv)
        mu = math.log(mean) - 0.5 * sigma2
        return cls(mu, math.sqrt(sigma2))

    def sample(self, rng, size=None):
        return rng.lognormal(self.mu, self.sigma, size)

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t)
        pos = t > 0
        out[pos] = stats.norm.cdf((np.log(t[pos]) - self.mu) / self.sigma)
        return out

    def quantile(self, q):
        return float(math.exp(self.mu + self.sigma * stats.norm.ppf(q)))

    def mean(self):
        return math.exp(self.mu + 0.5 * self.sigma**2)

    def __repr__(self):
        return f"LogNormal(mu={self.mu}, sigma={self.sigma})"
