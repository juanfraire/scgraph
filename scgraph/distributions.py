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

    def ppf(self, u):
        """Vectorised inverse CDF; the copula path samples through it."""
        return np.array([self.quantile(float(x)) for x in np.ravel(u)]).reshape(np.shape(u))

    def support(self) -> tuple[float, float]:
        """Closed interval holding all the mass (may be infinite)."""
        return -math.inf, math.inf


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

    def ppf(self, u):
        return np.full(np.shape(u), self.value)

    def support(self):
        return self.value, self.value

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

    def ppf(self, u):
        return self.lo + np.asarray(u, dtype=float) * (self.hi - self.lo)

    def support(self):
        return self.lo, self.hi

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

    def ppf(self, u):
        return self._frozen.ppf(u)

    def support(self):
        return self.lo, self.hi

    def std(self) -> float:
        return float(self._frozen.std())

    def __repr__(self):
        return f"TruncNormal(mu={self.mu}, sigma={self.sigma}, lo={self.lo}, hi={self.hi})"


class TruncStudentT(Distribution):
    """Location-scale Student-t truncated to [lo, hi]: the heavy-tailed truth.

    Used by the misspecification study (revision, R2/R3): the planner believes
    a TruncNormal of standard deviation sigma while the world draws from a
    t law of the same (untruncated) standard deviation but heavier tails.
    `matched_std` sets the scale so that the comparison changes the tail and
    nothing else.
    """

    def __init__(self, mu: float, scale: float, nu: float,
                 lo: float = -math.inf, hi: float = math.inf):
        assert scale > 0 and nu > 0
        self.mu, self.scale, self.nu = float(mu), float(scale), float(nu)
        self.lo, self.hi = float(lo), float(hi)
        self._t = stats.t(self.nu, loc=self.mu, scale=self.scale)
        self._flo = float(self._t.cdf(self.lo))
        self._fhi = float(self._t.cdf(self.hi))

    @classmethod
    def matched_std(cls, mu: float, std: float, nu: float,
                    lo: float = -math.inf, hi: float = math.inf) -> "TruncStudentT":
        assert nu > 2, "the t law has a finite variance only for nu > 2"
        return cls(mu, std * math.sqrt((nu - 2.0) / nu), nu, lo, hi)

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        return self._t.ppf(self._flo + u * (self._fhi - self._flo))

    def sample(self, rng, size=None):
        u = rng.random(size if size is not None else 1)
        out = self.ppf(u)
        return float(out[0]) if size is None else out

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        c = (self._t.cdf(t) - self._flo) / (self._fhi - self._flo)
        return np.clip(c, 0.0, 1.0)

    def quantile(self, q):
        return float(self.ppf(np.asarray([q]))[0])

    def mean(self):
        # numerical mean of the truncated law
        u = (np.arange(4000) + 0.5) / 4000
        return float(np.mean(self.ppf(u)))

    def support(self):
        return self.lo, self.hi

    def __repr__(self):
        return (f"TruncStudentT(mu={self.mu}, scale={self.scale}, nu={self.nu}, "
                f"lo={self.lo}, hi={self.hi})")


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

    def ppf(self, u):
        return np.exp(self.mu + self.sigma * stats.norm.ppf(u))

    def support(self):
        return 0.0, math.inf

    def __repr__(self):
        return f"LogNormal(mu={self.mu}, sigma={self.sigma})"
