"""Contact objects: the stochastic contact (eq. 1) and its realized form.

A StochasticContact carries laws over start, duration (so that T_e = T_s + D
guarantees A5's T_s <= T_e), and rate, plus an existence probability p_exist
(A3's factored form). Timing is continuous; a Dirac in every slot recovers the
classical deterministic contact.

Dependence (revision). Two optional mechanisms make the law genuinely joint:

  corr        a 3x3 Gaussian-copula correlation over the normal scores of
              (start, duration, rate). A negative start-duration entry makes a
              late opening shorten the window (the end stays roughly put); a
              negative start-rate entry makes a late relay also a faded one.
              The marginals are untouched, so the coupled contact and its
              independent twin have identical per-attribute laws and differ
              only in how the attributes move together.
  factor      a common-factor name shared across contacts, with loading
              `factor_rho` on one attribute (`factor_attr`). Two contacts that
              share the factor have that attribute's normal scores correlated
              at `factor_rho`: a weather front fading every feeder pass at
              once, or a wind regime delaying every drone together.

Both act through the normal scores, so feasibility is preserved by
construction: the duration is still a law on [0, inf), the end is still
start + duration, and the deliverable volume is still rate times the usable
overlap. A contact without either mechanism samples exactly as before, draw
for draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from scgraph.distributions import Dirac, Distribution

ATTR_INDEX = {"start": 0, "duration": 1, "rate": 2}
_U_EPS = 1e-12


@dataclass(frozen=True)
class DetContact:
    """A realized (deterministic) contact: what CGR/SABR consumes.

    `cid` links the realization back to the stochastic contact it was drawn
    from, so routes found on realizations can be re-evaluated under the laws.
    """

    cid: int
    frm: int
    to: int
    t_start: float
    t_end: float
    rate: float
    owlt: float = 0.0


@dataclass
class StochasticContact:
    """Eq. (1): existence x (start, duration, rate) laws + one-way light time."""

    cid: int
    frm: int
    to: int
    start: Distribution
    duration: Distribution
    rate: Distribution
    owlt: float = 0.0
    p_exist: float = 1.0
    layer: str = "space"  # space | air | ground
    group: str | None = None  # latent-state group (A4 correlation, UC3)
    # revision: joint law within the contact, and dependence across contacts
    corr: tuple | None = None       # 3x3 copula correlation over (start, duration, rate)
    factor: str | None = None       # common-factor name shared across contacts
    factor_attr: str = "rate"       # attribute the factor drives
    factor_rho: float = 0.0         # correlation between two contacts sharing it
    kind: str | None = None         # free-form link class ("feeder", "isl", "uav", ...)

    @property
    def uses_scores(self) -> bool:
        """True when the joint (copula / common-factor) sampler is needed."""
        return self.corr is not None or (self.factor is not None and self.factor_rho > 0.0)

    def _scored_attrs(self, rng: np.random.Generator, m: int, zf=None):
        """(t_s, dur, rate) through correlated normal scores and the marginal ppfs."""
        s = rng.standard_normal((m, 3))
        if self.corr is not None:
            chol = np.linalg.cholesky(np.asarray(self.corr, dtype=float))
            s = s @ chol.T
        if self.factor is not None and self.factor_rho > 0.0:
            z = None if zf is None else zf.get(self.factor)
            if z is None:
                z = rng.standard_normal(m)
            j = ATTR_INDEX[self.factor_attr]
            rho = float(self.factor_rho)
            s[:, j] = math.sqrt(rho) * np.asarray(z) + math.sqrt(1.0 - rho) * s[:, j]
        u = np.clip(stats.norm.cdf(s), _U_EPS, 1.0 - _U_EPS)
        t_s = np.broadcast_to(np.asarray(self.start.ppf(u[:, 0]), dtype=float), (m,)).copy()
        dur = np.maximum(0.0, np.asarray(self.duration.ppf(u[:, 1]), dtype=float))
        rate = np.maximum(1e-9, np.asarray(self.rate.ppf(u[:, 2]), dtype=float))
        return t_s, dur, rate

    def sample(self, rng: np.random.Generator, p_scale: float = 1.0, zf=None) -> DetContact | None:
        """Draw one realization; None = the contact never materializes.

        `zf` maps common-factor names to the factor's value for this
        realization (shape (1,)), shared by every contact that loads on it.
        """
        if rng.random() > self.p_exist * p_scale:
            return None
        if self.uses_scores:
            t_s, dur, r = (float(x[0]) for x in self._scored_attrs(rng, 1, zf))
            return DetContact(self.cid, self.frm, self.to, t_s, t_s + dur, r, self.owlt)
        t_s = float(self.start.sample(rng))
        dur = max(0.0, float(self.duration.sample(rng)))
        r = max(1e-9, float(self.rate.sample(rng)))
        return DetContact(self.cid, self.frm, self.to, t_s, t_s + dur, r, self.owlt)

    def sample_attrs(self, rng: np.random.Generator, m: int, p_scale=1.0, zf=None):
        """Vectorized attribute draws for MC route evaluation.

        Returns (exists, t_start, t_end, rate), each an array of shape (m,).
        `p_scale` may be a scalar or an (m,) array (latent-state conditioning).
        `zf` maps common-factor names to (m,) arrays shared across contacts.
        """
        exists = rng.random(m) <= self.p_exist * np.asarray(p_scale)
        if self.uses_scores:
            t_s, dur, rate = self._scored_attrs(rng, m, zf)
            return exists, t_s, t_s + dur, rate
        t_s = np.broadcast_to(np.asarray(self.start.sample(rng, m), dtype=float), (m,)).copy()
        dur = np.maximum(0.0, np.asarray(self.duration.sample(rng, m), dtype=float))
        rate = np.maximum(1e-9, np.asarray(self.rate.sample(rng, m), dtype=float))
        return exists, t_s, t_s + dur, rate

    def support_window(self) -> tuple[float, float]:
        """Earliest possible start and latest possible end over the law's support."""
        s_lo, s_hi = self.start.support()
        _, d_hi = self.duration.support()
        return s_lo, s_hi + d_hi

    def with_laws(self, **changes) -> "StochasticContact":
        """A copy with some fields replaced (laws, existence, coupling...)."""
        from dataclasses import replace

        return replace(self, **changes)

    def independent_twin(self) -> "StochasticContact":
        """Same marginals, within-contact coupling removed.

        This is the 'three separate marginals' model the paper argues against:
        each attribute keeps its own law, but they no longer move together.
        """
        return self.with_laws(corr=None)

    def mean_det(self) -> DetContact:
        """Point-mass (nominal) version at the law means: the 'mean plan' atom."""
        t_s = self.start.mean()
        return DetContact(
            self.cid, self.frm, self.to, t_s, t_s + self.duration.mean(),
            self.rate.mean(), self.owlt,
        )

    @property
    def is_point_mass(self) -> bool:
        return (
            self.p_exist >= 1.0
            and self.start.is_point_mass
            and self.duration.is_point_mass
            and self.rate.is_point_mass
        )

    @classmethod
    def deterministic(
        cls, cid, frm, to, t_start, t_end, rate, owlt=0.0, layer="space"
    ) -> "StochasticContact":
        """The subsumption constructor: a classical contact as all-Dirac laws."""
        return cls(
            cid=cid, frm=frm, to=to,
            start=Dirac(t_start), duration=Dirac(t_end - t_start), rate=Dirac(rate),
            owlt=owlt, p_exist=1.0, layer=layer,
        )
