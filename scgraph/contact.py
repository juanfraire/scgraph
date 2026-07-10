"""Contact objects: the stochastic contact (eq. 1) and its realized form.

A StochasticContact carries laws over start, duration (so that T_e = T_s + D
guarantees A5's T_s <= T_e), and rate, plus an existence probability p_exist
(A3's factored form). Timing is continuous; a Dirac in every slot recovers the
classical deterministic contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from scgraph.distributions import Dirac, Distribution


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

    def sample(self, rng: np.random.Generator, p_scale: float = 1.0) -> DetContact | None:
        """Draw one realization; None = the contact never materializes."""
        if rng.random() > self.p_exist * p_scale:
            return None
        t_s = float(self.start.sample(rng))
        dur = max(0.0, float(self.duration.sample(rng)))
        r = max(1e-9, float(self.rate.sample(rng)))
        return DetContact(self.cid, self.frm, self.to, t_s, t_s + dur, r, self.owlt)

    def sample_attrs(self, rng: np.random.Generator, m: int, p_scale=1.0):
        """Vectorized attribute draws for MC route evaluation.

        Returns (exists, t_start, t_end, rate), each an array of shape (m,).
        `p_scale` may be a scalar or an (m,) array (latent-state conditioning).
        """
        exists = rng.random(m) <= self.p_exist * np.asarray(p_scale)
        t_s = np.broadcast_to(np.asarray(self.start.sample(rng, m), dtype=float), (m,)).copy()
        dur = np.maximum(0.0, np.asarray(self.duration.sample(rng, m), dtype=float))
        rate = np.maximum(1e-9, np.asarray(self.rate.sample(rng, m), dtype=float))
        return exists, t_s, t_s + dur, rate

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
