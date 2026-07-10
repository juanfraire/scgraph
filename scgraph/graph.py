"""The stochastic contact graph, its realizations, and the latent state (A4).

A realization drawn from the graph is a plain deterministic contact plan, so
any classical router (A-SABR) can run on it unchanged: that is the sampled-
realization property the Monte-Carlo verification loop relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scgraph.contact import DetContact, StochasticContact


@dataclass
class LatentModel:
    """Shared latent state z inducing correlation across contact groups (A4 ext / UC3).

    `states` is a list of (probability, {group: p_exist_scale}) pairs.
    Contacts whose `group` appears in the mapping get their existence
    probability multiplied by the scale when that state is drawn.
    """

    states: list[tuple[float, dict[str, float]]]

    def __post_init__(self):
        total = sum(p for p, _ in self.states)
        assert abs(total - 1.0) < 1e-9, f"state probabilities sum to {total}"

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        probs = [p for p, _ in self.states]
        idx = rng.choice(len(self.states), p=probs)
        return self.states[idx][1]

    def sample_scales(self, rng: np.random.Generator, m: int, group: str | None):
        """(m,) array of p_exist scales for a given group under m latent draws."""
        probs = [p for p, _ in self.states]
        idx = rng.choice(len(self.states), size=m, p=probs)
        scales = np.array([self.states[i][1].get(group, 1.0) for i in idx])
        return scales


@dataclass
class Realization:
    """A deterministic contact plan drawn from the graph (or its mean)."""

    num_nodes: int
    contacts: list[DetContact]

    def write_asabr(self, path: str) -> None:
        """A-SABR dynamic-format plan (evl manager: volume-limited, no queue delay).

        A-SABR's plan lexer parses numbers as i64 (types.rs AnyNumber), so all
        attribute values must be integral. Realizations meant for the A-SABR
        cross-check must therefore be generated on an integer lattice; this
        writer refuses to round silently. (Real-valued plan support is a
        natural item for the future A-SABR feature/stochastic branch.)
        """
        def as_int(v: float, what: str) -> int:
            if abs(v - round(v)) > 1e-9:
                raise ValueError(
                    f"A-SABR plans take integers only; {what}={v} is not integral"
                )
            return int(round(v))

        with open(path, "w") as f:
            for n in range(self.num_nodes):
                f.write(f"node {n} n{n}\n")
            for c in self.contacts:
                f.write(
                    f"contact {c.frm} {c.to} {as_int(c.t_start, 't_start')} "
                    f"{as_int(c.t_end, 't_end')} evl {as_int(c.rate, 'rate')} "
                    f"{as_int(c.owlt, 'owlt')}\n"
                )


@dataclass
class StochasticContactGraph:
    """Eq. (1)'s collection: stochastic contacts over [0, H] + optional latent state."""

    num_nodes: int
    contacts: list[StochasticContact] = field(default_factory=list)
    horizon: float = float("inf")
    latent: LatentModel | None = None

    def add(self, contact: StochasticContact) -> None:
        self.contacts.append(contact)

    def by_id(self, cid: int) -> StochasticContact:
        return next(c for c in self.contacts if c.cid == cid)

    @property
    def is_point_mass(self) -> bool:
        return self.latent is None and all(c.is_point_mass for c in self.contacts)

    def realize(self, rng: np.random.Generator) -> Realization:
        """Draw one deterministic contact plan (the sampler of Sec. 7.2)."""
        scales = self.latent.sample(rng) if self.latent is not None else {}
        det = []
        for c in self.contacts:
            s = scales.get(c.group, 1.0) if c.group is not None else 1.0
            d = c.sample(rng, p_scale=s)
            if d is not None:
                det.append(d)
        return Realization(self.num_nodes, det)

    def mean_plan(self, include_threshold: float = 0.5) -> Realization:
        """The nominal plan a deterministic operator would build: law means,
        contacts included iff their existence probability clears the threshold."""
        det = [
            c.mean_det() for c in self.contacts if c.p_exist >= include_threshold
        ]
        return Realization(self.num_nodes, det)
