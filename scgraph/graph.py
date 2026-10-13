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
        # The index is rebuilt whenever the contact list changed size, which
        # covers add(); a linear scan was the dominant cost on large plans.
        idx = self.__dict__.get("_index")
        if idx is None or self.__dict__.get("_index_n") != len(self.contacts):
            idx = {c.cid: c for c in self.contacts}
            self.__dict__["_index"] = idx
            self.__dict__["_index_n"] = len(self.contacts)
        return idx[cid]

    def replaced(self, contacts: list[StochasticContact]) -> "StochasticContactGraph":
        """Same graph (nodes, horizon, latent state) over a different contact list."""
        return StochasticContactGraph(
            num_nodes=self.num_nodes, contacts=list(contacts),
            horizon=self.horizon, latent=self.latent,
        )

    def window(self, t_lo: float, t_hi: float) -> "StochasticContactGraph":
        """Contacts whose support can overlap [t_lo, t_hi], the only ones a
        bundle released at t_lo with deadline t_hi could ever use."""
        keep = []
        for c in self.contacts:
            s_lo, e_hi = c.support_window()
            if e_hi >= t_lo and s_lo <= t_hi:
                keep.append(c)
        return self.replaced(keep)

    def independent_twin(self) -> "StochasticContactGraph":
        """Every contact keeps its marginals but loses its within-contact coupling."""
        return self.replaced([c.independent_twin() for c in self.contacts])

    @property
    def is_point_mass(self) -> bool:
        return self.latent is None and all(c.is_point_mass for c in self.contacts)

    def realize(self, rng: np.random.Generator) -> Realization:
        """Draw one deterministic contact plan (the sampler of Sec. 7.2)."""
        scales = self.latent.sample(rng) if self.latent is not None else {}
        names = sorted({c.factor for c in self.contacts
                        if c.factor is not None and c.factor_rho > 0.0})
        zf = {n: rng.standard_normal(1) for n in names} if names else None
        det = []
        for c in self.contacts:
            s = scales.get(c.group, 1.0) if c.group is not None else 1.0
            d = c.sample(rng, p_scale=s, zf=zf)
            if d is not None:
                det.append(d)
        return Realization(self.num_nodes, det)

    def realize_batch(self, rng: np.random.Generator, m: int) -> list[Realization]:
        """m independent realizations at once (vectorised per contact).

        Equivalent in law to calling `realize` m times, with the latent state
        and every common factor drawn once per realization and shared across
        its contacts; much faster on large graphs because each contact's laws
        are sampled in one call. The random stream differs from m calls to
        `realize`, so results are not draw-for-draw comparable between the two.
        """
        if self.latent is not None:
            probs = [p for p, _ in self.latent.states]
            idx = rng.choice(len(self.latent.states), size=m, p=probs)
        else:
            idx = None
        names = sorted({c.factor for c in self.contacts
                        if c.factor is not None and c.factor_rho > 0.0})
        zf = {n: rng.standard_normal(m) for n in names} if names else None
        per = [[] for _ in range(m)]
        for c in self.contacts:
            if idx is not None and c.group is not None:
                scale = np.array([self.latent.states[i][1].get(c.group, 1.0) for i in idx])
            else:
                scale = 1.0
            ex, ts, te, rate = c.sample_attrs(rng, m, p_scale=scale, zf=zf)
            for j in np.flatnonzero(ex):
                per[j].append(DetContact(c.cid, c.frm, c.to, float(ts[j]), float(te[j]),
                                         float(rate[j]), c.owlt))
        return [Realization(self.num_nodes, cl) for cl in per]

    def mean_plan(self, include_threshold: float = 0.5) -> Realization:
        """The nominal plan a deterministic operator would build: law means,
        contacts included iff their existence probability clears the threshold."""
        det = [
            c.mean_det() for c in self.contacts if c.p_exist >= include_threshold
        ]
        return Realization(self.num_nodes, det)
