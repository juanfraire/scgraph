"""Read a Contact Plan Designer export as the deterministic orbital skeleton.

This is Stage 1 of the populate-and-learn pipeline (paper Sec. IV-A): the space
layer is *generated* from orbital mechanics rather than assumed. CPD propagates
the constellation, computes accesses, resolves interface conflicts, and exports
a classical contact plan; we read it back as stochastic contacts whose laws are
all point masses, which is assumption A1 (timing-certain, capacity-uncertain at
the space layer) expressed in code.

Because a point-mass graph is exactly what a classical router consumes, the
skeleton loaded here is the same object A-SABR would run on, which is what
anchors the subsumption claim in the toolchain and not only in the prose.

Two input formats are accepted:
  plan.json                  CPD's native export; carries float window edges,
                             per-contact range and (when computed) OWLT.
  ionFormatContactPlan.txt   the ION `a contact +start +end tx rx rate` lines;
                             integer-valued, which is what the A-SABR lexer wants.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from scgraph.contact import StochasticContact

# speed of light, km/s: CPD reports range in km and leaves owlt null for
# near-Earth scenarios where it is negligible
C_KM_S = 299792.458

_ION_RE = re.compile(
    r"^a\s+contact\s+\+?(-?[\d.]+)\s+\+?(-?[\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)"
)


@dataclass
class CpdSkeleton:
    """A deterministic contact plan plus the node-id mapping used to compact it.

    CPD node ids are sparse and meaningful (101 = a ground station, 2xx = the
    constellation). scgraph indexes nodes densely from zero, so `index` maps
    CPD id -> scgraph index and `label` inverts it. Keeping both means a route
    can always be reported back in the operator's own numbering.
    """

    contacts: list[StochasticContact]
    index: dict[int, int]
    horizon: float
    epoch: float = 0.0
    source: str = ""

    @property
    def num_nodes(self) -> int:
        return len(self.index)

    @property
    def label(self) -> dict[int, int]:
        return {v: k for k, v in self.index.items()}

    def node(self, cpd_id: int) -> int:
        """scgraph index of a CPD node id, by name in the scenario's numbering."""
        if cpd_id not in self.index:
            raise KeyError(
                f"node {cpd_id} is not in this plan; known ids: {sorted(self.index)}"
            )
        return self.index[cpd_id]

    def shifted(self, dt: float) -> "CpdSkeleton":
        """The same skeleton with the time origin moved by `dt`.

        Sampling a window of a long plan gives independent instances that are
        all still real orbital geometry, which is how one CPD run supplies the
        many random instances an experiment sweep needs.
        """
        from scgraph.distributions import Dirac

        out = []
        for c in self.contacts:
            s = c.start.mean() - dt
            out.append(
                StochasticContact(
                    c.cid, c.frm, c.to, Dirac(s), c.duration, c.rate,
                    owlt=c.owlt, p_exist=c.p_exist, layer=c.layer, group=c.group,
                )
            )
        return CpdSkeleton(out, self.index, self.horizon - dt, self.epoch + dt,
                           self.source)


def _mk(cid, frm, to, start, end, rate, owlt, layer):
    from scgraph.distributions import Dirac

    return StochasticContact(
        cid=cid, frm=frm, to=to,
        start=Dirac(float(start)),
        duration=Dirac(max(0.0, float(end) - float(start))),
        rate=Dirac(float(rate)),
        owlt=float(owlt),
        p_exist=1.0,
        layer=layer,
    )


def _skeleton_from_rows(rows, layer, source):
    """rows: iterable of (tx, rx, start, end, rate, owlt) in CPD numbering."""
    rows = sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    ids = sorted({r[0] for r in rows} | {r[1] for r in rows})
    index = {cpd_id: i for i, cpd_id in enumerate(ids)}

    contacts, horizon = [], 0.0
    for cid, (tx, rx, s, e, rate, owlt) in enumerate(rows):
        contacts.append(_mk(cid, index[tx], index[rx], s, e, rate, owlt, layer))
        horizon = max(horizon, float(e))
    return CpdSkeleton(contacts, index, horizon, 0.0, source)


def load_cpd_plan(
    path: str | Path,
    *,
    layer: str = "space",
    rate_scale: float = 1.0,
    keep_nodes: set[int] | None = None,
) -> CpdSkeleton:
    """Load `plan.json` or `ionFormatContactPlan.txt` from a CPD output dir.

    `path` may point at either file or at the directory containing them.
    `keep_nodes`, when given, restricts the plan to contacts whose endpoints are
    both in the set, which is how a scenario picks out the subset of the
    constellation it actually models.
    """
    p = Path(path)
    if p.is_dir():
        cand = [p / "plan.json", p / "ionFormatContactPlan.txt"]
        found = next((c for c in cand if c.exists()), None)
        if found is None:
            raise FileNotFoundError(f"no CPD plan export under {p}")
        p = found

    if p.suffix == ".json":
        rows = _rows_from_json(p, rate_scale)
    else:
        rows = _rows_from_ion(p, rate_scale)

    if keep_nodes is not None:
        rows = [r for r in rows if r[0] in keep_nodes and r[1] in keep_nodes]
    if not rows:
        raise ValueError(f"{p} yielded no contacts (keep_nodes too restrictive?)")
    return _skeleton_from_rows(rows, layer, str(p))


def _rows_from_json(p: Path, rate_scale: float):
    data = json.loads(p.read_text())
    rows = []
    for c in data["contacts"]:
        if c.get("transit_only"):
            continue
        owlt = c.get("owlt_s")
        if owlt is None:
            rng_km = c.get("range_km")
            # near-Earth OWLT is milliseconds; compute it when CPD left it out
            owlt = (rng_km / C_KM_S) if rng_km else 0.0
        rows.append(
            (
                int(c["tx_node"]), int(c["rx_node"]),
                float(c["start_s"]), float(c["end_s"]),
                float(c["rate_bps"]) * rate_scale, float(owlt),
            )
        )
    return rows


def _rows_from_ion(p: Path, rate_scale: float):
    rows = []
    for line in p.read_text().splitlines():
        m = _ION_RE.match(line.strip())
        if not m:
            continue
        s, e, tx, rx, rate = m.groups()
        rows.append(
            (int(tx), int(rx), float(s), float(e), float(rate) * rate_scale, 0.0)
        )
    return rows
