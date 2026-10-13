"""Control-plane size of the stochastic sidecar (revision T8, R1.e).

Reviewer 1 asked what it costs to keep a planner's stochastic annotations in
sync with the network. A stochastic contact plan is the classical plan plus a
sidecar: one extra record per contact that carries a law, and nothing for a
point-mass contact (which is the subsumption property on the wire). This
runner counts the bytes, for the CPD plans of run_overhead.py (UC1-style air
injection on scale_{6,12,24,48}) and for the 3,606-contact UC4 ISL plan, and
converts them into uplink time.

Records (little-endian, packed; sizes come from struct.calcsize):

  classical, ION text   `a contact +start +end from to rate`, one line per
                        contact, rendered from the plan itself (integer
                        seconds, CPD node numbers, the plan's rate field).
                        ION's `a range` lines (one per node pair) are extra.
  classical, binary     u32 start, u32 end, u16 from, u16 to, u32 rate: 16 B.
  sidecar header        u32 contact index, u8 bitmap of present parts: 5 B.
  TruncNormal law       mu, sigma, lo, hi as float32: 16 B (start or duration).
  LogNormal law         mu, sigma as float32: 8 B (rate / volume).
  existence             p as float32: 4 B.
  group / factor        u16 id, float32 loading: 6 B.
  copula                the 3 off-diagonal coefficients as float32: 12 B.

Law families, per annotated contact (header included):
  timing   = header + TruncNormal                                   21 B
  volume   = header + LogNormal                                     13 B
  exist    = header + existence                                      9 B
  tve      = header + TruncNormal + LogNormal + existence           33 B
  joint    = tve + group/factor + copula                            51 B

Scenarios, per plan:
  as_modelled   the annotation the experiment actually uses, encoded from the
                graph (UC1: TruncNormal start on the six air contacts). UC4's
                CPD skeleton carries no laws, so its "air/ground" rows below
                stand in for it.
  all_<family>  every contact of the plan carries that family (upper bound).
  refresh       the bytes to resend when only the air/ground contacts change,
                or when a fraction f in {1%, 10%, 100%} of contacts change,
                for the timing and joint families, next to the classical
                refresh of the same contacts (16 B binary each).
  telemetry     the downlink that feeds the modelling plane: per observed
                contact u32 index + observed start, end, delivered volume as
                float32 = 16 B, for the same fractions.

Uplink time is bytes * 8 / rate at 9.6 kbit/s, 100 kbit/s and 1 Mbit/s,
payload only (bundle and frame headers add a fixed few tens of bytes per
refresh). Everything here is counted, nothing is timed, so the CSV
reproduces exactly.

Usage: python experiments/run_sidecar.py
Outputs: results/overhead_sidecar.csv
"""

from __future__ import annotations

import csv
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from scgraph.distributions import (
    Dirac, LogNormal, TruncNormal, TruncStudentT, Uniform,
)
from scgraph.io import load_cpd_plan
from scgraph.scenarios.uc1_event import make_instance_cpd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260728
SIGMA = 40.0
RATES = {"9k6": 9_600.0, "100k": 100_000.0, "1M": 1_000_000.0}
FRACTIONS = [0.01, 0.10, 1.00]
SAT_ID_MIN = 200          # CPD numbering: constellation ids start at base-id 200
AIR_IDS = {121, 122, 123}  # UC4 HAPS

# record layouts
CLASSIC_BIN = "<IIHHI"
HDR = "<IB"
TN = "<ffff"
LN = "<ff"
EXIST = "<f"
FACTOR = "<Hf"
COPULA = "<fff"
TELEMETRY = "<Ifff"

B = struct.calcsize
FAMILY_BYTES = {
    "timing": B(HDR) + B(TN),
    "volume": B(HDR) + B(LN),
    "exist": B(HDR) + B(EXIST),
    "tve": B(HDR) + B(TN) + B(LN) + B(EXIST),
    "joint": B(HDR) + B(TN) + B(LN) + B(EXIST) + B(FACTOR) + B(COPULA),
}


def law_bytes(d) -> bytes:
    """Encode one attribute law; a point mass costs nothing (it is in the plan)."""
    if isinstance(d, Dirac):
        return b""
    if isinstance(d, TruncNormal):
        return struct.pack(TN, d.mu, d.sigma, d.lo, d.hi)
    if isinstance(d, LogNormal):
        return struct.pack(LN, d.mu, d.sigma)
    if isinstance(d, Uniform):
        return struct.pack("<ff", d.lo, d.hi)
    if isinstance(d, TruncStudentT):
        return struct.pack("<fffff", d.mu, d.scale, d.nu, d.lo, d.hi)
    raise TypeError(f"no wire format for {type(d).__name__}")


def encode_sidecar(c, index: int) -> bytes:
    """The sidecar record of one contact; b'' when the contact is a point mass."""
    parts, bits = [], 0
    for j, d in enumerate((c.start, c.duration, c.rate)):
        enc = law_bytes(d)
        if enc:
            parts.append(enc)
            bits |= 1 << j
    if c.p_exist < 1.0:
        parts.append(struct.pack(EXIST, c.p_exist))
        bits |= 1 << 3
    if c.factor is not None or c.group is not None:
        fid = zlib.crc32((c.factor or c.group).encode()) & 0xFFFF
        parts.append(struct.pack(FACTOR, fid, c.factor_rho))
        bits |= 1 << 4
    if c.corr is not None:
        r = np.asarray(c.corr, dtype=float)
        parts.append(struct.pack(COPULA, r[0, 1], r[0, 2], r[1, 2]))
        bits |= 1 << 5
    if not parts:
        return b""
    return struct.pack(HDR, index, bits) + b"".join(parts)


def ion_line(c, label) -> str:
    s = c.start.mean()
    e = s + c.duration.mean()
    return (f"a contact +{int(round(s))} +{int(round(e))} "
            f"{label(c.frm)} {label(c.to)} {int(round(c.rate.mean()))}\n")


def plans():
    """(name, contacts, label function, n_air, n_ground, as_modelled bytes)."""
    out = []
    for n in (6, 12, 24, 48):
        sk = load_cpd_plan(ROOT / "scenarios" / f"scale_{n}")
        rng = np.random.default_rng(SEED + 1000 * n)          # run_overhead's i = 0
        g, _ = make_instance_cpd(rng, SIGMA, skeleton=sk)
        lab = sk.label
        # injected relays are numbered after the skeleton; give them CPD-style ids
        label = lambda i, lab=lab: lab.get(i, 300 + i)
        n_air = sum(1 for c in g.contacts if c.layer == "air")
        n_ground = sum(1 for c in g.contacts
                       if c.layer != "air" and (label(c.frm) < SAT_ID_MIN
                                                or label(c.to) < SAT_ID_MIN))
        side = b"".join(encode_sidecar(c, k) for k, c in enumerate(g.contacts))
        out.append((f"scale_{n}+uc1air", g.contacts, label, n_air, n_ground, len(side)))

    sk = load_cpd_plan(ROOT / "scenarios" / "uc4_cpd")
    lab = sk.label
    label = lambda i, lab=lab: lab[i]
    n_air = sum(1 for c in sk.contacts if lab[c.frm] in AIR_IDS or lab[c.to] in AIR_IDS)
    n_ground = sum(1 for c in sk.contacts
                   if (lab[c.frm] < SAT_ID_MIN or lab[c.to] < SAT_ID_MIN)
                   and not (lab[c.frm] in AIR_IDS or lab[c.to] in AIR_IDS))
    out.append(("uc4_cpd", sk.contacts, label, n_air, n_ground, None))
    return out


def uplink(nbytes: float) -> dict:
    return {f"uplink_s_{k}": nbytes * 8.0 / r for k, r in RATES.items()}


def run():
    rows = []
    for name, contacts, label, n_air, n_ground, side_bytes in plans():
        N = len(contacts)
        n_ag = n_air + n_ground

        def row(item, family, fraction, records, nbytes):
            rows.append({
                "plan": name, "contacts": N, "n_air": n_air, "n_ground": n_ground,
                "n_air_ground": n_ag, "item": item, "family": family,
                "fraction": fraction, "records": records, "bytes": nbytes,
                "bytes_per_contact": nbytes / N, **uplink(nbytes),
            })

        text = sum(len(ion_line(c, label).encode()) for c in contacts)
        row("classical_ion_text", "", 1.0, N, text)
        row("classical_binary", "", 1.0, N, N * B(CLASSIC_BIN))
        if side_bytes is not None:
            row("sidecar_as_modelled", "uc1_air_timing", n_air / N, n_air, side_bytes)
        for fam, fb in FAMILY_BYTES.items():
            row("sidecar_all", fam, 1.0, N, N * fb)
        for fam in ("timing", "joint"):
            fb = FAMILY_BYTES[fam]
            row("refresh_air_ground", fam, n_ag / N, n_ag, n_ag * fb)
            row("refresh_air_only", fam, n_air / N, n_air, n_air * fb)
            for f in FRACTIONS:
                k = math.ceil(f * N)
                row("refresh_fraction", fam, f, k, k * fb)
        for f in FRACTIONS:
            k = math.ceil(f * N)
            row("refresh_fraction_classical", "binary", f, k, k * B(CLASSIC_BIN))
            row("telemetry_fraction", "obs", f, k, k * B(TELEMETRY))
        row("refresh_air_ground_classical", "binary", n_ag / N, n_ag,
            n_ag * B(CLASSIC_BIN))
        print(f"  {name:18s} N={N:6d} air={n_air:4d} ground={n_ground:5d}"
              f"  ION text={text / N:5.1f} B/contact  sidecar(joint, all)="
              f"{N * FAMILY_BYTES['joint'] / 1024:7.1f} KiB")
    return rows


def main():
    t0 = time.time()
    print("Sidecar size and uplink time:")
    print("  per-contact record bytes:",
          {"classical_binary": B(CLASSIC_BIN), **FAMILY_BYTES,
           "telemetry": B(TELEMETRY)})
    rows = run()
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "overhead_sidecar.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"done in {time.time() - t0:.1f}s -> results/overhead_sidecar.csv "
          f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
