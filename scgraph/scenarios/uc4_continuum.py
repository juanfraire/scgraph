"""UC4 - multi-hop continuum: every facet at once, over a meshed backbone (revision).

The three submitted use cases each isolate one facet on a two-hop topology.
Reviewers asked for a deeper continuum in which a contact can be late, faded
and absent in the same window, and in which timing jitter compounds along a
route. UC4 is that case. A remote terminal must reach a continental gateway:

    terminal --UAV--> HAPS --> LEO --(ISL mesh)--> LEO --feeder--> gateway

The space layer is the CPD skeleton `scenarios/uc4_cpd`: a 60-satellite
Walker-Delta shell (6 planes of 10, 550 km, 53 deg) with in-plane and
cross-plane inter-satellite links, three terminals at increasing distance from
a Paris gateway (Barcelona ~830 km, Rabat ~1800 km, Dakar ~4200 km), and a
stratospheric platform (HAPS, 20 km) beside each terminal. The air layer is
injected per instance. What each link class makes random:

  direct    terminal->HAPS narrowband link: the platform is on station, so
            the link is timing-certain and always present, but slow. It is
            the reliable fallback the UAV relays compete against, as the
            orbital backbone is in UC1.
  uav       terminal->UAV and UAV->HAPS hand-offs. Start jitter sigma, and the
            law is JOINT: a late opening also shortens the window (start-
            duration correlation) and degrades the link (start-rate
            correlation). The UAV may not fly at all (existence p_uav on the
            first hop). Late, faded and absent in the same window.
  isl_x     cross-plane ISLs must re-acquire as the geometry changes: the
            start is late by a random acquisition time, and the end does not
            move (start-duration correlation near -1), so acquisition eats
            into the window. In-plane ISLs stay point masses (A1).
  feeder    satellite->gateway passes: timing certain, rate lognormal (rain
            fade), and the passes share a weather factor at the gateway
            (common factor, correlation rho_wx between any two passes).
  haps_up   HAPS->satellite: timing certain, mild fade.
  d2s       terminal->satellite direct-to-satellite IoT access: timing
            certain from ephemeris, low existence (channel contention), low
            rate.

Downlinks toward terminals and platforms, and gateway uplinks, are kept as
point masses. A route to the gateway can still use them: the gap study found
routes that relay through another terminal's HAPS (satellite down to it, then
up again), which is legitimate store-carry-forward.

A route from the far terminal typically spans 5 to 9 contacts, with timing
uncertainty on two to four of them, which is the compounding regime of the
recursion's max(A_{i-1}, T_s) the submitted scenarios could not exercise.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np

from scgraph.contact import StochasticContact
from scgraph.distributions import Dirac, LogNormal, TruncNormal, TruncStudentT, Uniform
from scgraph.graph import StochasticContactGraph
from scgraph.routing.deterministic import cgr_route

CPD_PLAN = Path(__file__).resolve().parents[2] / "scenarios" / "uc4_cpd"
GATEWAY = 102
TERMINALS = {"near": 111, "mid": 112, "far": 113}
HAPS_OF = {111: 121, 112: 122, 113: 123}
SAT_BASE, SATS_PER_PLANE = 200, 10

BUNDLE = 100.0
# volume units per second; a bundle of 100 takes 10 s over a UAV hop, 200 s over
# the narrowband direct link, 2 s over an ISL and 50 s over a feeder pass at
# nominal rate (feeder windows last up to ~8 min, so rain fade bites at the
# end of a pass)
RATES = {"uav": 10.0, "direct": 0.5, "haps_up": 10.0, "isl": 50.0, "feeder": 2.0,
         "d2s": 2.0}


def _plane(cpd_id: int) -> int:
    return (cpd_id - SAT_BASE) // SATS_PER_PLANE


@functools.lru_cache(maxsize=8)
def load_skeleton(path: str | None = None):
    """The CPD plan, minus ground-to-ground pairs (the air layer is injected)."""
    from scgraph.io import load_cpd_plan

    sk = load_cpd_plan(path or CPD_PLAN, layer="space")
    lab = sk.label
    keep = [c for c in sk.contacts if lab[c.frm] >= SAT_BASE or lab[c.to] >= SAT_BASE]
    sk.contacts[:] = keep
    return sk


def _classify(sk, c) -> str:
    a, b = sk.label[c.frm], sk.label[c.to]
    if a >= SAT_BASE and b >= SAT_BASE:
        return "isl_x" if _plane(a) != _plane(b) else "isl_in"
    if a >= SAT_BASE and b == GATEWAY:
        return "feeder"
    if a in HAPS_OF.values() and b >= SAT_BASE:
        return "haps_up"
    if a in TERMINALS.values() and b >= SAT_BASE:
        return "d2s"
    return "down"          # sat->terminal/HAPS, gateway->sat: unused toward the gateway


def annotate(
    sk,
    *,
    acq_mean: float = 20.0,
    acq_sd: float = 10.0,
    fade_cv: float = 0.5,
    rho_wx: float = 0.3,
    haps_cv: float = 0.2,
    p_d2s: float = 0.35,
) -> list[StochasticContact]:
    """The stochastic annotation of the space skeleton (Stage 2 of the pipeline)."""
    out = []
    for c in sk.contacts:
        k = _classify(sk, c)
        s0, d0 = c.start.mean(), c.duration.mean()
        if k == "isl_in" or (k == "isl_x" and (acq_sd <= 0.0 or d0 < 2.0)):
            out.append(c.with_laws(rate=Dirac(RATES["isl"]), kind=k))
        elif k == "isl_x":
            # late by the acquisition time A; the end stays put, so the window
            # shrinks by A. Marginals: start = s0 + A, duration = d0 - A.
            hi = max(0.0, min(4.0 * acq_sd + acq_mean, d0))
            start = TruncNormal(s0 + acq_mean, acq_sd, s0, s0 + hi)
            dur = TruncNormal(d0 - acq_mean, acq_sd, d0 - hi, d0)
            out.append(c.with_laws(
                start=start, duration=dur, rate=Dirac(RATES["isl"]), kind=k,
                corr=((1.0, -0.99, 0.0), (-0.99, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ))
        elif k == "feeder":
            out.append(c.with_laws(
                rate=LogNormal.from_mean_cv(RATES["feeder"], fade_cv), kind=k,
                factor="wx_gw" if rho_wx > 0 else None, factor_attr="rate",
                factor_rho=rho_wx,
            ))
        elif k == "haps_up":
            out.append(c.with_laws(rate=LogNormal.from_mean_cv(RATES["haps_up"], haps_cv),
                                   kind=k))
        elif k == "d2s":
            out.append(c.with_laws(rate=Dirac(RATES["d2s"]), p_exist=p_d2s, kind=k))
        else:
            out.append(c.with_laws(kind=k))
    return out


def jitter_law(nominal: float, sigma: float, family: str = "normal"):
    """Start-time law of standard deviation sigma around `nominal`.

    "normal" (the paper's default, truncated at 3.5 sigma), "t3" (Student-t,
    3 degrees of freedom, same untruncated sd, truncated at 10 sigma), or
    "uniform" (same sd). Used by the generality sweep to vary the family.
    """
    if family == "normal":
        return TruncNormal(nominal, sigma, nominal - 3.5 * sigma, nominal + 3.5 * sigma)
    if family == "t3":
        return TruncStudentT.matched_std(nominal, sigma, 3.0, nominal - 10 * sigma,
                                         nominal + 10 * sigma)
    if family == "uniform":
        h = math.sqrt(3.0) * sigma
        return Uniform(nominal - h, nominal + h)
    raise ValueError(f"unknown jitter family {family!r}")


def uav_law(nominal_start: float, nominal_dur: float, sigma: float,
            fade_cv: float = 0.3, couple: bool = True, family: str = "normal"):
    """Joint law of one UAV hand-off window: (start, duration, rate) + copula."""
    if sigma <= 0.0:
        return Dirac(nominal_start), Dirac(nominal_dur), Dirac(RATES["uav"]), None
    start = jitter_law(nominal_start, sigma, family)
    # the window loses about what the start gains (the platform's schedule
    # ends on time); duration sd scales with sigma, capped by the window
    dsd = min(0.8 * sigma, 0.45 * nominal_dur)
    dur = TruncNormal(nominal_dur, dsd, 0.0, nominal_dur + 3.5 * dsd)
    rate = LogNormal.from_mean_cv(RATES["uav"], fade_cv)
    corr = ((1.0, -0.8, -0.5), (-0.8, 1.0, 0.4), (-0.5, 0.4, 1.0)) if couple else None
    return start, dur, rate, corr


def make_instance(
    rng: np.random.Generator,
    sigma: float,
    terminal: str = "far",
    n_uav: int = 3,
    margin: float = 60.0,
    p_uav: float = 0.9,
    couple: bool = True,
    skeleton=None,
    annotation: dict | None = None,
    window: float = 1800.0,
    deadline_mode: str = "reliable",
    size: float = BUNDLE,
    jitter_family: str = "normal",
) -> tuple[StochasticContactGraph, dict]:
    """One UC4 instance at UAV timing jitter `sigma` (seconds).

    The release time is drawn uniformly over the plan. The deadline sits
    `margin` seconds after a nominal arrival, chosen by `deadline_mode`:

      "reliable"  the best nominal arrival that avoids the UAV relays (the
                  narrowband fallback). At margin >= 0 a timing-certain access
                  path exists, as the backbone does in UC1, and what remains
                  random is the space segment.
      "nominal"   the nominal plan's best arrival overall (usually through a
                  UAV): the operator's own promise plus a fixed slack.

    The graph is restricted to the contacts whose support overlaps
    [t0, t0 + window], which bounds the search without changing any route
    that can meet the deadline. `size` is the bundle volume (100 by default)
    and `jitter_family` the UAV start-time law (see `jitter_law`).
    """
    sk = skeleton if skeleton is not None else load_skeleton()
    ann = dict(annotation or {})
    space = annotate(sk, **ann)
    src_id = TERMINALS[terminal]
    src, haps, dst = sk.node(src_id), sk.node(HAPS_OF[src_id]), sk.node(GATEWAY)

    t0 = float(rng.uniform(0.0, sk.horizon - window - 600.0))
    num_nodes = sk.num_nodes + n_uav
    g = StochasticContactGraph(num_nodes, space, horizon=sk.horizon).window(t0, t0 + window)
    cid = max(c.cid for c in sk.contacts) + 1

    # the platform's narrowband fallback: certain, continuous, slow
    g.add(StochasticContact(cid, src, haps, Dirac(t0), Dirac(window), Dirac(RATES["direct"]),
                            owlt=0.0001, layer="air", kind="direct"))
    cid += 1

    for k in range(n_uav):
        uav = sk.num_nodes + k
        s1 = t0 + float(rng.uniform(20.0, 80.0))
        d1 = float(rng.uniform(50.0, 80.0))
        s2 = s1 + size / RATES["uav"] + float(rng.uniform(5.0, 25.0))
        d2 = float(rng.uniform(50.0, 80.0))
        for (a, b, s, d, pe) in ((src, uav, s1, d1, p_uav), (uav, haps, s2, d2, 1.0)):
            st, du, ra, co = uav_law(s, d, sigma, couple=couple, family=jitter_family)
            g.add(StochasticContact(cid, a, b, st, du, ra, owlt=0.0001, p_exist=pe,
                                    layer="air", corr=co, kind="uav"))
            cid += 1

    mean = g.mean_plan()
    nominal_route, nominal_arr = cgr_route(mean, src, dst, size, t0)
    uav_cids = frozenset(c.cid for c in g.contacts if c.kind == "uav")
    rel_route, rel_arr = cgr_route(mean, src, dst, size, t0, excluded_cids=uav_cids)
    if nominal_route is None or rel_route is None:   # pragma: no cover - dense mesh
        raise RuntimeError("no nominal route to the gateway")
    ref = rel_arr if deadline_mode == "reliable" else nominal_arr
    deadline = ref + margin
    params = {
        "src": src, "dst": dst, "size": size, "t0": t0, "deadline": deadline,
        "nominal_arrival": nominal_arr, "nominal_hops": len(nominal_route),
        "reliable_arrival": rel_arr, "reliable_hops": len(rel_route),
        "terminal": terminal, "deadline_mode": deadline_mode,
    }
    return g, params
