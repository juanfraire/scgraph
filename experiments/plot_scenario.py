"""Scenario figure: the orbital skeleton UC1 and UC2 actually run on.

The paper (Sec. VII, "The orbital skeleton") claims that the space layer is
*computed* rather than assumed, and that connectivity between the event site
and the gateway is pass-limited. Both claims are visual, and until now the
reader had only prose. This script draws them.

Three panels:

  (a) the constellation. Walker-Delta 12/3/1 at 600 km and 53 deg, the
      configuration `scenarios/uc1_skeleton.yaml` hands to CPD, drawn from its
      closed-form geometry: three orbit planes, twelve satellites at epoch, the
      event site and the gateway, and the event site's 10-degree visibility cap.

  (b) every ground window in the checked-in CPD plan over the six-hour horizon,
      one row per satellite, uplink (event site -> satellite) against downlink
      (satellite -> gateway). The picture is mostly white: connectivity is a
      handful of minutes scattered through six hours, which is what "pass
      limited" means and what sets UC1's deadline regime.

  (c) one pass, magnified. The fair contact-plan design gives a satellite one
      interface at a time, so within a single pass it serves the event site,
      then the gateway, in consecutive non-overlapping slots. A bundle
      therefore goes up in one slot, is *held on board* across the switch, and
      goes down in the next: store-carry-forward produced by the plan rather
      than by geometry, since the two sites are only 391 km apart and sit
      inside the same 1762 km footprint. Verified over the whole plan: of the
      35 uplink windows, none overlaps a downlink window on the same satellite.

Panels (b) and (c) are read from `scenarios/uc1_cpd/`, the same plan the
experiments consume, so the figure cannot drift away from the results.

Dependencies: numpy and matplotlib only. Porting the cartopy/astropy ground-
track script from the sparse-constellation paper was the alternative; it would
have added two heavy dependencies (and a third package vendored from another
repo) to a companion library that installs with `pip install -e .` today, for
one figure. The orthographic projection here is forty lines of numpy and needs
no map data, and the geometry it draws is closed-form Walker-Delta rather than
a re-propagation, so there is nothing to disagree with CPD about.

Usage: python experiments/plot_scenario.py [--quick]
Outputs: figures/scenario.{pdf,png}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _style import COL_BASE1, COL_BOUND, COL_SCG

from scgraph.graph import StochasticContactGraph
from scgraph.routing.deterministic import cgr_route
from scgraph.scenarios.uc1_event import (
    EVENT_SITE,
    GATEWAY,
    PASS_BAND,
    load_skeleton,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260709                      # the seed run_uc1.py draws release times with

# --- scenario constants, mirroring scenarios/uc1_skeleton.yaml --------------
R_E = 6378.137                       # equatorial radius [km]
ALT_KM = 600.0                       # "D:600:53:12/3/1"
INC_DEG = 53.0
N_PLANES, N_PER_PLANE, PHASING = 3, 4, 1
MASK_DEG = 10.0                      # min-elevation-deg on both ground nodes
HORIZON_S = 21600.0                  # duration-s
SITES = {                            # id -> (name, lat, lon)
    EVENT_SITE: ("event site", 45.76, 4.83),
    GATEWAY: ("gateway", 48.85, 2.35),
}
SUB_OBS = (40.0, 4.0)                # sub-observer point: western Europe faces us

COL_UP, COL_DN = COL_SCG, COL_BOUND
COL_SITE = COL_BASE1                  # ground segment, shared palette


# ---------------------------------------------------------------------------
# panel (a): closed-form Walker-Delta geometry, orthographic projection
# ---------------------------------------------------------------------------


def _rx(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def geo_to_ecef(lat_deg: float, lon_deg: float, r: float = R_E) -> np.ndarray:
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return r * np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def view_basis(lat0: float, lon0: float):
    """Screen basis (east, north, towards-viewer) for an orthographic view."""
    n = geo_to_ecef(lat0, lon0, 1.0)
    e = np.cross(np.array([0.0, 0.0, 1.0]), n)
    e /= np.linalg.norm(e)
    return e, np.cross(n, e), n


def project(pts: np.ndarray, basis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e, u, n = basis
    pts = np.atleast_2d(pts)
    return pts @ e, pts @ u, pts @ n


def draw_3d_curve(ax, pts, basis, *, color, lw, zorder, front_alpha=1.0,
                  back_alpha=0.0, ls="-"):
    """Plot a projected 3-D curve, fading the part occluded by the Earth."""
    x, y, d = project(pts, basis)
    occluded = (d < 0.0) & (np.hypot(x, y) < R_E)
    for hidden, alpha in ((False, front_alpha), (True, back_alpha)):
        if alpha <= 0.0:
            continue
        seg = np.where(occluded == hidden, 1.0, np.nan)
        ax.plot(x * seg, y * seg, color=color, lw=lw, alpha=alpha, ls=ls,
                zorder=zorder, solid_capstyle="round")


def small_circle(centre_unit: np.ndarray, ang: float, n: int) -> np.ndarray:
    """Points on the sphere at central angle `ang` from `centre_unit`."""
    ref = np.array([0.0, 0.0, 1.0])
    if abs(centre_unit @ ref) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    a1 = np.cross(centre_unit, ref)
    a1 /= np.linalg.norm(a1)
    a2 = np.cross(centre_unit, a1)
    th = np.linspace(0.0, 2.0 * np.pi, n)
    return R_E * (
        np.cos(ang) * centre_unit
        + np.sin(ang) * (np.cos(th)[:, None] * a1 + np.sin(th)[:, None] * a2)
    )


def panel_constellation(ax, n_pts: int) -> None:
    import matplotlib.patches as mpatches

    basis = view_basis(*SUB_OBS)
    a_orb = R_E + ALT_KM
    inc = np.radians(INC_DEG)

    ax.add_patch(mpatches.Circle((0, 0), R_E, facecolor="#eef2f6",
                                 edgecolor="#9aa7b4", lw=0.6, zorder=1))

    # graticule
    for lat in (-60, -30, 0, 30, 60):
        ring = small_circle(np.array([0.0, 0.0, 1.0]), np.radians(90.0 - lat), n_pts)
        draw_3d_curve(ax, ring, basis, color="#c2ccd6",
                      lw=0.7 if lat == 0 else 0.4, zorder=2)
    for lon in range(-180, 180, 30):
        u = np.linspace(0.0, 2.0 * np.pi, n_pts)
        ring = R_E * np.stack(
            [np.cos(u) * np.cos(np.radians(lon)),
             np.cos(u) * np.sin(np.radians(lon)), np.sin(u)], axis=1)
        draw_3d_curve(ax, ring, basis, color="#c2ccd6", lw=0.4, zorder=2)

    # the three orbit planes, and the twelve satellites at epoch
    u = np.linspace(0.0, 2.0 * np.pi, n_pts)
    ring0 = a_orb * np.stack([np.cos(u), np.sin(u), np.zeros_like(u)], axis=1)
    for j in range(N_PLANES):
        rot = _rz(2.0 * np.pi * j / N_PLANES) @ _rx(inc)
        draw_3d_curve(ax, ring0 @ rot.T, basis, color=COL_SCG, lw=0.9,
                      zorder=4, front_alpha=0.95, back_alpha=0.22)
        for k in range(N_PER_PLANE):
            arg = (2.0 * np.pi * k / N_PER_PLANE
                   + 2.0 * np.pi * PHASING * j / (N_PLANES * N_PER_PLANE))
            p = rot @ (a_orb * np.array([np.cos(arg), np.sin(arg), 0.0]))
            x, y, d = project(p, basis)
            front = not (d[0] < 0.0 and np.hypot(x[0], y[0]) < R_E)
            ax.plot(x, y, marker="o", ms=2.8, mfc=COL_SCG if front else "white",
                    mec=COL_SCG, mew=0.6, ls="none",
                    alpha=1.0 if front else 0.45, zorder=5)

    # ground segment: the two sites and the event site's visibility cap
    lam = np.arccos(R_E * np.cos(np.radians(MASK_DEG)) / a_orb) - np.radians(MASK_DEG)
    cap = small_circle(geo_to_ecef(*SITES[EVENT_SITE][1:], r=1.0), lam, n_pts)
    draw_3d_curve(ax, cap, basis, color=COL_SITE, lw=0.8, zorder=6, ls=(0, (3, 1.6)))
    for nid, (name, lat, lon) in SITES.items():
        x, y, _ = project(geo_to_ecef(lat, lon), basis)
        ax.plot(x, y, marker="^" if nid == EVENT_SITE else "s", ms=3.4,
                color=COL_SITE, ls="none", zorder=7)
        ax.annotate(name, (x[0], y[0]),
                    xytext=(-40, 26) if nid == EVENT_SITE else (26, -26),
                    textcoords="offset points", fontsize=6.8, color=COL_SITE,
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=COL_SITE,
                                    shrinkA=0, shrinkB=1.5))

    # The constellation parameters live in the LaTeX caption: an in-axes note
    # here would sit outside the square box and be clipped by _style.save's
    # tight_layout.
    lim = a_orb * 1.02
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    # adjustable="datalim" keeps the globe circular by padding the data range
    # instead of shrinking the axes box, which is what lets _style.save's
    # tight_layout still place the panel.
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


# ---------------------------------------------------------------------------
# panels (b) and (c): the ground windows of the checked-in CPD plan
# ---------------------------------------------------------------------------


def ground_windows(skeleton):
    """{cpd_sat_id: {"up": [(s,e)...], "dn": [(s,e)...]}} for the two sites."""
    lab = skeleton.label
    src, dst = skeleton.node(EVENT_SITE), skeleton.node(GATEWAY)
    out: dict[int, dict[str, list]] = {}
    for c in skeleton.contacts:
        if c.frm == src:
            key, sat = "up", c.to
        elif c.to == dst:
            key, sat = "dn", c.frm
        else:
            continue
        s = c.start.mean()
        out.setdefault(lab[sat], {"up": [], "dn": []})[key].append(
            (s, s + c.duration.mean())
        )
    return {k: {d: sorted(v) for d, v in w.items()} for k, w in sorted(out.items())}


def example_route(skeleton):
    """The first backbone route the UC1 release-time draw actually produces."""
    g = StochasticContactGraph(num_nodes=skeleton.num_nodes,
                               contacts=list(skeleton.contacts),
                               horizon=skeleton.horizon)
    plan = g.mean_plan()
    src, dst = skeleton.node(EVENT_SITE), skeleton.node(GATEWAY)
    rng = np.random.default_rng(SEED)
    lo, hi = PASS_BAND
    for _ in range(500):
        t0 = float(rng.uniform(0.0, skeleton.horizon - 2.0 * hi))
        route, arr = cgr_route(plan, src, dst, 100.0, t0)
        if route is not None and lo <= arr - t0 <= hi:
            return t0, route, arr
    raise RuntimeError("no release time inside PASS_BAND")


def _bars(ax, spans, y, color, h=0.34):
    """Draw access windows as bars, with a floor so short passes stay visible."""
    from matplotlib.patches import Rectangle

    for s, e in spans:
        ax.add_patch(Rectangle((s, y - h / 2), max(e - s, 25.0), h,
                               facecolor=color, edgecolor="none", zorder=3))


def panel_timeline(ax, windows) -> None:
    sats = [s for s, w in windows.items() if w["up"] or w["dn"]]
    for i, sat in enumerate(sats):
        ax.axhline(i, color="#eeeeee", lw=0.5, zorder=1)
        _bars(ax, windows[sat]["up"], i + 0.20, COL_UP)
        _bars(ax, windows[sat]["dn"], i - 0.20, COL_DN)
    ax.set_yticks(range(len(sats)))
    ax.set_yticklabels([f"sat {s}" for s in sats], fontsize=6.5)
    ax.set_ylim(-0.8, len(sats) - 0.2)
    ax.set_xlim(0, HORIZON_S)
    ax.set_xticks(np.arange(0, HORIZON_S + 1, 3600))
    ax.set_xticklabels([f"{int(t / 3600)}" for t in np.arange(0, HORIZON_S + 1, 3600)])
    ax.set_xlabel("time into the 6 h horizon [h]", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)


def panel_zoom(ax, windows, t0, route, arr, skeleton) -> None:
    lab = skeleton.label
    sat = lab[route[0].to]
    lo = min(t0, route[0].t_start) - 70.0
    # reach past the delivery so the satellite's *next* event-site slot is
    # visible too: the alternation is the point of the panel
    hi = max(arr, route[-1].t_end) + 300.0

    _bars(ax, [(s, e) for s, e in windows[sat]["up"] if e > lo and s < hi],
          0.24, COL_UP, h=0.30)
    _bars(ax, [(s, e) for s, e in windows[sat]["dn"] if e > lo and s < hi],
          -0.24, COL_DN, h=0.30)

    up, dn = route[0], route[-1]
    ax.annotate("", xy=(dn.t_start, -0.24), xytext=(up.t_end, 0.24),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333",
                                shrinkA=1.0, shrinkB=1.0,
                                connectionstyle="arc3,rad=-0.35"), zorder=5)
    ax.annotate("held on board", xy=(up.t_end, 0.0), xytext=(9, -1),
                textcoords="offset points", fontsize=6.8, color="#333333",
                va="center")
    ax.plot([t0], [0.72], marker="v", ms=3.6, color="#333333", clip_on=False)
    ax.annotate("bundle released", (t0, 0.72), xytext=(4, 4),
                textcoords="offset points", fontsize=6.8, color="#333333")
    ax.annotate("", xy=(up.t_start, 0.60), xytext=(t0, 0.60),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#888888",
                                shrinkA=0, shrinkB=0))
    ax.text((t0 + up.t_start) / 2.0, 0.50, "waiting for the pass", fontsize=6.8,
            color="#666666", ha="center", va="top")

    ax.set_yticks([0.24, -0.24])
    ax.set_yticklabels(["uplink", "downlink"], fontsize=6.8)
    ax.set_ylim(-0.62, 0.92)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"one pass of sat {sat} [s into the horizon]", fontsize=7.5)
    ax.tick_params(labelsize=6.8)
    ax.grid(axis="x", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------


def main() -> None:
    quick = "--quick" in sys.argv
    t_start = time.time()
    _style.setup()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    skeleton = load_skeleton()
    windows = ground_windows(skeleton)
    t0, route, arr = example_route(skeleton)

    n_up = sum(len(w["up"]) for w in windows.values())
    overlaps = sum(
        1
        for w in windows.values()
        for a, b in w["up"]
        for c, d in w["dn"]
        if min(b, d) - max(a, c) > 0.0
    )
    print(f"{n_up} uplink windows, {overlaps} overlapping a downlink on the same "
          f"satellite; example release t0={t0:.0f}s, delivered {arr - t0:.0f}s later "
          f"over {len(route)} hops")

    # Panel (a) holds an equal-aspect globe, so it fills whichever of its two
    # box dimensions is smaller. Giving column 0 a roughly square box is what
    # lets the Earth use the full height instead of floating in it; the figure
    # is widened to pay for that without squeezing the timelines. wspace still
    # keeps a clear gutter, since the globe and the timelines are different
    # kinds of picture and should not read as one panel.
    fig = plt.figure(figsize=(7.5, 3.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.23], height_ratios=[1.0, 1.0],
                          wspace=0.26, hspace=0.72)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_constellation(ax_a, n_pts=90 if quick else 400)
    panel_timeline(ax_b, windows)
    panel_zoom(ax_c, windows, t0, route, arr, skeleton)

    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        ax.set_title(tag, loc="left", fontsize=7.5, pad=2.5)

    ax_b.legend(
        handles=[Line2D([], [], color=COL_UP, lw=3, label="event site $\\to$ satellite"),
                 Line2D([], [], color=COL_DN, lw=3, label="satellite $\\to$ gateway")],
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.8,
        handlelength=1.4, columnspacing=1.4, borderaxespad=0.0)

    (ROOT / "figures").mkdir(exist_ok=True)
    _style.save(fig, str(ROOT / "figures" / "scenario"))
    print(f"done in {time.time() - t_start:.1f}s -> figures/scenario.pdf")


if __name__ == "__main__":
    main()
