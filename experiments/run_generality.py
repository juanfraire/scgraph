"""Generality of the UC4 results: one-at-a-time sensitivity around the base point.

Answers R3: "the main use cases appear to rely on a relatively specific orbital
setup. Additional experiments varying constellation density, topology depth,
source-destination locations, traffic volumes, deadline regimes, uncertainty
levels, and distributional assumptions would help establish whether the
reported improvements generalize."

Base point (the UC4 sweep's centre): far terminal (Dakar) to the Paris gateway,
60-satellite ISL shell, UAV jitter sigma = 40 s, deadline margin 60 s over the
reliable path, bundle of 100 units, normal jitter. Each factor moves alone:

  terminal  near / mid / far      source-destination distance and route depth
  density   40 / 60 / 90 sats     constellation density (Walker 550 km, 53 deg,
                                  10 satellites per plane; scenarios/uc4_d40_cpd,
                                  uc4_cpd, uc4_d90_cpd)
  margin    0 / 60 / 240 s        deadline regime
  sigma     20 / 40 / 60 s        uncertainty level
  size      50 / 100 / 200        traffic volume (bundle size)
  family    normal / t3 / uniform distributional assumption of the UAV start
                                  law (planner and truth share the family)

Policies (copied from run_uc4.py, which stays untouched):

  det         static CGR on the nominal (mean) plan
  scg         static SLA-aware route, argmax P[T <= D] (pick_route)
  cgr_react   reactive CGR: every node repairs with CGR on the nominal plan
              when a contact fails (execute_reactive, no first route)
  scg_policy  SLA-aware with repair: the first route chosen by its simulated
              success under CGR repair (pick_route_under_repair), then
              execute_reactive with that first route
  scg_repair  (extra, cheap) the static SLA-aware route followed with CGR
              repair; separates the value of repair-aware selection

Candidates come from candidate_routes(..., risk_classes=True), as in
run_uc4.py: on this meshed backbone the safe route starts on the slow
terminal->HAPS link and shares its space contacts with the UAV routes, so the
disjoint sweep alone rarely offers it. `--legacy-cands` runs the default
generator instead (risk_classes=False) and writes *_legacy outputs, which is
how the comparison in results/notes/generality.md was produced. The columns
`rel_in_cands` (the UAV-free CGR route that defines the deadline is among the
candidates) and `scg_is_rel` (the SLA-aware router chose it) track this.

Static routes are scored on M_EVAL held-out draws per route and the contingent
policies on M_REAL held-out realizations shared by all of them. Selection uses
its own stream; nothing is scored on draws used to choose it.

Common random numbers: instance i uses the seed (SEED, i) at every factor
level, so the release time and UAV schedule are shared across levels and the
base point is computed once and reused for all six factors.

On the 40-satellite shell the four planes are 90 deg apart in RAAN and never
come within the 5000 km ISL range, so CPD generates no cross-plane ISLs. Some
30-minute windows then hold no nominal route from the terminal to the gateway;
such windows are redrawn (the count is recorded as `redraws`).

Usage: python experiments/run_generality.py [--quick] [--legacy-cands]
                                            [--workers N] [--replot]
Outputs: results/generality.csv (one row per factor level),
         results/generality_instances.csv (one row per instance and point),
         figures/generality.{pdf,png}
         (--quick and --legacy-cands write the same names with a _quick or
         _legacy suffix; --replot redraws the figure from the CSV; --workers
         is capped at 3)
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap
from _style import COL_BASE1, COL_SCG

from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, pick_route
from scgraph.routing.reactive import execute_reactive, pick_route_under_repair
from scgraph.scenarios.uc4_continuum import load_skeleton, make_instance
from scgraph.stats import clopper_pearson
from scgraph.verify.montecarlo import evaluate_route

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261014
M_SEL = 2000          # draws per candidate for static selection
M_POL = 300           # planning realizations for repair-aware selection
M_EVAL = 10_000       # held-out draws per static route
M_REAL = 1000         # held-out realizations for contingent policies
N_CAND_REAL = 40      # sampled realizations feeding the candidate set
R_FULL = 25
UNCERTAIN = {"uav", "isl_x", "feeder", "haps_up"}

SKELETONS = {
    40: ROOT / "scenarios" / "uc4_d40_cpd",
    60: ROOT / "scenarios" / "uc4_cpd",
    90: ROOT / "scenarios" / "uc4_d90_cpd",
}
BASE = {"terminal": "far", "density": 60, "sigma": 40.0, "margin": 60.0,
        "size": 100.0, "family": "normal"}
FACTORS = [
    ("terminal", ["near", "mid", "far"]),
    ("density", [40, 60, 90]),
    ("margin", [0.0, 60.0, 240.0]),
    ("sigma", [20.0, 40.0, 60.0]),
    ("size", [50.0, 100.0, 200.0]),
    ("family", ["normal", "t3", "uniform"]),
]
FACTOR_LABEL = {
    "terminal": "terminal (distance, depth)",
    "density": "constellation size [sats]",
    "margin": "deadline margin [s]",
    "sigma": r"UAV jitter $\sigma$ [s]",
    "size": "bundle size",
    "family": "jitter family",
}
STATIC = ("det", "scg")
CONTINGENT = ("cgr_react", "scg_repair", "scg_policy")
POLICIES = STATIC + CONTINGENT
CFG_KEYS = tuple(BASE)


def point_key(cfg: dict) -> tuple:
    return tuple(cfg[k] for k in CFG_KEYS)


def points() -> list[dict]:
    """The unique configurations: the base plus every non-base level."""
    out = [dict(BASE)]
    for f, levels in FACTORS:
        for v in levels:
            if v != BASE[f]:
                out.append({**BASE, f: v})
    return out


def one_instance(task, _rng):
    cfg, quick, inst = task["cfg"], task["quick"], task["inst"]
    m_sel = 600 if quick else M_SEL
    m_pol = 120 if quick else M_POL
    m_eval = 2000 if quick else M_EVAL
    m_real = 300 if quick else M_REAL
    # common random numbers: the same seed for instance i at every point
    rng = np.random.default_rng([SEED, inst])
    rng_e = np.random.default_rng(rng.integers(2**63))   # held-out stream
    t_inst = time.time()

    sk = load_skeleton(str(SKELETONS[cfg["density"]]))
    redraws = 0
    while True:
        try:
            g, p = make_instance(rng, cfg["sigma"], terminal=cfg["terminal"],
                                 margin=cfg["margin"], skeleton=sk, size=cfg["size"],
                                 jitter_family=cfg["family"])
            break
        except RuntimeError:          # no nominal route in this window
            redraws += 1
            if redraws > 200:
                raise
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    mean = g.mean_plan()
    cands = candidate_routes(g, src, dst, size, t0, rng, N_CAND_REAL,
                             risk_classes=task.get("risk_classes", True))

    routes = {}
    det, _ = cgr_route(mean, src, dst, size, t0)
    routes["det"] = [g.by_id(c.cid) for c in det] if det else None
    routes["scg"], _ = pick_route(g, src, dst, size, t0, D, rng, m=m_sel, cands=cands)
    r_pol, _ = pick_route_under_repair(g, cands, src, dst, size, t0, D, rng, m=m_pol,
                                       nominal=mean)

    # is the reliable path (the UAV-free CGR route that defines the deadline)
    # among the candidates?
    uav = frozenset(c.cid for c in g.contacts if c.kind == "uav")
    rel_det, _ = cgr_route(mean, src, dst, size, t0, excluded_cids=uav)
    rel_in = rel_det is not None and tuple(c.cid for c in rel_det) in {
        tuple(c.cid for c in r) for r in cands}

    out = {**{k: cfg[k] for k in CFG_KEYS}, "inst": inst, "redraws": redraws,
           "D_rel": D - t0, "nominal_hops": p["nominal_hops"],
           "reliable_hops": p["reliable_hops"], "n_cands": len(cands),
           "n_contacts": len(g.contacts)}
    for k in STATIC:
        ev = evaluate_route(routes[k], size, t0, D, m_eval, rng_e)
        out[f"{k}_fails"], out[f"{k}_n"] = ev["fails"], ev["n"]
        out[f"{k}_viol"] = ev["viol"]
        out[f"{k}_lat"] = ev["mean_delivery"] - t0
        r = routes[k] or []
        out[f"{k}_hops"] = len(r)
        out[f"{k}_unc"] = sum(c.kind in UNCERTAIN for c in r)

    reals = g.realize_batch(rng_e, m_real)
    first = {"cgr_react": None,
             "scg_repair": [c.cid for c in routes["scg"]] if routes["scg"] else None,
             "scg_policy": [c.cid for c in r_pol] if r_pol else None}
    for k in CONTINGENT:
        arr = np.array([execute_reactive(real, mean, src, dst, size, t0,
                                         first_route=first[k])[0] for real in reals])
        fails = int(np.count_nonzero(~(arr <= D)))
        out[f"{k}_fails"], out[f"{k}_n"] = fails, m_real
        out[f"{k}_viol"] = fails / m_real
        fin = arr[np.isfinite(arr)]
        out[f"{k}_lat"] = float(np.mean(fin)) - t0 if fin.size else math.nan
    out["scg_policy_hops"] = len(r_pol) if r_pol else 0
    out["rel_in_cands"] = bool(rel_in)
    out["scg_is_rel"] = bool(routes["scg"] is not None and rel_det is not None and
                             [c.cid for c in routes["scg"]] == [c.cid for c in rel_det])
    out["secs"] = time.time() - t_inst
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def t_half(xs) -> float:
    """Half-width of the 95% t-interval on the mean over instances."""
    xs = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if len(xs) < 2:
        return 0.0
    return float(sstats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / math.sqrt(len(xs)))


def paired(sub, a, b, prefix, row):
    """Instance-paired difference viol(a) - viol(b): mean, 95% t half-width,
    and how many instances favour b (positive), favour a, or tie."""
    d = np.array([r[f"{a}_viol"] - r[f"{b}_viol"] for r in sub])
    row[f"{prefix}_mean"] = float(d.mean())
    row[f"{prefix}_ci"] = t_half(d)
    row[f"{prefix}_wins"] = int(np.sum(d > 0))
    row[f"{prefix}_losses"] = int(np.sum(d < 0))
    row[f"{prefix}_ties"] = int(np.sum(d == 0))


def aggregate(results):
    by_point: dict[tuple, list] = {}
    for r in results:
        by_point.setdefault(point_key(r), []).append(r)
    rows = []
    for f, levels in FACTORS:
        for v in levels:
            cfg = {**BASE, f: v}
            sub = by_point.get(point_key(cfg), [])
            if not sub:
                continue
            row = {"factor": f, "level": v, "is_base": v == BASE[f], "R": len(sub),
                   "redraws": int(sum(r["redraws"] for r in sub)),
                   "D_rel": float(np.mean([r["D_rel"] for r in sub])),
                   "nominal_hops": float(np.mean([r["nominal_hops"] for r in sub])),
                   "reliable_hops": float(np.mean([r["reliable_hops"] for r in sub])),
                   "n_cands": float(np.mean([r["n_cands"] for r in sub])),
                   "n_contacts": float(np.mean([r["n_contacts"] for r in sub]))}
            for k in POLICIES:
                v_i = [r[f"{k}_viol"] for r in sub]
                fails = int(sum(r[f"{k}_fails"] for r in sub))
                n = int(sum(r[f"{k}_n"] for r in sub))
                lo, hi = clopper_pearson(fails, n)
                row[f"{k}_viol"] = fails / n          # = mean of per-instance rates
                row[f"{k}_cp_lo"], row[f"{k}_cp_hi"] = lo, hi
                row[f"{k}_inst_ci"] = t_half(v_i)
                row[f"{k}_fails"], row[f"{k}_n"] = fails, n
                lat = [r[f"{k}_lat"] for r in sub if np.isfinite(r[f"{k}_lat"])]
                row[f"{k}_lat"] = float(np.mean(lat)) if lat else math.nan
            for k in STATIC:
                row[f"{k}_hops"] = float(np.mean([r[f"{k}_hops"] for r in sub]))
                row[f"{k}_unc"] = float(np.mean([r[f"{k}_unc"] for r in sub]))
            row["scg_policy_hops"] = float(np.mean([r["scg_policy_hops"] for r in sub]))
            row["rel_in_cands"] = float(np.mean([r["rel_in_cands"] for r in sub]))
            row["scg_is_rel"] = float(np.mean([r["scg_is_rel"] for r in sub]))
            # positive = the SLA-aware policy violates less
            paired(sub, "det", "scg", "d_static", row)
            paired(sub, "cgr_react", "scg_policy", "d_repair", row)
            paired(sub, "det", "scg_policy", "d_best", row)
            # does planning beat repair? static SLA-aware against reactive CGR
            paired(sub, "cgr_react", "scg", "d_plan_vs_repair", row)
            rows.append(row)
    return rows


def write(path, rows):
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("factor", "level", "is_base"):
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
        r["is_base"] = r["is_base"] == "True"
    return rows


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

# hue = router (CGR red, SLA-aware blue); marker fill = execution model
# (hollow = static source route, filled = hop-by-hop repair)
PSTY = {
    "det": dict(color=COL_BASE1, marker="s", mfc="white", label="CGR, static"),
    "scg": dict(color=COL_SCG, marker="o", mfc="white", label="SLA-aware, static"),
    "cgr_react": dict(color=COL_BASE1, marker="s", label="CGR + repair"),
    "scg_policy": dict(color=COL_SCG, marker="o", label="SLA-aware + repair"),
}
PLOT = ("det", "scg", "cgr_react", "scg_policy")


def level_label(f, v):
    if f in ("margin", "sigma", "size"):
        s = f"{float(v):g}"
    elif f == "density":
        s = f"{int(float(v))}"
    else:
        s = str(v)
    return s


def plot(rows, stem="generality"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex=True)
    offs = np.linspace(0.27, -0.27, len(PLOT))
    xmax = max(r[f"{k}_viol"] + r[f"{k}_inst_ci"] for r in rows for k in PLOT)
    for ax, (f, levels) in zip(axes.flat, FACTORS):
        sub = [r for r in rows if r["factor"] == f]
        ys = np.arange(len(sub))[::-1]
        for k, dy in zip(PLOT, offs):
            v = np.array([r[f"{k}_viol"] for r in sub])
            h = np.array([r[f"{k}_inst_ci"] for r in sub])
            xerr = np.vstack([np.minimum(h, v), np.minimum(h, 1.0 - v)])
            ax.errorbar(v, ys + dy, xerr=xerr, linestyle="none",
                        markersize=4.2, markeredgewidth=1.1, elinewidth=0.9,
                        capsize=0, **PSTY[k])
        ax.set_yticks(ys)
        ax.set_yticklabels([level_label(f, r["level"]) + ("*" if r["is_base"] else "")
                            for r in sub])
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(FACTOR_LABEL[f], fontsize=7.5)
        ax.grid(axis="y", visible=False)
        for y in ys[:-1]:
            ax.axhline(y - 0.5, color="#dddddd", linewidth=0.5)
        ax.set_xlim(0, min(1.0, 1.08 * xmax))
    for ax in axes[1]:
        ax.set_xlabel("SLA-violation rate (held out)")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    fig.savefig(str(ROOT / "figures" / f"{stem}.pdf"))
    fig.savefig(str(ROOT / "figures" / f"{stem}.png"))


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    workers = None
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    workers = min(3, workers) if workers else 3
    t_start = time.time()
    _style.setup()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    legacy = "--legacy-cands" in sys.argv
    stem = "generality" + ("_quick" if quick else "") + ("_legacy" if legacy else "")
    out_csv = ROOT / "results" / f"{stem}.csv"
    inst_csv = ROOT / "results" / f"{stem}_instances.csv"

    if replot:
        rows = load_rows(out_csv)
    else:
        R = 3 if quick else R_FULL
        pts = points()
        tasks = [{"cfg": cfg, "quick": quick, "inst": i, "risk_classes": not legacy}
                 for cfg in pts for i in range(R)]
        print(f"{len(pts)} points x {R} instances = {len(tasks)} tasks, "
              f"{workers} workers", flush=True)
        res = pmap(one_instance, tasks, seed=SEED, workers=workers)
        write(inst_csv, res)
        rows = aggregate(res)
        write(out_csv, rows)
        cpu = sum(r["secs"] for r in res)
        print(f"instance CPU time {cpu:.0f} s (mean {cpu / len(res):.1f} s, "
              f"max {max(r['secs'] for r in res):.1f} s)")

    for r in rows:
        print(f"{r['factor']:>8}={level_label(r['factor'], r['level']):>7}{'*' if r['is_base'] else ' '}"
              f" R={int(r['R'])} | " + "  ".join(
                  f"{k}={r[f'{k}_viol']:.3f}" for k in POLICIES)
              + f" | hops det/scg {r['det_hops']:.1f}/{r['scg_hops']:.1f}"
              + f" | d_static {r['d_static_mean']:+.3f}+-{r['d_static_ci']:.3f}"
              + f" d_repair {r['d_repair_mean']:+.3f}+-{r['d_repair_ci']:.3f}"
              + (f" | rel offered {r['rel_in_cands']:.2f} chosen {r['scg_is_rel']:.2f}"
                 if "rel_in_cands" in r else ""))
    plot(rows, stem)
    print(f"done in {time.time() - t_start:.1f}s -> {out_csv.relative_to(ROOT)}, "
          f"figures/{stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
