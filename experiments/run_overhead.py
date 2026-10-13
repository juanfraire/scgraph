"""Planning-plane and forwarding-plane cost per routing decision (revision T8, R1.d).

Reviewer 1 read the runtime figure as "distributional scoring is about 1000x
slower than CGR, which a SWaP-constrained processor cannot afford". That ratio
compares a complete offline SLA-aware selection with ONE deterministic route
search. A SABR router does not stop at one search either: it builds a table of
alternative routes. This runner puts the five planning-plane costs next to each
other on the same CPD plans and the same UC1-style instances as
run_runtime.py, and adds the cost that actually runs onboard:

  (i)   det      one deterministic cgr_route search on the mean plan
  (ii)  sabr     SABR-style route-table construction: the deterministic part of
                 candidate_routes (mean-plan route, single-contact exclusions,
                 disjoint suppression sweep), with no realisations
  (iii) cand     full candidate generation, with the 40 realisations
  (iv)  score    per-route scoring under the laws at M = 2000 draws
  (v)   select   the full SLA-aware selection, pick_route end to end

  lean  (iii) and (v) again, with realisations that redraw only the contacts
        that carry a law (the point-mass backbone is copied as it stands).
        The shipped candidate_routes re-samples all 14,640 Dirac contacts of the
        largest plan per realisation, which is a Python implementation choice
        and dominates (iii). The lean variant measures what the same algorithm
        costs once that choice is removed; the candidate sets are statistically
        equivalent (they consume a different random stream).

  fwd   the forwarding plane. A ProxyPlan (Sec. VI, ranked table per node and
        arrival bucket) is built offline on the plan restricted to the bundle's
        lifetime [t0, D] (graph.window), then executed against realisations of
        the truth. We time one online table resolution, chooser()(node, a,
        avail, used), in microseconds, and report the table's size: entries,
        and bytes in two compact binary encodings (whole ranked routes, and the
        first-hop contact ids that the chooser actually consults).

All timings are medians over repeats, then medians over release times. They are
wall-clock measurements of the machine, sensitive to CPU contention from other
jobs, so run them single-process on an idle machine. `--replot` redraws the
figure from the committed CSV without re-measuring.

Usage: python experiments/run_overhead.py [--quick] [--replot]
Outputs: results/overhead_compute.csv, figures/overhead_compute.{pdf,png}
"""

from __future__ import annotations

import csv
import math
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _style import COL_BASE1, COL_BASE2, COL_BOUND, COL_SCG

from scgraph.graph import Realization
from scgraph.io import load_cpd_plan
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.placement import execute, plan_proxy
from scgraph.routing.quantile import candidate_routes, pick_route, route_stats
from scgraph.scenarios.uc1_event import make_instance_cpd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260728          # same instances as run_runtime.py
SIGMA = 40.0
SIZES = [6, 12, 24, 48]
M_SCORE = 2000
N_REAL = 40


# ---------------------------------------------------------------------------
# the SABR-style part of candidate generation, on a prebuilt plan
# ---------------------------------------------------------------------------


def sabr_routes(mean_plan: Realization, src, dst, size, t0) -> list[tuple[int, ...]]:
    """Deterministic part of quantile.candidate_routes, on a prebuilt mean plan.

    Same three steps in the same order (mean-plan route, one exclusion per
    contact of that route, disjoint suppression sweep), returning route keys.
    The plan is passed in already built, as it is for the single search (i), so
    (i) and (ii) differ only in the number of searches.
    """
    seen: dict[tuple[int, ...], None] = {}

    def add(route):
        if route is not None:
            seen.setdefault(tuple(c.cid for c in route), None)

    mean_route, _ = cgr_route(mean_plan, src, dst, size, t0)
    add(mean_route)
    if mean_route is not None:
        for c in mean_route:
            alt, _ = cgr_route(mean_plan, src, dst, size, t0,
                               excluded_cids=frozenset({c.cid}))
            add(alt)
    suppressed = set(c.cid for c in mean_route) if mean_route else set()
    while True:
        alt, _ = cgr_route(mean_plan, src, dst, size, t0,
                           excluded_cids=frozenset(suppressed))
        if alt is None:
            break
        add(alt)
        suppressed.update(c.cid for c in alt)
    return list(seen)


def lean_candidates(g, src, dst, size, t0, rng, n_realizations=N_REAL):
    """candidate_routes with realisations that redraw only the stochastic contacts.

    Valid when the graph has no latent state and no common factor (true for
    UC1): a point-mass contact then realises to the same DetContact every time,
    so copying it is exact. Everything is rebuilt inside the call, the mean
    plan included, so the timing is comparable with candidate_routes.
    """
    assert g.latent is None
    fixed, stoch = [], []
    for c in g.contacts:
        (fixed if c.is_point_mass else stoch).append(c)
    fixed_det = [c.mean_det() for c in fixed]
    mean_plan = Realization(g.num_nodes, fixed_det + [c.mean_det() for c in stoch
                                                       if c.p_exist >= 0.5])
    keys = sabr_routes(mean_plan, src, dst, size, t0)
    seen = dict.fromkeys(keys)
    for _ in range(n_realizations):
        det = list(fixed_det)
        for c in stoch:
            d = c.sample(rng)
            if d is not None:
                det.append(d)
        r, _ = cgr_route(Realization(g.num_nodes, det), src, dst, size, t0)
        if r is not None:
            seen.setdefault(tuple(c.cid for c in r), None)
    return [[g.by_id(cid) for cid in key] for key in seen]


# ---------------------------------------------------------------------------
# the forwarding table: size on the wire
# ---------------------------------------------------------------------------

# header: size, t0, deadline as float32, n_buckets as u32
_HDR = struct.calcsize("<fffI")


def table_bytes(plan) -> tuple[int, int, int, int, int]:
    """(keys, ranked routes, stored contact ids, bytes full, bytes first-hop).

    full       per key: u16 node, u8 bucket, u8 count; per route: u8 length
               followed by u32 contact ids.
    first-hop  per key: u16 node, u8 bucket, u8 count; per route: one u32,
               the first contact id, which is all ProxyPlan.chooser reads.
    Both are packed with struct so the count is the real encoded length.
    """
    full = bytearray(struct.pack("<fffI", plan.size, plan.t0, plan.deadline,
                                 plan.n_buckets))
    first = bytearray(full)
    n_routes = n_cids = 0
    for (node, k), routes in sorted(plan.table.items()):
        full += struct.pack("<HBB", node, k, len(routes))
        first += struct.pack("<HBB", node, k, len(routes))
        for cids in routes:
            full += struct.pack(f"<B{len(cids)}I", len(cids), *cids)
            first += struct.pack("<I", cids[0])
            n_routes += 1
            n_cids += len(cids)
    return len(plan.table), n_routes, n_cids, len(full), len(first)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def _timed(fn, repeats: int) -> float:
    """Median wall time of `fn` over `repeats` runs, in milliseconds."""
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def forwarding(g, p, rng, n_exec: int, reps: int) -> dict:
    """Build a ProxyPlan offline, then time its online resolution."""
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    w = g.window(t0, D)
    t = time.perf_counter()
    plan = plan_proxy(w, src, dst, size, t0, D, rng)
    t_build = (time.perf_counter() - t) * 1e3

    choose = plan.chooser()
    calls = []

    def recording(node, a, avail, used):
        calls.append((node, a, avail, frozenset(used)))
        return choose(node, a, avail, used)

    delivered = 0
    for _ in range(n_exec):
        real = w.realize(rng)
        if execute(real, src, dst, size, t0, recording) <= D:
            delivered += 1

    # replay the recorded decisions; per-call time includes the Python call
    per_call = []
    for _ in range(reps):
        t = time.perf_counter_ns()
        for node, a, avail, used in calls:
            choose(node, a, avail, used)
        per_call.append((time.perf_counter_ns() - t) / max(1, len(calls)) / 1e3)

    keys, n_routes, n_cids, b_full, b_first = table_bytes(plan)
    return {
        "window_contacts": len(w.contacts),
        "t_proxy_build_ms": t_build,
        "t_resolve_us": statistics.median(per_call),
        "resolve_calls": len(calls),
        "proxy_keys": keys,
        "proxy_routes": n_routes,
        "proxy_cids": n_cids,
        "proxy_bytes_full": b_full,
        "proxy_bytes_first_hop": b_first,
        "proxy_pdr": delivered / n_exec,
    }


def run(quick: bool = False):
    repeats = 3 if quick else 7
    n_inst = 3 if quick else 8
    n_exec = 50 if quick else 200
    fwd_reps = 5 if quick else 20

    rows = []
    for n in SIZES:
        path = ROOT / "scenarios" / f"scale_{n}"
        if not path.exists():
            print(f"  (skipping {n}: no skeleton at {path})")
            continue
        sk = load_cpd_plan(path)
        t_size = time.time()

        per: dict[str, list] = {}

        def put(k, v):
            per.setdefault(k, []).append(v)

        for i in range(n_inst):
            rng = np.random.default_rng(SEED + 1000 * n + i)
            g, p = make_instance_cpd(rng, SIGMA, skeleton=sk)
            src, dst = p["src"], p["dst"]
            size, t0, D = p["size"], p["t0"], p["deadline"]

            t_mp = _timed(lambda: g.mean_plan(), repeats)
            mean_plan = g.mean_plan()

            # the replica must reproduce the deterministic part of the library
            keys = sabr_routes(mean_plan, src, dst, size, t0)
            ref = candidate_routes(g, src, dst, size, t0, rng, n_realizations=0)
            assert keys == [tuple(c.cid for c in r) for r in ref], "SABR replica drifted"

            t_det = _timed(lambda: cgr_route(mean_plan, src, dst, size, t0), repeats)
            t_sabr = _timed(lambda: sabr_routes(mean_plan, src, dst, size, t0), repeats)

            cands = candidate_routes(g, src, dst, size, t0, rng, n_realizations=N_REAL)
            t_cand = _timed(
                lambda: candidate_routes(g, src, dst, size, t0, rng,
                                         n_realizations=N_REAL),
                repeats,
            )
            lean = lean_candidates(g, src, dst, size, t0, rng)
            t_cand_lean = _timed(
                lambda: lean_candidates(g, src, dst, size, t0, rng), repeats
            )
            route = cands[0]
            t_score = _timed(
                lambda: route_stats(route, size, t0, D, M_SCORE, rng, latent=g.latent),
                repeats,
            )
            t_sel = _timed(
                lambda: pick_route(g, src, dst, size, t0, D, rng, m=M_SCORE,
                                   n_realizations=N_REAL),
                repeats,
            )
            t_sel_lean = _timed(
                lambda: pick_route(g, src, dst, size, t0, D, rng, m=M_SCORE,
                                   cands=lean_candidates(g, src, dst, size, t0, rng)),
                repeats,
            )
            fw = forwarding(g, p, rng, n_exec, fwd_reps)

            put("k_sabr", len(keys))
            put("cands", len(cands))
            put("cands_lean", len(lean))
            put("t_mean_plan_ms", t_mp)
            put("t_det_ms", t_det)
            put("t_sabr_ms", t_sabr)
            put("t_cand_ms", t_cand)
            put("t_cand_lean_ms", t_cand_lean)
            put("t_score_per_route_ms", t_score)
            put("t_select_ms", t_sel)
            put("t_select_lean_ms", t_sel_lean)
            for k, v in fw.items():
                put(k, v)

        med = {k: statistics.median(v) for k, v in per.items()}
        row = {
            "sats": n,
            "contacts": len(sk.contacts),
            "nodes": sk.num_nodes,
            "k_sabr": med["k_sabr"],
            "candidates": med["cands"],
            "candidates_lean": med["cands_lean"],
            "t_det_ms": med["t_det_ms"],
            "t_sabr_ms": med["t_sabr_ms"],
            "t_mean_plan_ms": med["t_mean_plan_ms"],
            "t_cand_ms": med["t_cand_ms"],
            "t_cand_lean_ms": med["t_cand_lean_ms"],
            "t_score_per_route_ms": med["t_score_per_route_ms"],
            "t_select_ms": med["t_select_ms"],
            "t_select_lean_ms": med["t_select_lean_ms"],
            "ratio_select_vs_sabr": med["t_select_ms"] / med["t_sabr_ms"],
            "ratio_select_vs_det": med["t_select_ms"] / med["t_det_ms"],
            "ratio_sabr_vs_det": med["t_sabr_ms"] / med["t_det_ms"],
            "ratio_select_lean_vs_sabr": med["t_select_lean_ms"] / med["t_sabr_ms"],
            "ratio_select_lean_vs_det": med["t_select_lean_ms"] / med["t_det_ms"],
            "window_contacts": med["window_contacts"],
            "t_proxy_build_ms": med["t_proxy_build_ms"],
            "t_resolve_us": med["t_resolve_us"],
            "resolve_calls": med["resolve_calls"],
            "proxy_keys": med["proxy_keys"],
            "proxy_routes": med["proxy_routes"],
            "proxy_cids": med["proxy_cids"],
            "proxy_bytes_full": med["proxy_bytes_full"],
            "proxy_bytes_first_hop": med["proxy_bytes_first_hop"],
            "proxy_pdr": med["proxy_pdr"],
            "M": M_SCORE,
            "n_real": N_REAL,
            "instances": n_inst,
            "repeats": repeats,
        }
        rows.append(row)
        print(
            f"  sats={n:3d} contacts={row['contacts']:6d}  det={row['t_det_ms']:.3f}ms"
            f"  sabr={row['t_sabr_ms']:.2f}ms (k={row['k_sabr']:.0f})"
            f"  cand={row['t_cand_ms']:.1f}ms  lean={row['t_cand_lean_ms']:.1f}ms"
            f"  score={row['t_score_per_route_ms']:.3f}ms"
            f"  select={row['t_select_ms']:.1f}ms (lean {row['t_select_lean_ms']:.1f})"
            f"  v/ii={row['ratio_select_vs_sabr']:.1f} v/i={row['ratio_select_vs_det']:.0f}"
            f"  resolve={row['t_resolve_us']:.2f}us"
            f"  table={row['proxy_bytes_first_hop']:.0f}/{row['proxy_bytes_full']:.0f}B"
            f"  [{time.time() - t_size:.0f}s]"
        )
    return rows


def load():
    """Re-read a finished run, so the figure can be redrawn without re-measuring."""
    with open(ROOT / "results" / "overhead_compute.csv") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def plot(rows):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    x = [r["contacts"] for r in rows]
    series = [
        ("t_select_ms", "(v) full SLA-aware selection",
         dict(color=COL_SCG, linestyle="-", marker="o")),
        ("t_cand_ms", "(iii) candidate generation",
         dict(color="#7b4fa8", linestyle="-.", marker="^")),
        ("t_select_lean_ms", "(v) selection, lean realisations",
         dict(color=COL_SCG, linestyle=":", marker="o", markerfacecolor="white")),
        ("t_sabr_ms", "(ii) SABR-style route table",
         dict(color=COL_BASE2, linestyle="-.", marker="v")),
        ("t_score_per_route_ms", "(iv) scoring, per route",
         dict(color=COL_BOUND, linestyle=":", marker="d")),
        ("t_det_ms", "(i) one CGR search",
         dict(color=COL_BASE1, linestyle="--", marker="s")),
    ]
    for key, label, st in series:
        ax.plot(x, [r[key] for r in rows], label=label, **st)
    ax.plot(x, [r["t_resolve_us"] / 1e3 for r in rows],
            label="onboard table resolution",
            color="#222222", linestyle="--", marker="x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("contacts in the plan")
    ax.set_ylabel("wall time per decision [ms]")
    ax.set_ylim(bottom=1e-4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=5.8,
              ncol=2, borderpad=0.3, labelspacing=0.3, handlelength=2.6,
              columnspacing=1.0)
    _style.save(fig, str(ROOT / "figures" / "overhead_compute"))


def main():
    quick = "--quick" in sys.argv
    replot = "--replot" in sys.argv
    t_start = time.time()
    _style.setup()

    if replot:
        rows = load()
        print(f"replotting {len(rows)} rows from results/overhead_compute.csv")
    else:
        print("Planning- and forwarding-plane cost per decision (CPD skeletons):")
        rows = run(quick)
    if not rows:
        print("no skeletons found; generate them first")
        return

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    if not replot:
        with open(ROOT / "results" / "overhead_compute.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    plot(rows)
    print(f"done in {time.time() - t_start:.1f}s -> results/overhead_compute.csv, "
          "figures/overhead_compute.pdf")


if __name__ == "__main__":
    main()
