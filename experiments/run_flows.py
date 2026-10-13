"""Concurrent bundles, finite storage, residual volume: UC1 under offered load (T8 / D3).

The paper's router places ONE atomic bundle with unbounded storage. Here K
bundles of the same size are burst-released from the UC1 event site at the
instance's release time t0, all with the instance's deadline D, at timing
jitter sigma = 40 s. Each policy commits a route per bundle, and the
multi-bundle executor (`scgraph.flows.run_flows`) plays the committed routes
on held-out realizations: contacts serve one bundle at a time in FIFO order,
transfers are atomic and cut at the window end, and every node other than the
source and destination stores at most B bundles.

Policies (routes chosen on the selection stream only):
  P1  deterministic CGR on the mean plan, each bundle routed alone, so all K
      bundles share the earliest-arrival route (contention-blind);
  P2  CGR with SABR-style volume booking on the mean plan: bundles are routed
      in turn and each books its volume, so later bundles see the residual
      volume and the backlog (`flows.cgr_route_booked`);
  P3  SLA-aware, contention-blind: `pick_route` (argmax P[T <= D], M = 2000)
      for a single bundle. The K bundles are identical, so routing each alone
      returns the same route; it is computed once and given to all of them;
  P4  SLA-aware, contention-aware: bundles are committed in turn, and bundle j
      takes the candidate maximising its own on-time frequency over M_SEL
      Monte-Carlo plans, simulated with the executor and bundles 1..j-1
      already committed (`flows.assign_contention_aware`), at the storage B
      under test.
  P4i (buffer sweep only) the B = inf assignment of P4 played at finite B:
      contention-aware but storage-blind, to separate the two effects.
  P2f, P4f (load sweep only) proactive fragmentation: every bundle is split
      into FRAG equal fragments routed as independent bundles by P2 / P4; a
      bundle is on time iff all its fragments are.

Every policy is evaluated on the SAME N_EVAL held-out realizations per
instance, drawn from a stream the policies never saw. Sweeps: offered load
K in KS at B = inf, and storage B in BS at K in K_BUF (K = 16 is plotted). The
sequential policies (P2, P4) have the prefix property, so the K = 16
assignment restricted to its first K bundles is the K-bundle assignment, and
one assignment per B serves the whole load sweep.

Reported per (sweep, K, B, policy): fraction of bundles on time with the
exact two-sided 95% Clopper-Pearson interval on the pooled count (it treats
bundles as independent, so it understates the spread across instances), the
mean and t-interval of the per-instance on-time fraction (the error bars in
the figure), the probability that the whole burst is on time, the mean
delivery time of delivered bundles, the loss breakdown (late, window, buffer,
missing, noroute), the mean peak storage occupancy at the busiest
intermediate node, and the share of bundles routed through an air relay.

Usage: python experiments/run_flows.py [--quick] [--workers N]
Outputs: results/flows_load.csv, results/flows_buffer.csv,
         figures/flows.{pdf,png}
"""

from __future__ import annotations

import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import _style
from _parallel import pmap
from _style import STYLE

from scgraph.flows import (
    Bundle, assign_contention_aware, book_routes, contact_map, run_flows,
)
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes, pick_route
from scgraph.scenarios.uc1_event import make_instance_cpd
from scgraph.stats import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013
SIGMA = 40.0
KS = (1, 2, 4, 8, 16)
BS = (math.inf, 8, 4, 2)
K_BUF = (16, 8)
FRAG = 2
M_PICK = 2000              # P3's per-route Monte Carlo, as in run_uc1.py
OUTCOMES = ("ontime", "late", "window", "buffer", "missing", "noroute")

LABELS = {
    "P1": "CGR, mean plan (contention-blind)",
    "P2": "CGR + volume booking (SABR-style)",
    "P3": "SLA-aware, contention-blind",
    "P4": "SLA-aware, contention-aware",
    "P4i": "SLA-aware, contention-aware, storage-blind",
    "P2f": f"P2 with {FRAG}-way proactive fragmentation",
    "P4f": f"P4 with {FRAG}-way proactive fragmentation",
}


def config(quick: bool) -> dict:
    if quick:
        return {"R": 6, "M_SEL": 100, "N_EVAL": 150}
    return {"R": 30, "M_SEL": 500, "N_EVAL": 1000}


def _eval(plans, bundles, deadline, buffer, unbounded, parents=None, n_parents=None):
    """Outcome counts of one committed bundle set over the held-out plans."""
    cnt = Counter()
    lat_sum, lat_n, all_ok, peak_sum = 0.0, 0, 0, 0.0
    t0 = bundles[0].release
    for pl in plans:
        res = run_flows(pl, bundles, buffer, unbounded)
        st = res.status([deadline] * len(bundles))
        dl = res.delivery
        if parents is not None:            # a bundle is its fragments' worst
            worst = ["ontime"] * n_parents
            last = np.full(n_parents, -math.inf)
            rank = {s: i for i, s in enumerate(OUTCOMES)}
            for i, s in enumerate(st):
                q = parents[i]
                if rank[s] > rank[worst[q]]:
                    worst[q] = s
                last[q] = max(last[q], dl[i])
            st, dl = worst, last
        cnt.update(st)
        for s, d in zip(st, dl):
            if s in ("ontime", "late"):
                lat_sum += d - t0
                lat_n += 1
        all_ok += all(s == "ontime" for s in st)
        peak_sum += max(res.peak_occupancy.values(), default=0)
    return {"cnt": cnt, "lat_sum": lat_sum, "lat_n": lat_n, "all_ok": all_ok,
            "peak_sum": peak_sum, "n_plans": len(plans)}


def one_instance(task, rng):
    """Build one UC1 instance, choose every policy's routes, score them held out."""
    cfg = task["cfg"]
    r_inst, r_sel, r_eval = rng.spawn(3)
    g, p = make_instance_cpd(r_inst, sigma=SIGMA)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    g = g.window(t0, g.horizon)            # contacts that end before t0 are unusable
    kmax = max(max(KS), max(K_BUF))
    tmpl = [Bundle(src, dst, size, t0, D) for _ in range(kmax)]
    frag_tmpl = [Bundle(src, dst, size / FRAG, t0, D) for _ in range(kmax * FRAG)]
    unbounded = {src, dst}
    air = {c.cid for c in g.contacts if c.layer == "air"}

    # ---------------- route selection (selection stream r_sel only) ----------
    tic = time.perf_counter()
    mean_plan = g.mean_plan()
    r1, _ = cgr_route(mean_plan, src, dst, size, t0)
    routes = {"P1": [tuple(c.cid for c in r1) if r1 else ()] * kmax}
    routes["P2"], _ = book_routes(mean_plan, tmpl)
    cands = candidate_routes(g, src, dst, size, t0, r_sel)
    r3, _ = pick_route(g, src, dst, size, t0, D, r_sel, m=M_PICK, cands=cands)
    routes["P3"] = [tuple(c.cid for c in r3) if r3 else ()] * kmax
    cand_ids = [tuple(c.cid for c in r) for r in cands]
    sel = [contact_map(g.realize(r_sel)) for _ in range(cfg["M_SEL"])]
    p4 = {}
    for B in BS:
        p4[B], _ = assign_contention_aware(cand_ids, tmpl, sel, B, unbounded)
    routes["P4"] = p4[math.inf]
    routes["P2f"], _ = book_routes(mean_plan, frag_tmpl)
    routes["P4f"], _ = assign_contention_aware(cand_ids, frag_tmpl, sel, math.inf,
                                               unbounded)
    t_select = time.perf_counter() - tic

    # ---------------- held-out evaluation (stream r_eval only) ---------------
    tic = time.perf_counter()
    ev = [contact_map(g.realize(r_eval)) for _ in range(cfg["N_EVAL"])]

    def bundles(pol, k, route_list=None):
        rl = route_list if route_list is not None else routes[pol]
        if pol in ("P2f", "P4f"):
            return [b.on(r) for b, r in zip(frag_tmpl[: k * FRAG], rl[: k * FRAG])]
        return [b.on(r) for b, r in zip(tmpl[:k], rl[:k])]

    def air_share(bs):
        return sum(any(c in air for c in b.route) for b in bs) / len(bs)

    out = {}
    for k in KS:
        for pol in ("P1", "P2", "P3", "P4", "P2f", "P4f"):
            bs = bundles(pol, k)
            frag = pol.endswith("f")
            parents = [i // FRAG for i in range(len(bs))] if frag else None
            r = _eval(ev, bs, D, math.inf, unbounded, parents, k if frag else None)
            r["air"] = air_share(bs)
            out[("load", k, math.inf, pol)] = r
    for k in K_BUF:
        for B in BS:
            for pol in ("P1", "P2", "P3", "P4", "P4i"):
                if pol == "P4":
                    bs = bundles(pol, k, p4[B])
                elif pol == "P4i":
                    bs = bundles("P4", k, p4[math.inf])
                else:
                    bs = bundles(pol, k)
                r = _eval(ev, bs, D, B, unbounded)
                r["air"] = air_share(bs)
                out[("buffer", k, B, pol)] = r
    t_eval = time.perf_counter() - tic

    info = {"t0": t0, "deadline": D, "pass_wait": p["pass_wait"],
            "n_cands": len(cands), "p3_air": any(c.layer == "air" for c in r3),
            "t_select": t_select, "t_eval": t_eval}
    return {"out": out, "info": info}


def aggregate(results, sweep: str) -> list[dict]:
    keys = sorted({k for r in results for k in r["out"] if k[0] == sweep},
                  key=lambda k: (k[1], -k[2] if math.isfinite(k[2]) else -1e9, k[3]))
    rows = []
    R = len(results)
    tq = float(sps.t.ppf(0.975, R - 1)) if R > 1 else 0.0
    for key in keys:
        _, k, B, pol = key
        recs = [r["out"][key] for r in results]
        units = k                        # bundles per plan (fragments regrouped)
        cnt = Counter()
        for x in recs:
            cnt.update(x["cnt"])
        n = sum(x["n_plans"] for x in recs) * units
        n_plans = sum(x["n_plans"] for x in recs)
        on = cnt["ontime"]
        lo, hi = clopper_pearson(on, n)
        inst = np.array([x["cnt"]["ontime"] / (x["n_plans"] * units) for x in recs])
        all_ok = sum(x["all_ok"] for x in recs)
        a_lo, a_hi = clopper_pearson(all_ok, n_plans)
        lat_n = sum(x["lat_n"] for x in recs)
        row = {
            "K": k, "B": "inf" if not math.isfinite(B) else int(B), "policy": pol,
            "label": LABELS[pol], "R": R, "n_bundles": n,
            "ontime": on / n, "ontime_cp_lo": lo, "ontime_cp_hi": hi,
            "ontime_inst_mean": float(inst.mean()),
            "ontime_inst_ci": tq * float(inst.std(ddof=1)) / math.sqrt(R) if R > 1 else 0.0,
            "ontime_inst_min": float(inst.min()),
            "p_all_ontime": all_ok / n_plans, "p_all_cp_lo": a_lo, "p_all_cp_hi": a_hi,
            "mean_delivery": (sum(x["lat_sum"] for x in recs) / lat_n) if lat_n else math.nan,
        }
        for o in OUTCOMES[1:]:
            row[f"frac_{o}"] = cnt[o] / n
        row["peak_storage"] = sum(x["peak_sum"] for x in recs) / n_plans
        row["air_share"] = float(np.mean([x["air"] for x in recs]))
        rows.append(row)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def plot(load_rows, buf_rows, stem):
    import matplotlib.pyplot as plt

    series = {
        "P1": dict(STYLE["base1"], label="CGR (mean plan)"),
        "P2": dict(STYLE["base2"], label="CGR + volume booking"),
        "P3": dict(STYLE["scg"], linestyle=":", marker="o", markerfacecolor="white",
                   label="SLA-aware, contention-blind"),
        "P4": dict(STYLE["scg"], label="SLA-aware, contention-aware"),
    }
    # small horizontal dodge so that error bars at the same x stay readable
    dodge = dict(zip(series, (-1.5, -0.5, 0.5, 1.5)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for pol, st in series.items():
        rs = [r for r in load_rows if r["policy"] == pol]
        ax1.errorbar([r["K"] * 2 ** (0.035 * dodge[pol]) for r in rs],
                     [r["ontime_inst_mean"] for r in rs],
                     yerr=[r["ontime_inst_ci"] for r in rs], capsize=2, **st)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(KS)
    ax1.set_xticklabels([str(k) for k in KS])
    ax1.set_xlabel("bundles released at once, $K$  ($B=\\infty$)")
    ax1.set_ylabel("fraction of bundles on time")
    ax1.set_ylim(0, 1.03)
    ax1.legend(loc="lower left")

    bx = list(range(len(BS)))
    for pol, st in series.items():
        rs = [r for r in buf_rows if r["policy"] == pol and r["K"] == max(K_BUF)]
        order = {("inf" if not math.isfinite(B) else int(B)): i for i, B in enumerate(BS)}
        rs = sorted(rs, key=lambda r: order[r["B"]])
        ax2.errorbar([x + 0.035 * dodge[pol] for x in bx],
                     [r["ontime_inst_mean"] for r in rs],
                     yerr=[r["ontime_inst_ci"] for r in rs], capsize=2, **st)
    ax2.set_xticks(bx)
    ax2.set_xticklabels(["$\\infty$" if not math.isfinite(B) else str(int(B)) for B in BS])
    ax2.set_xlabel(f"storage per relay node $B$ [bundles]  ($K={max(K_BUF)}$)")
    ax2.set_ylabel("fraction of bundles on time")
    ax2.set_ylim(0, 1.03)
    _style.save(fig, stem)


def main():
    quick = "--quick" in sys.argv
    workers = None
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    cfg = config(quick)
    t_start = time.time()
    _style.setup()
    tasks = [{"i": i, "cfg": cfg} for i in range(cfg["R"])]
    results = pmap(one_instance, tasks, seed=SEED, workers=workers)

    load_rows = aggregate(results, "load")
    buf_rows = aggregate(results, "buffer")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    write_csv(load_rows, ROOT / "results" / "flows_load.csv")
    write_csv(buf_rows, ROOT / "results" / "flows_buffer.csv")
    plot(load_rows, buf_rows, str(ROOT / "figures" / "flows"))

    def show(rows, title):
        print(title)
        for r in rows:
            print(f"  K={r['K']:>2} B={r['B']!s:>3} {r['policy']:<4} on-time "
                  f"{r['ontime']:.3f} [{r['ontime_cp_lo']:.3f},{r['ontime_cp_hi']:.3f}]"
                  f" inst {r['ontime_inst_mean']:.3f}+-{r['ontime_inst_ci']:.3f}"
                  f"  all {r['p_all_ontime']:.3f}  late {r['frac_late']:.3f}"
                  f" win {r['frac_window']:.3f} buf {r['frac_buffer']:.3f}"
                  f"  lat {r['mean_delivery']:.0f}  peak {r['peak_storage']:.1f}"
                  f"  air {r['air_share']:.2f}")

    show(load_rows, "load sweep (B = inf)")
    show(buf_rows, "buffer sweep")
    sel = [r["info"]["t_select"] for r in results]
    evl = [r["info"]["t_eval"] for r in results]
    p3_air = sum(r["info"]["p3_air"] for r in results)
    print(f"P3 chose an air route on {p3_air}/{len(results)} instances; "
          f"candidates per instance {np.mean([r['info']['n_cands'] for r in results]):.1f}")
    print(f"per instance: selection {np.mean(sel):.1f}s, evaluation {np.mean(evl):.1f}s")
    print(f"done in {time.time() - t_start:.1f}s -> results/flows_load.csv, "
          "results/flows_buffer.csv, figures/flows.pdf")


if __name__ == "__main__":
    main()
