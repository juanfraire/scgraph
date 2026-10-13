"""Placement probe on the deep topology (UC4): does the two-hop picture survive?

Answers R1: "The authors explicitly admit on page 14 that the distributed
planner is 'flattered' by the 2-hop setting." The paper's probe
(run_placement.py) runs on UC1 and UC3, where every route has two hops. Here
the same three placements run on UC4's far terminal (Dakar) to the Paris
gateway over the 60-satellite ISL mesh, where routes span 5 to 9 contacts, at
sigma = 40 s while the deadline margin over the reliable path moves through
{-60, 0, 60, 120, 240} s.

Placements (scgraph/routing/placement.py, used unchanged):

  ground      a ground centre: full candidate budget (40 sampled
              realizations, M_SEL draws per candidate), one SLA-aware route
              uplinked, followed blindly (GroundPlan); no contingency
  proxy       a sky proxy: ranked table per (node, time bucket), top 3 routes
              by P[T <= D] on the reduced budget of plan_proxy (300 draws,
              8 realizations, 4 buckets), resolved online (ProxyPlan).
              plan_proxy enumerates every node of the plan, which costs about
              70 s per UC4 instance, so the probe's main proxy restricts the
              table to the nodes on the source's candidate routes. Its table
              routes are computed on the sub-network those nodes span (the
              contacts between them), so a bundle that follows the table never
              reaches a node without an entry. On test instances, a table
              restricted in its nodes but with routes drawn on the full plan
              led the bundle to nodes without an entry in up to 59% of the
              realizations, a loss caused by the restriction itself.
  proxy_full  the unrestricted plan_proxy (every node), as a check on what the
              restriction costs
  dist        fully distributed: each node ranks its own outgoing contacts by
              earliest arrival at the next hop (DistributedPlan)

Every candidate set (ground, both proxy tables, scg_policy) comes from
candidate_routes(..., risk_classes=True), as in run_uc4.py: on UC4 the safe
route starts on the slow terminal->HAPS link and shares its space contacts
with the UAV routes, so the default disjoint sweep rarely offers it. Since
plan_proxy calls candidate_routes with the default generator, proxy_full uses
a local copy of it (same node set, ranking and budget) with the fix.

References (routing/reactive.py), on the same realizations:

  cgr_react   reactive CGR: each node holds the nominal plan and repairs with
              CGR when a contact fails; this is distributed execution with a
              global nominal view
  scg_repair  the ground route, then CGR repair (ground plus a fallback)
  scg_policy  SLA-aware with repair: first route by pick_route_under_repair,
              then CGR repair

The placements run through placement.execute(..., max_hops=20); the references
through execute_reactive. Every policy sees the same M held-out realizations,
drawn on a stream no planner touched. A semantic difference to keep in mind:
under placement.execute the proxy and distributed choosers inspect the realized
contact before committing (an absent contact, or one whose window will close
before the transfer ends, is skipped at no cost), while the ground route and the
reactive policies discover a failure only when it happens.

Common random numbers: instance i uses the seed (SEED, i) at every margin, so
the release time, UAV schedule and candidate set are shared and only the
deadline moves.

Usage: python experiments/run_placement_uc4.py [--quick] [--workers N] [--replot]
       python experiments/run_placement_uc4.py --proxy-diag
Outputs: results/placement_uc4.csv (one row per margin),
         results/placement_uc4_instances.csv (one row per instance),
         figures/placement_uc4.{pdf,png} (left: the paper's UC1 probe replotted
         from results/placement_uc1.csv; right: UC4)
--proxy-diag (4 instances at margins 60 and 240, 300 realizations, about two
minutes) writes results/placement_uc4_proxydiag.csv: the ranked table with 4
and 16 time buckets, each with and without loop avoidance, against the ground
route, and how the 4-bucket table's walks end (on time, late, stuck in a loop,
stuck on closed entries).
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
from _style import COL_BASE1, COL_BOUND, COL_SCG

from scgraph.routing.placement import DistributedPlan, GroundPlan, ProxyPlan, _hop, execute
from scgraph.routing.quantile import candidate_routes, pick_route, route_stats
from scgraph.routing.reactive import execute_reactive, pick_route_under_repair
from scgraph.scenarios.uc4_continuum import make_instance
from scgraph.stats import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261015
SIGMA = 40.0
MARGINS = [-60.0, 0.0, 60.0, 120.0, 240.0]
R_FULL = 20
M_SEL = 2000          # ground: draws per candidate
N_CAND_REAL = 40      # ground: sampled realizations feeding the candidate set
M_POL = 300           # scg_policy: planning realizations
M_REAL = 1000         # held-out realizations per instance
MAX_HOPS = 20
PROXY = dict(m=300, n_realizations=8, depth=3, n_buckets=4)   # plan_proxy defaults

PLACEMENTS = ("ground", "proxy", "proxy_full", "dist")
REFERENCES = ("cgr_react", "scg_repair", "scg_policy")
POLICIES = PLACEMENTS + REFERENCES


def proxy_restricted(graph, src, dst, size, t0, deadline, rng, cands, m, n_realizations,
                     depth, n_buckets):
    """plan_proxy's table, over the nodes on the source's candidate routes only.

    Same ranking as plan_proxy (top `depth` candidates by P[T <= D] per node
    and time bucket, on the same reduced budget). Candidate routes are drawn
    on the sub-network spanned by those nodes, so every table route stays in
    the table.
    """
    nodes = {src} | {c.to for r in cands for c in r}
    sub = graph.replaced([c for c in graph.contacts if c.frm in nodes and c.to in nodes])
    plan = _proxy_table(sub, nodes, dst, size, t0, deadline, rng, m, n_realizations,
                        depth, n_buckets, latent=graph.latent)
    return plan, len(nodes) - 1, len(sub.contacts)


def proxy_every_node(graph, src, dst, size, t0, deadline, rng, m, n_realizations,
                     depth, n_buckets):
    """plan_proxy (routing/placement.py) over every node, with risk-class
    candidates. A local copy only because plan_proxy calls candidate_routes
    with the default generator; the node set, ranking and budget are its own."""
    nodes = {src} | {c.to for c in graph.contacts}
    return _proxy_table(graph, nodes, dst, size, t0, deadline, rng, m, n_realizations,
                        depth, n_buckets, latent=graph.latent)


def _proxy_table(graph, nodes, dst, size, t0, deadline, rng, m, n_realizations, depth,
                 n_buckets, latent=None):
    table: dict[tuple[int, int], list[list[int]]] = {}
    span = max(1e-9, deadline - t0)
    for u in sorted(nodes):
        if u == dst:
            continue
        for k in range(n_buckets):
            t_k = t0 + span * k / n_buckets
            cs = candidate_routes(graph, u, dst, size, t_k, rng, n_realizations,
                                  risk_classes=True)
            if not cs:
                continue
            scored = [(r, route_stats(r, size, t_k, deadline, m, rng, latent=latent))
                      for r in cs]
            scored.sort(key=lambda rs: -rs[1]["p_success"])
            table[(u, k)] = [[c.cid for c in r] for r, _ in scored[:depth]]
    return ProxyPlan(size, t0, deadline, n_buckets, table)


def run_placement(reals, src, dst, size, t0, plan, max_hops=MAX_HOPS):
    """Arrival per realization, and per realization whether the bundle ever sat
    at a node whose (node, bucket) entry was missing (proxies only)."""
    base = plan.chooser()
    table = getattr(plan, "table", None)
    off = np.zeros(len(reals), dtype=bool)
    arr = np.empty(len(reals))
    for j, real in enumerate(reals):
        flag = [False]
        if table is not None:
            def choose(node, a, avail, used, _f=flag):
                if (node, plan.bucket(a)) not in table:
                    _f[0] = True
                return base(node, a, avail, used)
        else:
            choose = base
        arr[j] = execute(real, src, dst, size, t0, choose, max_hops=max_hops)
        off[j] = flag[0]
    return arr, off


def one_instance(task, _rng):
    margin, quick, inst = task["margin"], task["quick"], task["inst"]
    m_sel = 600 if quick else M_SEL
    m_pol = 120 if quick else M_POL
    m_real = 300 if quick else M_REAL
    pk = dict(PROXY, m=100, n_realizations=4) if quick else dict(PROXY)
    rng = np.random.default_rng([SEED, inst])
    rng_e = np.random.default_rng(rng.integers(2**63))   # held-out stream
    t_inst = time.time()

    g, p = make_instance(rng, SIGMA, margin=margin)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    mean = g.mean_plan()
    cands = candidate_routes(g, src, dst, size, t0, rng, N_CAND_REAL, risk_classes=True)

    r_g, _ = pick_route(g, src, dst, size, t0, D, rng, m=m_sel, cands=cands)
    plans = {"ground": GroundPlan([c.cid for c in r_g] if r_g else [])}
    t = time.time()
    plans["proxy"], n_nodes, n_sub = proxy_restricted(g, src, dst, size, t0, D, rng, cands, **pk)
    t_proxy = time.time() - t
    t = time.time()
    plans["proxy_full"] = proxy_every_node(g, src, dst, size, t0, D, rng, **pk)
    t_full = time.time() - t
    plans["dist"] = DistributedPlan(size)
    r_pol, _ = pick_route_under_repair(g, cands, src, dst, size, t0, D, rng, m=m_pol,
                                       nominal=mean)

    out = {"margin": margin, "inst": inst, "D_rel": D - t0,
           "nominal_hops": p["nominal_hops"], "reliable_hops": p["reliable_hops"],
           "n_cands": len(cands), "n_contacts": len(g.contacts),
           "ground_hops": len(r_g) if r_g else 0,
           "proxy_nodes": n_nodes, "proxy_sub_contacts": n_sub,
           "proxy_entries": len(plans["proxy"].table),
           "proxy_full_entries": len(plans["proxy_full"].table),
           "proxy_secs": t_proxy, "proxy_full_secs": t_full}

    reals = g.realize_batch(rng_e, m_real)
    arrs = {}
    for k in PLACEMENTS:
        arrs[k], off = run_placement(reals, src, dst, size, t0, plans[k])
        if k in ("proxy", "proxy_full"):
            out[f"{k}_offtable"] = float(off.mean())
    first = {"cgr_react": None,
             "scg_repair": [c.cid for c in r_g] if r_g else None,
             "scg_policy": [c.cid for c in r_pol] if r_pol else None}
    for k in REFERENCES:
        arrs[k] = np.array([execute_reactive(real, mean, src, dst, size, t0,
                                             first_route=first[k])[0] for real in reals])
    for k in POLICIES:
        a = arrs[k]
        ok = int(np.count_nonzero(a <= D))
        out[f"{k}_ok"], out[f"{k}_n"] = ok, len(a)
        out[f"{k}_succ"] = ok / len(a)
        out[f"{k}_reach"] = float(np.isfinite(a).mean())
        fin = a[np.isfinite(a)]
        out[f"{k}_lat"] = float(np.mean(fin)) - t0 if fin.size else math.nan
    # is the distributed collapse only the hop cap? the same walk, 10x the cap
    a200, _ = run_placement(reals, src, dst, size, t0, plans["dist"], max_hops=200)
    out["dist200_succ"] = float(np.mean(a200 <= D))
    out["dist200_reach"] = float(np.isfinite(a200).mean())
    out["secs"] = time.time() - t_inst
    return out


# --------------------------------------------------------------------------
# --proxy-diag: why the ranked table fails on a deep route
# --------------------------------------------------------------------------


def loopfree_chooser(plan: ProxyPlan):
    """ProxyPlan's online resolution, skipping entries whose first contact
    leads back to a node the bundle already visited."""
    def choose(node, a, avail, used):
        seen = {avail[c].frm for c in used if c in avail} | {node}
        for cids in plan.table.get((node, plan.bucket(a)), []):
            c = avail.get(cids[0]) if cids else None
            if c is None or c.frm != node or c.cid in used or c.to in seen:
                continue
            if math.isfinite(_hop(c, plan.size, a)):
                return c
        return None
    return choose


def classify_proxy(real, src, dst, size, t0, deadline, plan, max_hops=MAX_HOPS):
    """Outcome of ProxyPlan's walk: on time, late, or stuck because every
    top entry starts with a contact already used (the tables sent the bundle
    round a loop), or because they are absent, closed or too short."""
    avail = {c.cid: c for c in real.contacts}
    base = plan.chooser()
    node, a, used = src, t0, set()
    for _ in range(max_hops):
        if node == dst:
            break
        c = base(node, a, avail, used)
        if c is None:
            ents = plan.table.get((node, plan.bucket(a)), [])
            if ents and all(e and e[0] in used for e in ents):
                return "stuck_loop"
            return "stuck_closed"
        nxt = _hop(c, size, a)
        if not math.isfinite(nxt):
            return "stuck_closed"
        used.add(c.cid)
        node, a = c.to, nxt
    if node != dst:
        return "hop_cap"
    return "on_time" if a <= deadline else "late"


def diag_instance(task, _rng):
    margin, inst, m_real = task["margin"], task["inst"], task["m_real"]
    # same seeds and the same draw order as one_instance up to the 4-bucket
    # table, so that table is the main run's restricted proxy
    rng = np.random.default_rng([SEED, inst])
    rng_e = np.random.default_rng(rng.integers(2**63))
    g, p = make_instance(rng, SIGMA, margin=margin)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    cands = candidate_routes(g, src, dst, size, t0, rng, N_CAND_REAL, risk_classes=True)
    r_g, _ = pick_route(g, src, dst, size, t0, D, rng, m=M_SEL, cands=cands)
    choosers = {"ground": GroundPlan([c.cid for c in r_g] if r_g else []).chooser()}
    plans = {}
    for nb in (4, 16):
        plans[nb], _, _ = proxy_restricted(g, src, dst, size, t0, D, rng, cands,
                                           **dict(PROXY, n_buckets=nb))
        choosers[f"proxy_b{nb}"] = plans[nb].chooser()
        choosers[f"proxy_b{nb}_loopfree"] = loopfree_chooser(plans[nb])
    reals = g.realize_batch(rng_e, m_real)
    out = {"margin": margin, "inst": inst}
    for k, ch in choosers.items():
        out[k] = float(np.mean([execute(r, src, dst, size, t0, ch, max_hops=MAX_HOPS) <= D
                                for r in reals]))
    cls = [classify_proxy(r, src, dst, size, t0, D, plans[4]) for r in reals]
    for k in ("on_time", "late", "stuck_loop", "stuck_closed", "hop_cap"):
        out[f"b4_{k}"] = cls.count(k) / len(cls)
    return out


def proxy_diag(workers):
    tasks = [{"margin": mg, "inst": i, "m_real": 300} for mg in (60.0, 240.0) for i in range(4)]
    res = pmap(diag_instance, tasks, seed=SEED, workers=workers)
    rows = []
    for mg in (60.0, 240.0):
        sub = [r for r in res if r["margin"] == mg]
        row = {"margin": mg, "R": len(sub), "M": 300}
        for k in sub[0]:
            if k not in ("margin", "inst"):
                row[k] = float(np.mean([r[k] for r in sub]))
        rows.append(row)
        print(f"margin={mg:6.1f} | " + "  ".join(f"{k}={row[k]:.3f}" for k in row
                                                  if k not in ("margin", "R", "M")))
    write(ROOT / "results" / "placement_uc4_proxydiag.csv", rows)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def t_half(xs) -> float:
    xs = np.asarray([x for x in xs if np.isfinite(x)], dtype=float)
    if len(xs) < 2:
        return 0.0
    return float(sstats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / math.sqrt(len(xs)))


def aggregate(results):
    rows = []
    for mg in sorted({r["margin"] for r in results}):
        sub = [r for r in results if r["margin"] == mg]
        row = {"margin": mg, "R": len(sub), "M": int(sub[0]["ground_n"]),
               "D_rel": float(np.mean([r["D_rel"] for r in sub]))}
        for k in ("nominal_hops", "reliable_hops", "ground_hops", "n_cands", "n_contacts",
                  "proxy_nodes", "proxy_sub_contacts", "proxy_entries",
                  "proxy_full_entries", "proxy_secs", "proxy_full_secs",
                  "proxy_offtable", "proxy_full_offtable", "dist200_succ", "dist200_reach"):
            row[k] = float(np.mean([r[k] for r in sub]))
        for k in POLICIES:
            ok = int(sum(r[f"{k}_ok"] for r in sub))
            n = int(sum(r[f"{k}_n"] for r in sub))
            lo, hi = clopper_pearson(ok, n)
            row[k] = ok / n                                   # delivery ratio by D
            row[f"{k}_cp_lo"], row[f"{k}_cp_hi"] = lo, hi
            row[f"ci_{k}"] = t_half([r[f"{k}_succ"] for r in sub])
            row[f"{k}_reach"] = float(np.mean([r[f"{k}_reach"] for r in sub]))
            lat = [r[f"{k}_lat"] for r in sub if np.isfinite(r[f"{k}_lat"])]
            row[f"{k}_lat"] = float(np.mean(lat)) if lat else math.nan
        # instance-paired differences against the ground placement
        for k in ("proxy", "proxy_full", "dist", "cgr_react", "scg_policy"):
            d = np.array([r[f"{k}_succ"] - r["ground_succ"] for r in sub])
            row[f"d_{k}_ground"] = float(d.mean())
            row[f"d_{k}_ground_ci"] = t_half(d)
            row[f"d_{k}_ground_wins"] = int(np.sum(d > 0))
            row[f"d_{k}_ground_losses"] = int(np.sum(d < 0))
        d = np.array([r["proxy_full_succ"] - r["proxy_succ"] for r in sub])
        row["d_full_restr"], row["d_full_restr_ci"] = float(d.mean()), t_half(d)
        order = sorted(PLACEMENTS, key=lambda k: -row[k])
        row["placement_order"] = ">".join(order)
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
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

# the paper's Fig. 8 entity styles for the three placements, plus references
PSTYLE = {
    "ground": dict(color=COL_BASE1, linestyle="--", marker="s", markersize=5.5,
                   linewidth=2.4, zorder=1),
    "proxy": dict(color=COL_SCG, linestyle="-", marker="o", zorder=3),
    "proxy_full": dict(color=COL_SCG, linestyle=":", marker="o", mfc="white", zorder=3),
    "dist": dict(color="#7b4fa8", linestyle="-.", marker="^", zorder=2),
    "cgr_react": dict(color=COL_BOUND, linestyle=":", marker="v", zorder=2),
    "scg_policy": dict(color="#1b9e77", linestyle="-", marker="D", markersize=3.2, zorder=2),
}
PLABEL = {
    "ground": "ground-central (no contingency)",
    "proxy": "sky proxy (ranked table)",
    "proxy_full": "sky proxy, every node",
    "dist": "fully distributed (local, greedy)",
    "cgr_react": "ref.: CGR + repair",
    "scg_policy": "ref.: SLA-aware + repair",
}


def _yerr(rows, k):
    v = np.array([r[k] for r in rows])
    h = np.array([r[f"ci_{k}"] for r in rows])
    return v, np.vstack([np.minimum(h, v), np.minimum(h, 1.0 - v)])


def plot(rows, stem):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    uc1_path = ROOT / "results" / "placement_uc1.csv"
    if uc1_path.exists():
        uc1 = load_rows(uc1_path)
        x1 = [r["margin"] for r in uc1]
        for k in ("ground", "proxy", "dist"):
            v, e = _yerr(uc1, k)
            ax1.errorbar(x1, v, yerr=e, label=PLABEL[k], capsize=2, **PSTYLE[k])
    ax1.set_xlabel("deadline margin over the certain backbone [s]")
    ax1.set_ylabel("delivery ratio by deadline")
    ax1.invert_xaxis()
    ax1.set_title(r"UC1, two hops, $\sigma=40$ s (paper, Fig. 8)", fontsize=7.5)

    x2 = [r["margin"] for r in rows]
    for k in ("ground", "proxy", "proxy_full", "dist", "cgr_react", "scg_policy"):
        v, e = _yerr(rows, k)
        ax2.errorbar(x2, v, yerr=e, label=PLABEL[k], capsize=2, **PSTYLE[k])
    ax2.set_xlabel("deadline margin over the reliable path [s]")
    ax2.invert_xaxis()
    ax2.set_title(r"UC4, far terminal, 5 to 9 hops, $\sigma=40$ s", fontsize=7.5)
    ax2.set_ylim(-0.03, 1.03)
    ax2.legend(loc="lower center", ncol=2, fontsize=5.6, bbox_to_anchor=(0.5, 0.07),
               columnspacing=0.8, handlelength=2.4)
    _style.save(fig, str(ROOT / "figures" / stem))


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
    if "--proxy-diag" in sys.argv:
        proxy_diag(workers)
        print(f"done in {time.time() - t_start:.1f}s -> results/placement_uc4_proxydiag.csv")
        return
    sfx = "_quick" if quick else ""
    out_csv = ROOT / "results" / f"placement_uc4{sfx}.csv"
    inst_csv = ROOT / "results" / f"placement_uc4{sfx}_instances.csv"

    if replot:
        rows = load_rows(out_csv)
    else:
        R = 2 if quick else R_FULL
        margins = [0.0, 120.0] if quick else MARGINS
        tasks = [{"margin": mg, "quick": quick, "inst": i} for mg in margins for i in range(R)]
        print(f"{len(margins)} margins x {R} instances = {len(tasks)} tasks, "
              f"{workers} workers", flush=True)
        res = pmap(one_instance, tasks, seed=SEED, workers=workers)
        write(inst_csv, res)
        rows = aggregate(res)
        write(out_csv, rows)
        cpu = sum(r["secs"] for r in res)
        print(f"instance CPU time {cpu:.0f} s (mean {cpu / len(res):.1f} s, "
              f"max {max(r['secs'] for r in res):.1f} s)")

    for r in rows:
        print(f"margin={r['margin']:6.1f} R={int(r['R'])} | " + "  ".join(
            f"{k}={r[k]:.3f}" for k in POLICIES)
            + f" | reach dist {r['dist_reach']:.3f} (200 hops {r['dist200_reach']:.3f})"
            + f" | offtable {r['proxy_offtable']:.3f}/{r['proxy_full_offtable']:.3f}"
            + f" | {r['placement_order']}")
    plot(rows, f"placement_uc4{sfx}")
    print(f"done in {time.time() - t_start:.1f}s -> {out_csv.relative_to(ROOT)}, "
          f"figures/placement_uc4{sfx}.{{pdf,png}}")


if __name__ == "__main__":
    main()
