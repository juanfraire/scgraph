"""E2, second revision (R1.3): why the Bernoulli corner fails on the multi-hop continuum.

The paper attributed the corner's 60-84% violation on UC4 to the one-hop-per-
slot rule. This experiment tests that on the instances of Fig. 9:

  bern@D        the static Bernoulli router (eq. bernproj, product objective,
                one hop per slot) at slot width D in SLOTS
  chain@D       the same projection, but hops may chain inside a slot
                (`advance=0`), an optimistic reading of the slotted model that
                removes the slot rule altogether
  dur@D         the same projection, with each hop advancing by its mean
                transfer time plus propagation rounded up to whole slots
                (`hop_slots`), so the slot bookkeeping follows the hop's mean
                duration: the fairest slotted baseline we can build from
                eq. (bernproj)
  scg           the max-probability router on the full laws, for reference

The 10 s projection is drawn right after the candidate set, from the same
stream position as `run_uc4`'s Bernoulli router, so `bern@10` routes are those
of Fig. 9.

Every route is executed in continuous time on held-out realisations, and each
failure is attributed to the first hop that breaks: the contact is absent, or
its window closes before the bundle is through (missed), by contact kind; or
every hop works but the bundle arrives after the deadline (late). The model's
own predicted success is recorded next to the held-out one.

Instances replay `run_uc4.one_instance` (held-out seed, then instance) under
the same seeds and task order.

Usage: python experiments/run_r1_bernoulli.py [--quick]
Outputs: results/r1_bernoulli.csv, results/r1_bernoulli_instances.csv
"""

from __future__ import annotations

import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _parallel import pmap

from scgraph.routing.bernoulli import bernoulli_projection, bernoulli_route
from scgraph.routing.quantile import candidate_routes, pick_route
from scgraph.scenarios.uc4_continuum import make_instance

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013                     # run_uc4.SEED
SLOTS = (30.0, 10.0, 5.0, 2.0)      # one hop per slot
CHAIN_SLOTS = (30.0, 10.0)          # hops chained inside a slot
DUR_SLOTS = (30.0, 10.0, 5.0, 2.0)  # hops advance by their mean duration
M_PROJ = 2000                       # draws per contact for the projection (as run_uc4)
M_SEL = 2000
M_EVAL = 4000                       # held-out realisations per route
CAUSES = ("absent", "missed")
FIG9_SLOT = 10.0                    # run_uc4.BERN_SLOT
TOL_FRAC = 0.01                     # an overflow under 1% of a slot does not cost a slot


def hop_slots(g, size, slot):
    """{cid: slots a hop takes}: mean transfer time plus propagation, rounded up.

    The tolerance keeps millisecond propagation (owlt of 1e-4 to a few 1e-3 s
    on the multi-hop plan) from pushing an exact multiple of the slot into one
    more slot; a real overrun, such as the mobile event's 10.5 s and 13 s hops
    at a 10 s slot, still takes the extra slot.
    """
    return {c.cid: max(1, int(math.ceil((size / c.rate.mean() + c.owlt) / slot - TOL_FRAC)))
            for c in g.contacts}


def walk(route, real, size, t0, D):
    """Continuous-time execution of a committed route on one realisation.

    Returns (outcome, kind): outcome in {"ok", "late", "absent", "missed"},
    kind the contact kind of the failing hop (None for ok / late).
    """
    avail = {c.cid: c for c in real.contacts}
    a = t0
    for sc in route:
        c = avail.get(sc.cid)
        if c is None:
            return "absent", sc.kind
        tau = max(a, c.t_start)
        e = tau + size / c.rate
        if e > c.t_end:
            return "missed", sc.kind
        a = e + c.owlt
    return ("ok" if a <= D else "late"), None


def score(route, g, size, t0, D, reals):
    if not route:
        return {"viol": 1.0, "late": 0.0, "causes": Counter({("noroute", None): len(reals)})}
    out = Counter(walk(route, r, size, t0, D) for r in reals)
    n = len(reals)
    return {"viol": 1.0 - out[("ok", None)] / n, "causes": out}


def one_instance(task, rng):
    sigma, margin, quick = task["sigma"], task["margin"], task["quick"]
    m_eval = 600 if quick else M_EVAL
    rng_e = np.random.default_rng(rng.integers(2**63))   # held-out stream (as run_uc4)
    g, p = make_instance(rng, sigma, margin=margin)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    reals = g.realize_batch(rng_e, m_eval)

    routes, preds = {}, {}
    cands = candidate_routes(g, src, dst, size, t0, rng, 40, risk_classes=True)

    def slotted(slot):
        K = max(1, int(math.ceil((D - t0) / slot)))
        q = bernoulli_projection(g, K, slot, size, M_PROJ, rng, t0=t0)
        variants = []
        if slot in SLOTS:
            variants.append((f"bern@{slot:g}", 1))
        if slot in CHAIN_SLOTS:
            variants.append((f"chain@{slot:g}", 0))
        if slot in DUR_SLOTS:
            variants.append((f"dur@{slot:g}", hop_slots(g, size, slot)))
        for name, adv in variants:
            r, pr = bernoulli_route(g, src, dst, size, t0, D, slot, rng, q=q, advance=adv)
            routes[name], preds[name] = r, pr

    # the Fig. 9 projection first, from run_uc4's stream position, so bern@10 is its route
    slotted(FIG9_SLOT)
    routes["scg"], st = pick_route(g, src, dst, size, t0, D, rng, m=M_SEL, cands=cands)
    preds["scg"] = st["p_success"]
    for slot in sorted((set(SLOTS) | set(CHAIN_SLOTS) | set(DUR_SLOTS)) - {FIG9_SLOT},
                       reverse=True):
        slotted(slot)

    out = {"sigma": sigma, "margin": margin, "D_rel": D - t0}
    for name, r in routes.items():
        s = score(r, g, size, t0, D, reals)
        out[f"{name}_viol"] = s["viol"]
        out[f"{name}_pred"] = preds[name]
        out[f"{name}_hops"] = len(r) if r else 0
        out[f"{name}_kinds"] = "-".join(c.kind for c in r) if r else ""
        n = len(reals)
        for (outcome, kind), cnt in s["causes"].items():
            if outcome == "ok":
                continue
            label = outcome if kind is None else f"{outcome}:{kind}"
            out[f"{name}_c_{label}"] = cnt / n
    return out


def aggregate(res, key, names):
    rows = []
    labels = sorted({k.split("_c_", 1)[1] for r in res for k in r if "_c_" in k})
    for x in sorted({r[key] for r in res}):
        sub = [r for r in res if r[key] == x]
        for name in names:
            row = {key: x, "policy": name, "R": len(sub),
                   "viol": float(np.mean([r[f"{name}_viol"] for r in sub])),
                   "pred_viol": float(np.mean([1.0 - r[f"{name}_pred"] for r in sub])),
                   "hops": float(np.mean([r[f"{name}_hops"] for r in sub]))}
            for lab in labels:
                row[f"c_{lab}"] = float(np.mean([r.get(f"{name}_c_{lab}", 0.0) for r in sub]))
            rows.append(row)
    return rows


def write(path, rows):
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


UC1_SIGMAS = (30.0, 60.0)
UC1_SLOTS = (10.0, 30.0)


def uc1_instance(task, rng):
    """Mobile event: published slot rule against duration-aware slots.

    Hops take 10.5 s (air) and 13 s (space, 3 s propagation), so the rule
    binds only at slots under 13 s. Both routes are scored on the same
    held-out draws (paired).
    """
    from scgraph.scenarios.uc1_event import make_instance_cpd
    from scgraph.verify.montecarlo import evaluate_route

    sigma, quick = task["sigma"], task["quick"]
    m_eval = 2000 if quick else 10_000
    seed_e = int(rng.integers(2**63))
    g, p = make_instance_cpd(rng, sigma)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]
    out = {"sigma": sigma}
    for slot in UC1_SLOTS:
        K = max(1, int(math.ceil((D - t0) / slot)))
        q = bernoulli_projection(g, K, slot, size, M_PROJ, rng, t0=t0)
        adv = hop_slots(g, size, slot)
        for name, a in ((f"bern@{slot:g}", 1), (f"dur@{slot:g}", adv)):
            r, _ = bernoulli_route(g, src, dst, size, t0, D, slot, rng, q=q, advance=a)
            ev = evaluate_route(r, size, t0, D, m_eval, np.random.default_rng(seed_e))
            out[f"{name}_viol"] = ev["viol"]
    return out


def run_uc1(quick):
    R = 6 if quick else 40
    tasks = [{"sigma": s, "quick": quick} for s in UC1_SIGMAS for _ in range(R)]
    res = pmap(uc1_instance, tasks, seed=SEED + 7)
    rows = []
    for s in UC1_SIGMAS:
        sub = [r for r in res if r["sigma"] == s]
        row = {"sigma": s, "R": len(sub)}
        for slot in UC1_SLOTS:
            for name in (f"bern@{slot:g}", f"dur@{slot:g}"):
                row[name] = float(np.mean([r[f"{name}_viol"] for r in sub]))
        rows.append(row)
    sfx = "_quick" if quick else ""
    write(ROOT / "results" / f"r1_bernoulli_uc1{sfx}.csv", rows)
    for r in rows:
        print("uc1 sigma={:.0f} ".format(r["sigma"]) + "  ".join(
            f"{k}={v:.4f}" for k, v in r.items() if "@" in k))


def main():
    quick = "--quick" in sys.argv
    if "--uc1" in sys.argv:
        run_uc1(quick)
        return
    t = time.time()
    R = 4 if quick else 30
    sigmas = [0.0, 40.0] if quick else [0.0, 10.0, 20.0, 40.0, 60.0]
    margins = [0.0, 240.0] if quick else [-60.0, 0.0, 60.0, 120.0, 240.0]
    tasks = [{"sigma": s, "margin": 60.0, "quick": quick} for s in sigmas for _ in range(R)]
    res_s = pmap(one_instance, tasks, seed=SEED)
    tasks = [{"sigma": 40.0, "margin": mg, "quick": quick} for mg in margins for _ in range(R)]
    res_m = pmap(one_instance, tasks, seed=SEED + 1)
    names = (["scg"] + [f"bern@{s:g}" for s in SLOTS] + [f"chain@{s:g}" for s in CHAIN_SLOTS]
             + [f"dur@{s:g}" for s in DUR_SLOTS])
    rows = ([{"sweep": "sigma", "margin": 60.0, **r} for r in aggregate(res_s, "sigma", names)]
            + [{"sweep": "margin", "sigma": 40.0, **r} for r in aggregate(res_m, "margin", names)])
    (ROOT / "results").mkdir(exist_ok=True)
    sfx = "_quick" if quick else ""
    write(ROOT / "results" / f"r1_bernoulli{sfx}.csv", rows)
    write(ROOT / "results" / f"r1_bernoulli_instances{sfx}.csv",
          [{"sweep": "sigma", **r} for r in res_s] + [{"sweep": "margin", **r} for r in res_m])
    for r in rows:
        causes = {k[2:]: v for k, v in r.items() if k.startswith("c_") and v > 0.005}
        top = ", ".join(f"{k} {v:.2f}" for k, v in sorted(causes.items(), key=lambda kv: -kv[1])[:4])
        print(f"{r['sweep']:6s} sigma={r['sigma']:4.0f} margin={r['margin']:5.0f} {r['policy']:9s}"
              f" viol {r['viol']:.3f} (model {r['pred_viol']:.3f}) hops {r['hops']:.1f} | {top}")
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
