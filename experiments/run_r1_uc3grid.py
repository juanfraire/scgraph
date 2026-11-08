"""E4, second revision (R1.4): refining the grid does not recover dependence.

On the emergency case, two copies under a correlated collapse. The Bernoulli
corner's replica set (`bernoulli_select`, eq. bernproj) is chosen at slot
widths DELTAS and scored, like the correlation-aware set (`greedy_select` on
the joint law) and an oracle that sees the realisation, on held-out joint
realisations. Every transfer here takes 5 s, inside every slot width, so the
slot bookkeeping is not what separates the curves: the projection draws the
latent state per contact, and no slot width gives back which contacts fail
together.

Instances replay `run_rucop.uc3_task` (same seed, task order and instance
stream), so they are those of Fig. 10.

Usage: python experiments/run_r1_uc3grid.py [--quick]
Outputs: results/r1_uc3grid.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from _parallel import pmap

from scgraph.routing.bernoulli import bernoulli_select
from scgraph.routing.deterministic import cgr_route
from scgraph.routing.quantile import candidate_routes
from scgraph.routing.replication import greedy_select
from scgraph.routing.rucop import execute_route_set, realize_with_state
from scgraph.scenarios.uc3_emergency import make_instance as make_uc3
from scgraph.stats import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261013 + 1                 # run_rucop: pmap(uc3_task, ..., seed=SEED + 1)
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DELTAS = (60.0, 30.0, 10.0, 5.0, 2.0)
N_COPIES = 2
M_PROJ = 4000
M_SCORE = 2000
M_EVAL = 2000


def _cids(route):
    return [c.cid for c in route]


def uc3_task(task, rng):
    rho, _i, m_eval = task
    inst_rng, plan_rng, eval_rng = rng.spawn(3)       # as run_rucop.uc3_task
    g, p = make_uc3(inst_rng, rho)
    src, dst, size, t0, D = p["src"], p["dst"], p["size"], p["t0"], p["deadline"]

    sets = {}
    cands = candidate_routes(g, src, dst, size, t0, plan_rng, n_realizations=30)
    sets["scg"] = [_cids(r) for r in greedy_select(cands, N_COPIES, size, t0, D, M_SCORE,
                                                   plan_rng, latent=g.latent)] if cands else []
    for d in DELTAS:
        sets[f"bern@{d:g}"] = [_cids(r) for r in bernoulli_select(g, src, dst, size, t0, D, d,
                                                                  N_COPIES, plan_rng, m=M_PROJ)]
    ok = {k: 0 for k in list(sets) + ["oracle"]}
    for _ in range(m_eval):
        real, _z = realize_with_state(g, eval_rng)
        by_cid = {c.cid: c for c in real.contacts}
        for k, rs in sets.items():
            ok[k] += execute_route_set(by_cid, rs, size, t0) <= D
        _, arr = cgr_route(real, src, dst, size, t0)
        ok["oracle"] += arr <= D
    return {"rho": rho, "ok": ok, "n": m_eval}


def main():
    quick = "--quick" in sys.argv
    t = time.time()
    R = 6 if quick else 40
    m_eval = 300 if quick else M_EVAL
    rhos = [0.0, 0.5, 0.9] if quick else RHOS
    tasks = [(r, i, m_eval) for r in rhos for i in range(R)]
    res = pmap(uc3_task, tasks, seed=SEED)
    names = ["oracle", "scg"] + [f"bern@{d:g}" for d in DELTAS]
    rows = []
    for rho in rhos:
        part = [r for r in res if r["rho"] == rho]
        for k in names:
            ok = sum(r["ok"][k] for r in part)
            n = sum(r["n"] for r in part)
            lo, hi = clopper_pearson(ok, n)
            per = [r["ok"][k] / r["n"] for r in part]
            rows.append({"rho": rho, "policy": k, "delivery": ok / n, "lo": lo, "hi": hi,
                         "inst_ci": 1.96 * float(np.std(per, ddof=1)) / np.sqrt(len(per)),
                         "R": len(part), "M_eval": m_eval})
    (ROOT / "results").mkdir(exist_ok=True)
    sfx = "_quick" if quick else ""
    with open(ROOT / "results" / f"r1_uc3grid{sfx}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for rho in rhos:
        line = "  ".join(f"{r['policy']}={r['delivery']:.3f}" for r in rows if r["rho"] == rho)
        print(f"rho={rho:.1f}  {line}")
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
