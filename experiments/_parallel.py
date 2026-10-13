"""Seeded, order-independent parallel map for the revision experiments.

Each task gets its own generator spawned from one SeedSequence, so a result
depends only on (seed, task index) and never on scheduling or on how many
worker processes ran it. `workers=1` runs inline, which is what the tests and
--quick runs use.

    from _parallel import pmap
    rows = pmap(one_instance, tasks, seed=SEED, workers=3)

`one_instance(task, rng)` must be a module-level function (picklable).
The worker count defaults to SCGRAPH_WORKERS, else 3, which leaves room for
several experiment tracks to share the machine.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def default_workers() -> int:
    return int(os.environ.get("SCGRAPH_WORKERS", "3"))


def _call(args):
    fn, task, seed_seq = args
    return fn(task, np.random.default_rng(seed_seq))


def pmap(fn, tasks, seed: int, workers: int | None = None, chunksize: int = 1):
    """[fn(task_i, rng_i) for each task], rng_i spawned from SeedSequence(seed)."""
    tasks = list(tasks)
    seqs = np.random.SeedSequence(seed).spawn(len(tasks))
    jobs = [(fn, t, s) for t, s in zip(tasks, seqs)]
    workers = default_workers() if workers is None else workers
    if workers <= 1 or len(jobs) <= 1:
        return [_call(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_call, jobs, chunksize=chunksize))
