"""UC2 - ubiquitous logistics: capacity uncertainty (H2).

Bulk demand W must be delivered over feeder passes whose deliverable volume is
lognormal (fade; A2). Timing is known (A1), so the only question is HOW MANY
passes to reserve. Provisioning strategies:

  mean-case  - k = ceil(W / E[V]): sized to the average, ignores variance
  worst-case - k = ceil(W / q05(V)): every pass assumed near its 5th percentile
  chance     - min k with P[sum_k V >= W] >= 1-eps  (eq. 11)

Metric: provisioned overhead k*E[V]/W and achieved SLA. H2: the chance-
constrained plan meets the SLA at materially lower overhead than worst-case,
while mean-case misses it.
"""

from __future__ import annotations

import math

import numpy as np

from scgraph.distributions import LogNormal


def provisioning(
    rng: np.random.Generator,
    demand: float,
    mean_vol: float,
    cv: float,
    eps: float = 0.05,
    m: int = 200_000,
    k_max: int = 40,
) -> dict:
    """Provisioned pass counts + achieved SLA for the three strategies."""
    law = LogNormal.from_mean_cv(mean_vol, cv)
    k_mean = math.ceil(demand / mean_vol)
    k_wc = math.ceil(demand / law.quantile(0.05))

    # one big draw, reused across k via cumulative sums (common random numbers)
    v = law.sample(rng, (m, k_max))
    csum = np.cumsum(v, axis=1)

    def sla(k: int) -> float:
        k = min(k, k_max)
        return float(np.mean(csum[:, k - 1] >= demand))

    k_cc = next((k for k in range(k_mean, k_max + 1) if sla(k) >= 1 - eps), k_max)

    return {
        "k_mean": k_mean, "sla_mean": sla(k_mean),
        "k_wc": k_wc, "sla_wc": sla(k_wc),
        "k_cc": k_cc, "sla_cc": sla(k_cc),
        "overhead_mean": k_mean * mean_vol / demand,
        "overhead_wc": k_wc * mean_vol / demand,
        "overhead_cc": k_cc * mean_vol / demand,
    }


def cpd_pass_volumes(
    skeleton, gs_index: int, rate: float = 1.0, min_seconds: float = 30.0
) -> np.ndarray:
    """Nominal deliverable volume of each real downlink pass, in plan order.

    CPD passes are not interchangeable: their durations depend on how close to
    overhead the satellite goes, so the volumes are non-identically distributed
    for a purely *geometric* reason, on top of the fade A2 puts on each one.

    Order is chronological, because an operator reserves the *next* k passes
    and does not get to pick the good ones. Passes shorter than `min_seconds`
    are dropped: grazing passes at the elevation mask carry almost nothing and
    no real scheduler would reserve against them.
    """
    vols = [
        c.duration.mean() * c.rate.mean() * rate
        for c in skeleton.contacts
        if c.to == gs_index and c.duration.mean() >= min_seconds
    ]
    return np.asarray(vols)


def provisioning_cpd(
    rng: np.random.Generator,
    skeleton,
    gs_index: int,
    n_passes: int,
    cv: float,
    eps: float = 0.05,
    m: int = 200_000,
    k_max: int = 40,
) -> dict:
    """H2 on real pass geometry: heterogeneous nominal volumes, lognormal fade.

    The demand is what `n_passes` *average* passes would carry, so the question
    is unchanged (how many passes to reserve), but each reserved pass now has
    its own nominal volume drawn from the plan, and each is independently faded.
    Sizing the aggregate (eq. 11) is worth more here than in the homogeneous
    case, because the operator can no longer assume the next pass is as good as
    the last.
    """
    nominal = cpd_pass_volumes(skeleton, gs_index)
    if nominal.size == 0:
        raise ValueError(f"no downlink passes to node index {gs_index}")
    mean_vol = float(nominal.mean())
    demand = n_passes * mean_vol

    # passes are consumed in plan order; cycle the real sequence if we need more
    seq = np.resize(nominal, k_max)

    fade = LogNormal.from_mean_cv(1.0, cv).sample(rng, (m, k_max))
    v = seq[None, :] * fade
    csum = np.cumsum(v, axis=1)

    def sla(k: int) -> float:
        k = min(max(k, 1), k_max)
        return float(np.mean(csum[:, k - 1] >= demand))

    def overhead(k: int) -> float:
        return float(seq[: min(k, k_max)].sum() / demand)

    k_mean = int(np.searchsorted(np.cumsum(seq), demand) + 1)
    q05 = LogNormal.from_mean_cv(1.0, cv).quantile(0.05)
    k_wc = int(np.searchsorted(np.cumsum(seq) * q05, demand) + 1)
    k_wc = min(k_wc, k_max)
    k_cc = next((k for k in range(k_mean, k_max + 1) if sla(k) >= 1 - eps), k_max)

    return {
        "k_mean": k_mean, "sla_mean": sla(k_mean),
        "k_wc": k_wc, "sla_wc": sla(k_wc),
        "k_cc": k_cc, "sla_cc": sla(k_cc),
        "overhead_mean": overhead(k_mean),
        "overhead_wc": overhead(k_wc),
        "overhead_cc": overhead(k_cc),
        "mean_vol": mean_vol, "n_real_passes": int(nominal.size),
    }
