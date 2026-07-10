"""UC2 - ubiquitous logistics: capacity uncertainty (H2).

Bulk demand W must be delivered over feeder passes whose deliverable volume is
lognormal (fade; A2). Timing is known (A1), so the only question is HOW MANY
passes to reserve. Provisioning strategies:

  mean-case  - k = ceil(W / E[V]): sized to the average, ignores variance
  worst-case - k = ceil(W / q05(V)): every pass assumed near its 5th percentile
  chance     - min k with P[sum_k V >= W] >= 1-eps  (eq. 12)

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
