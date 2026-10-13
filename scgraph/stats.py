"""Exact binomial confidence bounds for Monte-Carlo SLA estimates (revision).

A Monte-Carlo run of M draws with k deadline successes estimates the delivery
probability p = P[T <= D]. The point estimate k/M says nothing about how far
the true p may sit below it, and a zero observed violation rate does not
establish a zero true violation probability. These helpers turn (k, M) into
statements with a stated confidence:

  cp_lower(k, n, delta)   one-sided Clopper-Pearson lower bound on p: the true
                          success probability is at least this, with
                          confidence 1 - delta.
  cp_upper(k, n, delta)   one-sided upper bound on p.
  clopper_pearson(k, n)   the two-sided exact interval.
  zero_failure_n(eps, d)  draws needed so that a run with no violation at all
                          supports P[violation] <= eps at confidence 1 - d.

They are exact (beta quantiles), so they hold at any M, including the small-M
and zero-failure corners where the normal approximation breaks down.
"""

from __future__ import annotations

import math

from scipy import stats


def cp_lower(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided (1 - delta) Clopper-Pearson lower bound on a proportion."""
    if n <= 0 or k <= 0:
        return 0.0
    return float(stats.beta.ppf(delta, k, n - k + 1))


def cp_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided (1 - delta) Clopper-Pearson upper bound on a proportion."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1.0 - delta, k + 1, n - k))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided exact interval for a proportion at confidence `conf`."""
    a = 1.0 - conf
    return cp_lower(k, n, a / 2), cp_upper(k, n, a / 2)


def zero_failure_n(eps: float, delta: float = 0.05) -> int:
    """Smallest M such that 0 violations in M draws gives P[viol] <= eps at 1-delta.

    From (1 - eps)^M <= delta, i.e. M >= ln(1/delta) / -ln(1 - eps), which is
    ln(1/delta)/eps to first order (the "rule of three" at delta = 0.05).
    """
    return int(math.ceil(math.log(1.0 / delta) / -math.log1p(-eps)))


def admission_draws(eps: float, n_candidates: int = 1, delta: float = 0.05,
                    headroom: float = 3.0, floor: int = 2000) -> int:
    """Draws per candidate so that a sound route can pass the admission test.

    Admission requires the Bonferroni-corrected lower bound to clear 1 - eps.
    With the bare minimum `zero_failure_n`, only a route that never fails in
    the sample is admitted; `headroom` buys room for routes whose true
    violation sits a factor below the budget.
    """
    base = zero_failure_n(eps, delta / max(1, n_candidates))
    return max(floor, int(math.ceil(headroom * base)))
