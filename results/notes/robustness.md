# Robustness to misspecified laws, inter-contact dependence, and learned laws

Revision items T6 (R2.6, R3.2) and T7 (R2.10). Four new experiments, run on
2026-10-13. Library code was not modified: every builder, planner variant and
helper lives in the experiment files.

| Study | Script | Outputs | Full run | `--quick` |
|---|---|---|---|---|
| 1. Heavy-tailed timing (UC1) | `experiments/run_misspec_timing.py` | `results/robust_timing.csv`, `figures/robust_timing.{pdf,png}` | 62 s | 9 s |
| 2. Fade variance and pass dependence (UC2) | `experiments/run_misspec_capacity.py` | `results/robust_capacity.csv`, `figures/robust_capacity.{pdf,png}` | 22 s | 7 s |
| 3. Dependence strength and structure (UC3) | `experiments/run_dependence.py` | `results/robust_dependence.csv`, `figures/robust_dependence.{pdf,png}` | 171 s | 5 s |
| 4. Learning the timing law (UC1) | `experiments/run_estimation.py` | `results/estimation.csv`, `figures/estimation.{pdf,png}` | 145 s | 7 s |

Wall times are for 3 worker processes on this Mac (Python 3.14, shared with
other jobs), so treat them as indicative. Run any study with

```bash
.venv/bin/python experiments/run_misspec_timing.py      # add --quick for a smoke test
.venv/bin/python experiments/run_misspec_capacity.py
.venv/bin/python experiments/run_dependence.py --check   # --check also asserts the builder equivalence below
.venv/bin/python experiments/run_estimation.py
```

Common conventions. Every task gets its own generator from
`_parallel.pmap` (one `SeedSequence` per study: 20261013, 20261014, 20261015,
20261016), so results do not depend on the worker count; `SCGRAPH_WORKERS`
sets it (default 3). Every policy is selected on the planner's belief and
scored on an independent held-out stream of the TRUE model. Violation rates
come with an instance-level 95% CI and a pooled exact (Clopper-Pearson)
one-sided upper bound; "breach" means the instance's held-out violation
exceeds the budget, and "significant breach" means its exact 97.5% lower
bound does. `.venv/bin/pytest -q` stays green (121 passed); no test file was
added because no reusable helper module was created.

---

## 1. Heavy-tailed timing law (UC1, `make_instance_cpd`)

**Question (R2.6).** What happens to the SLA when the true timing law has
heavier tails than the TruncNormal the planner believes?

**Setup.** The planner believes each air contact's start is
TruncNormal(nominal, sigma), truncated at +-3.5 sigma, exactly as the instance
is built. The truth replaces every air start with a location-scale Student-t
with nu in {inf, 10, 5, 3, 2.5} degrees of freedom, truncated at +-12 sigma,
centred on the same nominal start. sigma in {20, 40} s; deadline margin 40 s
(the UC1 default, so the timing-certain backbone always meets the deadline).
300 instances per sigma, 40 000 held-out truth draws per route and truth.

Matching the spread. `TruncStudentT.matched_std` matches the untruncated
variance. At nu = 2.5 the t variance converges so slowly that cutting at
+-12 sigma removes a quarter of it (truncated std 0.85 sigma; 0.95 sigma at
nu = 3), which would make the heavy-tailed truth narrower as well. The script
therefore re-solves the scale so the truncated law has std exactly sigma
(`t_scale`; the CSV column `std_true` is 1.000 for every nu). The comparison
changes the tail shape and holds the standard deviation fixed.

Policies: deterministic CGR on the mean plan; the SLA-aware max-P router
(`pick_route`, eq. 6); the chance-constrained router (`pick_route_chance`,
lcb admission, delta = 0.05) at eps = 0.05 and at eps = 0.01; and two
mitigations of the chance-constrained router, (i) a heavy-tailed belief
(t with nu = 3, same std, +-12 sigma) and (ii) an inflated belief (1.5 sigma).
All share one candidate set per instance, generated on the nominal belief.

**Key numbers, sigma = 20 s.** Mean held-out violation; in brackets the
fraction of instances whose violation exceeds the policy's budget (eps, or
0.05 for the two eps-free policies). "pred" is what the planner itself
predicts for its route under its belief.

| Policy | pred | nu = inf | 10 | 5 | 3 | 2.5 | delivery time |
|---|---|---|---|---|---|---|---|
| deterministic CGR | 0.0124 | 0.0127 [0] | 0.0148 [0.3%] | 0.0173 [0] | 0.0199 [0] | 0.0209 [0] | 75 s |
| SLA-aware max P | 0.0001 | 6.4e-5 [0] | 1.4e-4 [0] | 2.6e-4 [0] | 4.4e-4 [0] | 5.3e-4 [0] | 541 s |
| chance, eps = 0.05, normal belief | 0.0103 | 0.0105 [0] | 0.0127 [0] | 0.0154 [0] | 0.0184 [0] | 0.0195 [0] | 75 s |
| chance, eps = 0.05, t3 belief | 0.0186 | 0.0109 [0] | 0.0130 [0.3%] | 0.0157 [0] | 0.0186 [0] | 0.0197 [0] | 76 s |
| chance, eps = 0.05, 1.5 sigma belief | 0.0170 | 0.0013 [0] | 0.0024 [0] | 0.0039 [0] | 0.0061 [0] | 0.0071 [0] | 286 s |
| chance, eps = 0.01, normal belief | 0.0018 | 0.0019 [0] | 0.0032 [0] | 0.0052 [9%] | 0.0077 [47%] | 0.0089 [63%] | 231 s |
| chance, eps = 0.01, t3 belief | 0.0000 | 2.4e-6 [0] | 5.8e-6 [0] | 1.5e-5 [0] | 2.4e-5 [0] | 3.0e-5 [0] | 565 s |
| chance, eps = 0.01, 1.5 sigma belief | 0 | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 0 [0] | 567 s |

Significant breaches (exact lower bound above eps) for the eps = 0.01 normal
belief: 5%, 36%, 53% of instances at nu = 5, 3, 2.5. The worst single
instance there measures 0.0195 against a budget of 0.01. At eps = 0.05 the
worst instance of the normal-belief router measures 0.040.

**sigma = 40 s.** Every SCG policy (max P and all chance-constrained variants,
both budgets) routes every instance on the backbone, and its held-out
violation is 0 in 12 million draws for every nu (pooled 95% upper bound
2.5e-7). The deterministic plan violates in every instance; its mean
violation falls as the tail gets heavier, 0.134, 0.131, 0.123, 0.105, 0.097
for nu = inf, 10, 5, 3, 2.5, because at equal variance the t law moves mass
from the shoulders into the centre and the far tail, and this plan fails at
the shoulders.

**Reading.**
- How far the SLA moves under a tail misspecification depends on how deep in
  the tail the budget sits. At eps = 0.05 the admitted air routes have true
  violation near 1%, the heaviest tail roughly doubles it (0.0105 to 0.0195),
  and the budget still holds with room to spare in every instance. At
  eps = 0.01 the admitted routes sit closer to the budget and the tail hurts
  more in relative terms (planner predicts 0.0018, truth 0.0089 at nu = 2.5,
  a factor of 4.9): the mean stays under budget, yet 47% to 63% of instances
  exceed it once nu <= 3. The SLA breaks at the instance level before it
  breaks on average.
- The planner's predicted violation is accurate under the normal truth
  (0.0103 predicted, 0.0105 measured) and drifts away steadily as the tail
  thickens. The drift is the model error; at 12 million held-out draws per
  point, sampling noise is two orders of magnitude smaller.
- A heavy-tailed belief (t3) is the targeted fix. At eps = 0.01 it restores
  the budget for every tail tested (worst instance 0.0090 at nu = 2.5), at the
  price of falling back on the backbone (565 s against 231 s). At eps = 0.05
  it changes almost nothing, because at equal variance a t3 law puts less
  mass at the two-sigma shoulder where that budget binds, so the router
  admits the same fast routes. Inflating sigma by 1.5 is a blunt fix: it buys
  a large margin at eps = 0.05 (mean 0.0071 at nu = 2.5) by tripling the delivery
  time (286 s against 75 s), and at eps = 0.01 it forces the backbone too.
- When the backbone is available and the jitter is large (sigma = 40 s), the
  SLA-aware policies are immune: they already escape to the timing-certain
  path. Tail misspecification only matters for policies that keep a jittered
  route, which at sigma = 40 s means the deterministic plan.
- Not covered: a negative deadline margin, where no route is safe and every
  policy has to gamble on the air layer. Heavy tails should hurt more there.

**Sentences for the paper.**
1. We replaced each air contact's TruncNormal start law by a Student-t law of
   equal standard deviation with nu down to 2.5, while the router kept
   planning on the normal law.
2. At eps = 0.05 the chance-constrained router held the budget in every
   instance, with the held-out violation rising from 1.05% to 1.95%; at
   eps = 0.01 the mean stayed under budget (0.89%) but 63% of instances
   exceeded it at nu = 2.5.
3. Planning on a heavy-tailed belief (t, nu = 3) restored the 1% budget for
   every tail tested, at the cost of routing on the timing-certain backbone.
4. Tighter budgets therefore call for a tail-aware law, while at the 5%
   operating point the normal law is a safe working assumption in UC1.

---

## 2. Fade variance and inter-pass dependence (UC2, `provisioning_cpd`)

**Question (R2.6, R3.2).** Do the Fig. 6 plans keep the 0.95 SLA when the
fade variance is under-estimated, or when passes thought independent fade
together? How much of the chance-constrained saving over worst-case sizing
survives once the planner knows about the dependence?

**Setup.** Plans are sized on the belief (lognormal fade with cv in
{0.3, 0.5, 0.9}, independent passes; demand = 4 average passes, eps = 0.05)
and scored on 200 000 held-out draws of the truth. Truth: cv_true/cv in
{1, 1.25, 1.5, 2} at rho = 0, and rho_true in {0, 0.1, 0.2, 0.3, 0.5} at
cv_true = cv, where rho is the correlation of the passes' log-fades through
one common factor (`true_fades`). The task asked for cv in {0.3, 0.5}; 0.9 is
added because the paper's 2.5x headline lives there.

Two instance sets. "paper" is the Fig. 6 pass sequence (reservation starts at
the first real pass) through `provisioning_cpd`. "rotations" repeats
everything for all 27 starting passes of the CPD plan (100 000 sizing and
100 000 held-out draws, common random numbers across starts), because a
single sequence's integer rounding can hide or exaggerate fragility.

Dependence-aware planner (`size_chance` in the script). It sizes the reserve
on draws of the same one-factor model at an assumed rho_plan, and admits a
reserve size when the exact 95% lower bound on its SLA clears 0.95 (the
same rule as the lcb route admission), so a plan sized on finite draws does
not miss by Monte-Carlo noise. Run on the diagonal rho_plan = rho_true it
gives the reserve actually needed; run at a fixed rho_plan against every
rho_true it shows what a robust assumption buys.

**Key numbers: the Fig. 6 plans on their own sequence** (held-out SLA; target
0.95; worst-case plan >= 0.994 everywhere).

| cv (plan k, overhead) | truth | mean-case | chance-constrained |
|---|---|---|---|
| 0.3 (k_cc = 9, 1.36) | cv x1 / x1.25 / x1.5 / x2 | 0.926 / 0.869 / 0.818 / 0.738 | 0.997 / 0.985 / 0.963 / **0.906** |
| | rho 0.1 / 0.2 / 0.3 / 0.5 | 0.874 / 0.833 / 0.803 / 0.754 | 0.983 / 0.963 / **0.941** / **0.898** |
| 0.5 (k_cc = 10, 1.66) | cv x1 / x1.25 / x1.5 / x2 | 0.789 / 0.725 / 0.676 / 0.603 | 0.998 / 0.990 / 0.972 / **0.919** |
| | rho 0.1 / 0.2 / 0.3 / 0.5 | 0.733 / 0.695 / 0.668 / 0.625 | 0.986 / 0.964 / **0.941** / **0.894** |
| 0.9 (k_cc = 11, 1.99) | cv x1 / x1.25 / x1.5 / x2 | 0.630 / 0.579 / 0.540 / 0.483 | 0.991 / 0.969 / **0.941** / **0.877** |
| | rho 0.1 / 0.2 / 0.3 / 0.5 | 0.590 / 0.562 / 0.539 / 0.502 | 0.957 / **0.918** / **0.881** / **0.816** |

Bold: below 0.95. Every bold value stays below 0.95 at its exact one-sided
95% upper bound (the largest is 0.9422), so these breaches are not
Monte-Carlo noise.

**All 27 starting passes** (fraction of starts whose chance-constrained plan
still meets 0.95; at cv x1, rho = 0 it is 1.00 for every cv):

| cv | x1.25 | x1.5 | x2 | rho 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.89 | 0.67 | 0.26 | 0.89 | 0.81 | 0.63 | 0.52 |
| 0.5 | 0.59 | 0.30 | 0.00 | 0.74 | 0.44 | 0.30 | 0.07 |
| 0.9 | 0.41 | 0.04 | 0.00 | 0.37 | 0.07 | 0.00 | 0.00 |

**The saving that survives dependence** (dependence-aware plan sized at the
true rho; saving = worst-case overhead / dependence-aware overhead; mean over
the 27 starts, Fig. 6 sequence in brackets). These plans meet the SLA
held-out up to Monte-Carlo resolution: the few starts that measure below
0.95 do so by at most 0.0004, inside the noise of 100 000 draws.

| rho | cv = 0.9: overhead, saving, share of the rho = 0 saving kept | cv = 0.5 | cv = 0.3 |
|---|---|---|---|
| 0 | 2.04, 2.46x [2.52x], 100% | 1.65, 1.62x [1.55x] | 1.50, 1.26x [1.46x] |
| 0.1 | 2.27, 2.21x [2.52x], 92% | 1.75, 1.52x | 1.53, 1.24x |
| 0.2 | 2.47, 2.02x [2.21x], 85% | 1.85, 1.44x | 1.53, 1.23x |
| 0.3 | 2.74, 1.82x [1.95x], 76% | 1.94, 1.37x | 1.58, 1.19x |
| 0.5 | 3.25, 1.53x [1.59x], 59% | 2.14, 1.24x | 1.67, 1.13x |
| 0.7 | 3.89, 1.28x [1.28x], 37% | 2.33, 1.14x | 1.73, 1.09x |
| 0.9 | 4.61, 1.08x [1.00x], 12% | 2.55, 1.03x | 1.83, 1.03x |
| 1.0 | 5.06, 0.98x [1.00x], 0% | 2.66, 0.99x | 1.87, 1.00x |

Worst-case overheads (rotation mean): 4.96 (cv 0.9), 2.64 (0.5), 1.87 (0.3).
At rho = 1 every pass fades identically, so sizing for P >= 0.95 is exactly
the worst-case rule, and the measured saving lands on 1.0 as it must.

**Robust assumption (cv = 0.5, all starts).** A planner that assumes
rho_plan = 0.1 / 0.2 / 0.3 / 0.5 reserves 1.75 / 1.85 / 1.94 / 2.14 times
the demand (independence: 1.67; worst case: 2.64) and meets the SLA in every
start up to its assumed rho; one step beyond, the share drops to 0.70 / 0.70
/ 0.41 / 0.37. The rho_plan = 0.3 plan also absorbs a 1.5x variance
under-estimate in 96% of starts.

**Variance under-estimate, what restores it.** Sizing on the true cv
(independent passes) needs 1.77 / 1.88 / 2.15 at cv = 0.5 for x1.25 / x1.5 /
x2 (against 1.67 at the belief) and 2.25 / 2.41 / 2.76 at cv = 0.9 (against
2.03), all far below worst-case.

**Reading.**
- The mean-case plan misses the target everywhere, as in Fig. 6, and gets
  worse with either misspecification. The worst-case plan holds everywhere
  tested (its SLA dips to 0.994 at rho = 0.5) because the 5% quantile per pass
  is a very conservative rule under any of these truths.
- The chance-constrained plan is sized for the belief, so it has no built-in
  margin against a wrong law. On the Fig. 6 sequence, at cv 0.3 and 0.5, it
  tolerates a 1.5x variance under-estimate (0.963 and 0.972) and breaks at 2x
  (0.906 and 0.919); it tolerates rho = 0.2 and breaks at rho = 0.3 (0.941).
  At cv = 0.9 it breaks earlier, at 1.5x (0.941) and at rho = 0.2 (0.918).
  These are the favourable numbers: the Fig. 6 sequence carries integer
  headroom (in-sample SLA 0.997 at cv 0.3). Across all starting passes,
  rho = 0.1 already breaks the plan in 11% (cv 0.3), 26% (cv 0.5) and 63%
  (cv 0.9) of starts.
- The 2.5x saving is a concentration effect, and dependence erodes it
  smoothly. With the dependence modelled, the saving at cv = 0.9 is 2.0x at
  rho = 0.2, 1.8x at 0.3, 1.5x at 0.5 and vanishes at rho = 1, where the
  chance-constrained rule and the worst-case rule coincide. Most of the
  saving (three quarters at rho = 0.3) survives the weak correlation levels
  that a weather front across consecutive passes plausibly produces.
- What restores the SLA is to size on a model that contains the dependence.
  Knowing rho exactly is not required: assuming a moderate rho_plan = 0.3
  costs 16% more reserve than independence at cv = 0.5 (1.94 against 1.67)
  and protects every start up to rho = 0.3, still 26% below worst-case.

**Sentences for the paper.**
1. Sized for independent passes, the chance-constrained reserve kept the 95%
   SLA on the Fig. 6 sequence (cv 0.3 and 0.5) up to a 1.5x under-estimate of
   the fade CV and up to an inter-pass fade correlation of 0.2, and failed
   beyond (at cv = 0.5: 0.919 at 2x, 0.941 at rho = 0.3, 0.894 at
   rho = 0.5); at cv = 0.9 it already failed at rho = 0.2 (0.918).
2. The saving over worst-case sizing comes from the aggregate of independent
   fades concentrating, so it shrinks as passes fade together: with the
   correlation modelled, the 2.5x saving at cv = 0.9 becomes 2.0x at
   rho = 0.2, 1.5x at rho = 0.5, and 1.0x at rho = 1, where the two rules
   coincide.
3. A planner that assumes a moderate correlation (rho = 0.3) reserved 16%
   more than the independence plan at cv = 0.5, met the SLA in every
   starting pass up to rho = 0.3, and stayed 26% below the worst-case
   reserve.
4. The dependence is part of the law the planner must estimate; once it is,
   the chance-constrained rule degrades gracefully toward worst-case sizing
   and never undercuts it.

---

## 3. Strength and structure of inter-contact dependence (UC3)

**Question (R3.2, R2.6).** How sensitive are replication and the SLA to how
strongly, and in which pattern, contacts fail together? What does a planner
lose if it assumes the districts collapse independently?

**Setup.** `make_instance_dep` (in the script; `uc3_emergency.make_instance`
is untouched) builds UC3 with n_d in {2, 3, 4} districts and six relays split
3+3, 2+2+2, 2+2+1+1. District j's first hand-off opens in [20, 60],
[80, 140], [140, 200], [200, 260] s. Each district collapses with marginal
probability rho = 0.5. Coupling is a common-shock mixture: with probability
kappa the coupled districts share one collapse coin, otherwise each draws its
own, so marginals stay 0.5 and any coupled pair has
P[both] = rho^2 + kappa rho (1 - rho) (collapse correlation kappa). Two
structures: "common" (all districts coupled, exchangeable) and "pair" (only
the two fastest districts A and B coupled, the rest independent).
kappa in {0, 0.25, 0.5, 0.75, 1}, N = 2 copies, 200 instances per cell,
20 000 held-out joint draws per policy. `--check` asserts that n_d = 2,
kappa = 0 reproduces `make_instance` draw for draw (contacts and latent states
in the same order) and that marginals and P[A and B collapse] are exact in
every cell.

Policies: SCG greedy with the true latent model; SCG greedy on a latent model
that keeps the districts but assumes kappa = 0 ("SCG-mis"); the Bernoulli-TEG
corner (`bernoulli_select`); naive N fastest; stale single plan; epidemic.

**Key numbers** (held-out delivery ratio; CIs are at most +-0.011).

Two districts (common = pair):

| kappa | SCG | SCG-mis (predicts) | Bernoulli | naive | stale | epidemic |
|---|---|---|---|---|---|---|
| 0 | 0.651 | 0.651 (0.651) | 0.580 | 0.490 | 0.409 | 0.759 |
| 0.25 | 0.611 | 0.610 (0.651) | 0.556 | 0.490 | 0.409 | 0.700 |
| 0.5 | 0.570 | 0.571 (0.651) | 0.538 | 0.490 | 0.409 | 0.641 |
| 0.75 | 0.531 | 0.530 (0.651) | 0.516 | 0.490 | 0.409 | 0.582 |
| 1 | 0.490 | 0.490 (0.651) | 0.490 | 0.490 | 0.409 | 0.523 |

With 3 or 4 districts all coupled, SCG and SCG-mis give the same values as
the two-district row at every kappa (differences <= 0.001), and so does
naive. The Bernoulli corner gains a little (0.620 / 0.629 at kappa = 0,
0.490 at kappa = 1) and epidemic gains a lot (0.867 / 0.909 at kappa = 0,
0.524 at kappa = 1).

Only A and B coupled:

| kappa | SCG (3 or 4 districts) | SCG-mis, 3 / 4 districts | Bernoulli, 3 / 4 | naive | epidemic, 3 / 4 |
|---|---|---|---|---|---|
| 0 | 0.651 | 0.651 / 0.650 | 0.618 / 0.629 | 0.490 | 0.867 / 0.909 |
| 0.25 | 0.651 | 0.634 / 0.639 | 0.608 / 0.619 | 0.490 | 0.839 / 0.890 |
| 0.5 | 0.651 | 0.621 / 0.628 | 0.603 / 0.612 | 0.490 | 0.810 / 0.870 |
| 0.75 | 0.651 | 0.607 / 0.613 | 0.595 / 0.598 | 0.490 | 0.781 / 0.850 |
| 1 | 0.651 | 0.595 / 0.596 | 0.572 / 0.583 | 0.490 | 0.753 / 0.831 |

Paired loss of SCG-mis against SCG at kappa = 1: 0.056 +- 0.011 (3
districts), 0.055 +- 0.011 (4). With the pair coupled, SCG never puts both
copies in A and B (0% of instances for kappa > 0); SCG-mis does so in 28% to
41% of instances, which is the one-in-three tie among cross-district pairs it
cannot break.

**Reading.**
- Delivery here depends only on existence: every relay route meets the
  deadline when its contacts exist (Dirac timing), so each policy's value is a
  closed-form function of the copies it picks, and the instances matter only
  through tie-breaking. Two copies in two coupled districts deliver
  0.651 - 0.161 kappa, which is the SCG row; two copies in one district
  deliver 0.490 at any kappa, which is naive.
- Dependence strength erodes what replication can buy. Under a shock shared
  by all districts, the value of correlation-aware replication over naive
  falls linearly from 0.161 at kappa = 0 to zero at kappa = 1, for any number
  of districts: when every district fails together there is nothing to
  diversify across, and no copy budget of 2 helps.
- Under an exchangeable shock, assuming independence costs nothing in the
  choice: SCG-mis picks equally good copies. It costs in what the planner
  promises. Its predicted delivery stays at 0.651 while the truth falls to
  0.490, an over-confidence of 0.161 kappa, which is the quantity an SLA
  admission on that prediction would get wrong.
- Structure matters as much as strength. When only some districts are
  coupled, the correlation-aware planner keeps its full 0.651 by placing the
  second copy outside the coupled pair, while the planner that assumes
  independence loses up to 0.056 and over-states its delivery by the same
  amount. The Bernoulli corner, blind to both existence correlation and joint
  structure, sits below both (0.572 to 0.583 at kappa = 1).
- More districts help the unbounded epidemic reference and, slightly, the
  Bernoulli corner, whose disjoint picks land in different districts more
  often by chance (56% of instances with 2 districts, 86% with 4). They do
  not help the correlation-aware planner, which already reaches the two-copy
  optimum with two districts.

**Sentences for the paper.**
1. We coupled the districts' collapses through a common shock of strength
   kappa (collapse correlation kappa, marginals unchanged) and varied the
   number of districts from two to four.
2. When all districts share the shock, the gain of correlation-aware
   replication over the naive policy falls linearly from 0.16 to zero as
   kappa goes to one, and a planner that assumes independent districts still
   chooses the same copies but over-states its delivery ratio by 0.16 kappa.
3. When only the two fastest districts are coupled, the correlation-aware
   planner keeps 0.651 at every kappa by placing the second copy elsewhere,
   while the planner that assumes independence loses up to 0.056.
4. The value of modelling dependence thus lies both in the routes it picks
   when the dependence is structured and in the delivery probability it
   reports, which is what the SLA admission relies on.

---

## 4. Learning the timing law from observed windows (UC1)

**Question (R2.10).** Learning is described as a drop-in extension. What
happens when the planner estimates the law from data, and how much data does
it need?

**Setup.** The truth is the UC1 instance itself (TruncNormal jitter,
+-3.5 sigma_true). The planner observes n i.i.d. air-window start offsets
(realised minus nominal start, pooled over air contacts), fits the mean
offset and sigma, builds its belief graph (TruncNormal(nominal + mu_hat,
sigma_hat), +-3.5 sigma_hat), generates candidates on that belief and routes
with `pick_route_chance` (eps = 0.05, lcb admission). Two fits: plug-in
(sample std) and a 90% chi-square upper confidence bound on sigma,
s * sqrt((n - 1) / chi2_{0.10, n-1}); the oracle plans on the true law.
n in {3, 5, 10, 20, 50, 200}, nested training sets per instance (an operator
accumulating windows), 300 instances (so 300 independent training sets per
n), 20 000 held-out truth draws per route.

sigma_true = 40 s is the main case. At eps = 0.05 the oracle never admits an
air route (as in `results/eps.csv`), so any air admission is a learning error.
sigma_true = 20 s was added because there the oracle does admit air routes,
which exposes the delivery-time side of the trade.

**Key numbers.**

sigma_true = 40 s (oracle: violation 0, delivery time 556 s):

| n | plug-in: violation, instances > eps, time | 90% upper bound: violation, instances > eps, time | P[sigma_hat < sigma], plug-in / bound |
|---|---|---|---|
| 3 | 0.050, 42%, 352 s | 0.0037, 3.0%, 540 s | 0.63 / 0.07 |
| 5 | 0.035, 32%, 410 s | 0.0039, 4.0%, 537 s | 0.57 / 0.09 |
| 10 | 0.021, 20%, 458 s | 0.0021, 2.0%, 546 s | 0.54 / 0.11 |
| 20 | 0.010, 11%, 498 s | 0.0018, 1.7%, 550 s | 0.53 / 0.08 |
| 50 | 0.003, 3.7%, 541 s | 0.0007, 0.7%, 553 s | 0.57 / 0.10 |
| 200 | 0, 0%, 556 s | 0, 0%, 556 s | 0.55 / 0.13 |

Every plug-in air admission at sigma_true = 40 s is a breach (the share of
instances above eps equals the share routed on air), and all of them are
significant breaches; the breached routes violate 12% on average and up to
23%.

sigma_true = 20 s (oracle: violation 0.0095, delivery time 77 s, all on air):

| n | plug-in: violation, time | 90% upper bound: violation, time |
|---|---|---|
| 3 | 0.0089, 137 s | 0.0018, 429 s |
| 5 | 0.0084, 95 s | 0.0018, 381 s |
| 10 | 0.0088, 93 s | 0.0036, 255 s |
| 20 | 0.0088, 79 s | 0.0048, 164 s |
| 50 | 0.0090, 77 s | 0.0069, 87 s |
| 200 | 0.0096, 77 s | 0.0086, 78 s |

No instance exceeds eps at sigma_true = 20 s for either fit (worst instance
0.049).

**Reading.**
- The plug-in fit falls below the true sigma in 53% to 63% of training sets
  (the sample variance is unbiased, its square root slightly low), and an
  under-estimate is exactly the error a chance constraint cannot absorb. At sigma_true = 40 s and n = 3 the plug-in planner
  breaches the 5% budget in 42% of instances, with a mean violation at the
  budget itself (0.050); it needs about 50 windows to bring breaches under
  4%, and 200 to remove them.
- Planning on a 90% upper confidence bound for sigma caps the breach rate at
  4% at any n, even from 3 windows, below the 10% non-coverage the bound is
  designed for, because a mild under-estimate rarely flips the route. Its
  delivery times stay within 19 s of the oracle's at sigma_true = 40 s.
- The bound has a price when the oracle would take the fast route
  (sigma_true = 20 s): the conservative planner falls back on the backbone in
  three quarters of the instances at n = 3 (429 s against 77 s), the gap
  closes to 10 s at n = 50, and to 1 s at n = 200. At that jitter level the
  plug-in fit is harmless (no breach), so the cost of caution is paid with
  nothing to show for it when data is scarce and the law is benign.
- The learning loop works as a drop-in: the router consumes the fitted law
  unchanged. What it needs is a confidence-aware fit: the SLA admission is
  only as good as the variance estimate it rests on.

**Sentences for the paper.**
1. We let the planner estimate the jitter law from n observed air windows
   and route on the fitted law, scoring every choice on the true law.
2. With a plug-in estimate at sigma_true = 40 s, 42% of instances breached the
   5% budget after 3 windows, 20% after 10 and none after 200, since an
   under-estimated variance makes risky air routes look admissible.
3. Fitting a 90% chi-square upper bound on sigma kept breaches at or below 4%
   at every n, with delivery times within 19 s of the oracle.
4. When the true jitter is small (20 s) the same bound delays delivery while
   data is scarce (429 s against 77 s at n = 3) and comes within 10 s of the
   oracle by n = 50, which quantifies the price of learning conservatively.

---

## Caveats (all studies)

- Study 1 runs at a positive deadline margin, so the SLA-aware routers can
  always escape to the backbone; at sigma = 40 s that is what they do, which
  makes them immune to the tail. A zero or negative margin, where every policy
  must gamble on jittered contacts, was not run.
- Study 1 truncates the t truth at +-12 sigma and re-solves its scale so the
  truncated standard deviation equals sigma. With `matched_std` as such the
  heavy-tailed truth would be narrower (0.85 sigma at nu = 2.5) and the tail
  effect would look smaller than it is.
- Study 2's Fig. 6 sequence is favourable. Its integer reserve sizes leave
  headroom (in-sample SLA 0.997 at cv 0.3), so the per-start table is the
  fairer view of fragility.
- Study 2's dependence-aware planner sizes on the exact lower bound; the
  belief plans in the "paper" block use `provisioning_cpd`'s point-estimate
  rule unchanged, so those numbers line up with Fig. 6.
- Study 3's UC3 instance has point-mass timing, so its delivery ratios are
  closed-form in the selected copies. The 200 instances per cell mostly vary
  tie-breaking. This keeps the study exact but leaves out timing and existence
  uncertainty interacting.
- Study 4 learns one pooled jitter law shared by all air contacts, from
  i.i.d. observations, in the correct parametric family. Per-contact laws,
  non-stationary observations and a misspecified family (study 1 combined
  with study 4) were not tested. Only sigma is bounded; the mean offset is a
  plug-in estimate, which is harmless here because a common shift moves both
  hops of an air route together.
