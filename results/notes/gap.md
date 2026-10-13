# Optimality gap of candidate generation, and the infeasibility policy

Answers R1 ("zero theoretical bounding or empirical proof regarding the
Optimality Gap of the candidate generation step"), R2 ("report how often the
selected route is the Monte-Carlo-best route, and state the policy when no
candidate meets the chance constraint") and R3 ("evidence that the
candidate-generation procedure does not systematically miss better routes").

Short answer. On UC1 the candidate set always contained the best route, and
the router picked it (or a statistical tie) in 96 to 100% of instances at the
paper's margin, with every miss below 0.0021 of probability. On UC4 the default
generator does miss better routes, systematically when the deadline leaves room
for the slow timing-certain access link: at a 60 s margin the best route was in
the n = 40 candidate set in 25% of instances and the mean held-out gap was
0.081. The cause is identifiable, and one extra generator step (mean-plan
routes with a whole class of uncertain contacts excluded) brings the hit rate
to 97% and the mean gap to 0.0004 at the same cost. Infeasibility under the
chance constraint is frequent on the contested settings; most of it is
intrinsic (no enumerated route meets the budget either), and the rest comes
from the same generator blind spot or from the admission test's sample size.

## What was implemented

| File | Content |
|---|---|
| `scgraph/routing/enumerate.py` | `envelope_contact`, `envelope_arrival`: the support envelope of a contact and a route. `enumerate_routes`: every node-simple route whose envelope arrival meets the deadline, with a hop bound, a route cap, an optional time budget, and the flags that say whether a cap was hit. `screen_routes`: scores many routes on common random numbers along a prefix tree, draw for draw equal to `arrival.joint_route_set_deliveries`. |
| `tests/test_enumerate.py` | 36 tests (see below). |
| `experiments/run_gap.py` | The experiment (`--quick`, `--only uc1/uc4`, `--replot`). |
| `results/gap_uc1.csv`, `results/gap_uc4.csv` | One row per instance (selection, oracle, paired held-out comparison, chance-constrained runs, diagnostics, timings). |
| `results/gap_uc1_agg.csv`, `results/gap_uc4_agg.csv` | One row per (setting, candidate budget): rates, gap quantiles, infeasibility, enumeration size, wall time. |
| `figures/gap.{pdf,png}` | (a) hit rate and (b) held-out gap against the candidate budget; (c) UC4: fraction of instances where a tighter hop bound would lose the oracle. |

No shared file was modified. `.venv/bin/pytest -q`: all tests pass (see the
end of this note for the count at the time of writing).

## The reference set: exhaustive enumeration under the support envelope

Each stochastic contact is replaced by its most favourable corner:

- earliest start `s_lo`, the lower end of the start law's support;
- latest end `e_hi`, the start's upper end plus the duration's upper end
  (`StochasticContact.support_window`, with the sampler's clip of the duration
  at zero);
- shortest transfer `size / r_hi`, with `r_hi` the upper end of the rate
  support, or its `rate_q` quantile when the support is unbounded (lognormal);
  `rate_q = 1` keeps the true supremum (transfer time 0);
- the contact is usable iff its existence probability is positive under some
  latent state.

The arrival recursion, eqs. (2) to (4), is monotone: a later arrival at the
sender, a later start, a lower rate or an earlier end can only delay delivery
or break feasibility. By induction along a route, the realized arrival is never
earlier than the envelope arrival, and a transfer that fits its realized window
also fits the envelope window. So:

> If some realization delivers route pi by D, the envelope delivers pi by D,
> and `enumerate_routes` returns pi, provided pi is node-simple, has at most
> `hop_bound` contacts, and the route cap was not hit.

With `rate_q < 1` this holds up to the truncated rate tail: a route the
envelope excludes can only be on time if one of its k unbounded-rate contacts
draws a rate above its `rate_q` quantile, so its success probability is at most
k (1 - rate_q), whatever the dependence between contacts (union bound on the
marginals). The experiment uses `rate_q = 0.999`; UC4 routes carry two to four
lognormal-rate contacts (feeder, HAPS uplink, UAV), so an excluded route has
P <= 0.004. UC1 has no unbounded laws.

Node-simple routes suffice: cutting a cycle out of a route gives a route that
reaches the repeated node earlier and needs a subset of the contacts, so it is
on time in every realization where the cyclic route is.

The search is a depth-first extension on the envelope, pruned by two bounds
that cannot cut a route the set should contain: the static hop distance to the
destination, and a latest-arrival bound (the latest envelope arrival at a node
from which the destination is still reachable by D within r more hops,
computed backwards over walks). The hop bound is the shortest envelope-feasible
hop count plus `hop_slack`, capped by `max_hops`; when it reaches the number of
nodes minus one it cannot bind (`hop_bound_free`).

Tests (`tests/test_enumerate.py`, 36 cases, about 9 s):

- point-mass graphs (12 random plans, 3 deadlines each): the enumeration equals
  the brute-force set of simple routes whose `delivery_time` meets D;
- stochastic graphs with jittered starts, random durations, lognormal and
  uniform rates, uncertain existence, a copula and a shared factor (10 plans):
  the enumeration equals brute force under the envelope, with and without the
  latest-arrival prune, and respects every hop bound;
- soundness (8 plans): every simple route that any of 300 sampled realizations,
  or any of 2000 vectorised joint draws, delivers by D is enumerated;
- the fixtures are non-vacuous (dozens of on-time routes, and the envelope set
  strictly larger than what the draws exhibit);
- `rate_q = 0.999` gives a subset of `rate_q = 1`; the route cap is reported
  and returns a subset; unreachable destinations and latent existence;
- `screen_routes` equals `joint_route_set_deliveries` draw for draw (on-time
  matrix, counts and mean delivery) with a copula, a factor and a latent state;
- on UC1 instances, every candidate route that delivers on time in some draw is
  in the enumerated set.

## Protocol

Per instance, eight independent streams are spawned from the task seed
(`_parallel.pmap`, 3 workers): instance, candidates, selection,
chance-constrained selection, three screening stages, held-out evaluation.

1. **Selection.** `candidate_routes` at n_realizations in {0, 10, 40, 160}
   (40 is the paper's default), each generated from the same seed, so the four
   sets are nested. `pick_route` (argmax P[T <= D], M = 2000 draws per
   candidate, as in the paper) over each set. A fifth variant, **40+RC**
   (risk-class exclusion), adds to the n = 40 set the mean-plan CGR routes
   computed with a whole class of uncertain contacts excluded (contacts with
   p_exist < 1; contacts with random timing; both), plus their single-contact
   exclusions. It lives in `run_gap.py` only (`risk_class_routes`); the library
   is unchanged.
2. **Oracle.** Pool = every enumerated route plus every candidate route of every
   variant (so the oracle can never score below a candidate). Screening funnel
   on common random numbers: all routes on 1000 draws; the best 2000 on 5000
   fresh draws; the best 200 on 20000 fresh draws. Candidates, selected routes
   and chance-constrained routes always reach the last stage, as do the best
   routes within each hop level used for the hop-bound diagnostic. Oracle =
   argmax of the last stage (ties to the lower mean delivery, as in
   `pick_route`). Enumeration: UC1 `hop_slack = 99` (the bound cannot bind),
   UC4 `hop_slack = 24`; both `max_routes = 200k/500k`, `rate_q = 0.999`.
3. **Held-out comparison.** The selected routes, the chance-constrained routes
   and the oracle are evaluated on the same 20000 fresh paired draws
   (`joint_route_set_deliveries`). Per budget:
   - *identical*: same contact ids;
   - *tie*: different routes, exact McNemar test on the discordant pairs not
     significant at 95% (two-sided binomial test);
   - *hit* = identical or tie (the answer to R2's "how often");
   - *significantly worse*: McNemar significant and gap > 0; *material*:
     significantly worse with a gap above 0.01;
   - *gap* = P_oracle - P_selected on the paired draws, with a paired 95% CI;
   - *oracle in set*: the oracle's contact ids are among the candidates;
   - decomposition on the stage-3 draws: generation gap = P(oracle) - P(best
     candidate); ranking gap = P(best candidate) - P(selected).
4. **Chance constraint.** `pick_route_chance` with `admission="lcb"`,
   delta = 0.05, M = `stats.admission_draws(eps, |candidates|, delta)` (2000 in
   every call here, the function's floor; at most 37 candidates), at eps in {0.01, 0.05, 0.10}, for every candidate variant. When it
   reports `feasible=False` we ask whether the enumerated set holds a route that
   clears 1 - eps: the oracle's Clopper-Pearson lower bound on its 20000 stage-3
   draws, Bonferroni-corrected over the stage (about 250 routes), must reach
   1 - eps. The same test is applied to every member of the candidate set
   ("a candidate is certifiably feasible at 20k draws"), and the oracle's
   held-out estimate is recorded too.

Settings:

- UC1, `make_instance_cpd`, sigma in {20, 40, 60} s at the default margin
  (40 s), 40 instances each; plus a contested block, sigma = 40 s, margin
  -30 s (the backbone misses the deadline, only the air relays can make it),
  40 instances.
- UC4, `uc4_continuum.make_instance`, sigma = 40 s, margin in {0, 60} s
  (deadline relative to the reliable path, the scenario's default mode), far
  terminal, 40 instances each. The brief said about 30; the run was cheap
  enough to use 40.

## How to run

```bash
.venv/bin/python experiments/run_gap.py --quick     # 4 instances per setting, inline, about 65 s
.venv/bin/python experiments/run_gap.py             # full run, 3 workers
.venv/bin/python experiments/run_gap.py --only uc4  # one scenario
.venv/bin/python experiments/run_gap.py --replot    # re-aggregate and redraw from the CSVs
```

`--quick` writes `*_quick.csv` and `figures/gap_quick.*` and never touches the
full outputs.

Wall time of the full run: **322 s** (UC1 20 s, UC4 301 s), 949 s CPU, on 3
worker processes, Apple M3 Pro shared with other jobs (load average 5 to 8).
Per UC4 instance: 10 to 12 s in total, of which candidate generation for the
four budgets 6.1 s (mostly the 160 realizations), selection 0.3 s,
chance-constrained selection 0.9 s, enumeration 0.4 to 0.9 s on average (2.3 s max),
screening 1.9 to 3.1 s on average (8.3 s max). Per UC1 instance: 0.35 s.

## Results

### Enumeration

| Setting | routes enumerated (median / max) | shortest feasible hops | hop bound | cap hit | oracle hops (mean / max slack over shortest) | oracle's rank in screens 1 / 2 (max) |
|---|---|---|---|---|---|---|
| UC1, margin 40 s | 4 / 5 | 2 | free (all simple routes) | 0 / 120 | 2.0 / 0 | 1 / 0 |
| UC1, margin -30 s | 3 / 3 | 2 | free | 0 / 40 | 2.0 / 0 | 2 / 1 |
| UC4, margin 60 s | 49 949 / 187 984 | 3.8 | shortest + 24 | 0 / 40 | 6.8 / 13 | 2 / 5 |
| UC4, margin 0 s | 24 681 / 64 233 | 3.9 | shortest + 24 | 0 / 40 | 8.1 / 10 | 3 / 2 |

UC1's envelope admits only the three air routes and one or two backbone routes;
the other 13 to 14 candidate routes per instance (later passes) can never meet
D. In UC4 the oracle was always an enumerated route, never a candidate beyond
the hop bound. The funnel's cut (2000, then 200) was never close to deciding
the oracle: its rank in the first screen was at most 3. On two UC4 instances a
flat 5000-draw screen of every enumerated route (15 766 and 50 684 routes, 1.8
and 5.1 s) put a route with the oracle's link sequence and hop count on top.

UC4 hop bound (panel c of the figure): restricting the oracle to routes of at
most shortest + s hops loses more than 0.005 of probability in this fraction of
instances, with the mean loss in brackets.

| s | 0 | 2 | 4 | 6 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| margin 60 s | 0.98 (0.536) | 0.38 (0.064) | 0.15 (0.007) | 0.05 (0.002) | 0.03 (0.001) | 0 | 0 | 0 |
| margin 0 s | 1.00 (0.450) | 0.58 (0.091) | 0.40 (0.019) | 0.15 (0.004) | 0.10 (0.002) | 0 | 0 | 0 |

The run used s = 24, so the oracle is stable well inside the bound. Routes with
P > 0 exist at every depth (ISL detours are cheap), so the bound always binds
for the count of routes, which is why it is reported as a convergence result.

### Selected against oracle (held-out, 20000 paired draws)

UC1 at the default 40 s margin, sigma = 40 and 60 s: identical routes in 40/40
instances at every budget, gap 0, P = 1.0000 for both. Sigma = 20 s and the
contested block:

| Setting | n | candidates | hit | identical | sig. worse | material (> 0.01) | oracle in set | gap mean (95% CI) | gap q90 | gap max | P selected | P oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| UC1 sigma 20, margin 40 | 0 | 17.4 | 0.88 | 0.88 | 0.12 | 0.00 | 1.00 | +0.0001 (0.0001) | +0.0003 | +0.0021 | 0.9999 | 1.0000 |
| | 10 | 17.4 | 0.97 | 0.97 | 0.03 | 0.00 | 1.00 | +0.0000 | +0.0000 | +0.0009 | 1.0000 | 1.0000 |
| | 40 | 17.4 | 0.90 | 0.90 | 0.10 | 0.00 | 1.00 | +0.0001 (0.0001) | +0.0000 | +0.0011 | 0.9999 | 1.0000 |
| | 160 | 17.4 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0 | 0 | 0 | 1.0000 | 1.0000 |
| | 40+RC | 17.9 | 0.85 | 0.85 | 0.15 | 0.00 | 1.00 | +0.0001 (0.0001) | +0.0008 | +0.0015 | 0.9999 | 1.0000 |
| UC1 sigma 40, margin -30 | 0 | 15.3 | 0.95 | 0.95 | 0.03 | 0.03 | 1.00 | +0.0001 (0.0006) | +0.0000 | +0.0111 | 0.9048 | 0.9049 |
| | 10 | 15.3 | 0.95 | 0.93 | 0.05 | 0.03 | 1.00 | +0.0005 (0.0006) | +0.0000 | +0.0111 | 0.9044 | 0.9049 |
| | 40 | 15.3 | 0.88 | 0.88 | 0.10 | 0.05 | 1.00 | +0.0011 (0.0016) | +0.0006 | +0.0295 | 0.9038 | 0.9049 |
| | 160 | 15.3 | 0.88 | 0.88 | 0.10 | 0.03 | 1.00 | +0.0008 (0.0010) | +0.0006 | +0.0159 | 0.9042 | 0.9049 |
| | 40+RC | 15.6 | 0.93 | 0.93 | 0.05 | 0.03 | 1.00 | +0.0004 (0.0009) | +0.0000 | +0.0169 | 0.9045 | 0.9049 |
| UC4 margin 60 | 0 | 8.0 | 0.07 | 0.05 | 0.93 | 0.93 | 0.05 | +0.1368 (0.0198) | +0.2257 | +0.3055 | 0.7718 | 0.9086 |
| | 10 | 12.0 | 0.17 | 0.12 | 0.82 | 0.82 | 0.12 | +0.1081 (0.0209) | +0.1628 | +0.2742 | 0.8005 | 0.9086 |
| | 40 | 16.6 | 0.28 | 0.25 | 0.72 | 0.70 | 0.25 | +0.0812 (0.0185) | +0.1361 | +0.2046 | 0.8275 | 0.9086 |
| | 160 | 22.4 | 0.50 | 0.47 | 0.50 | 0.47 | 0.47 | +0.0502 (0.0199) | +0.1361 | +0.2248 | 0.8584 | 0.9086 |
| | 40+RC | 23.4 | 0.97 | 0.88 | 0.03 | 0.03 | 0.95 | +0.0004 (0.0006) | +0.0000 | +0.0105 | 0.9083 | 0.9086 |
| UC4 margin 0 | 0 | 7.9 | 0.50 | 0.38 | 0.50 | 0.42 | 0.53 | +0.0158 (0.0106) | +0.0336 | +0.2029 | 0.7545 | 0.7703 |
| | 10 | 11.6 | 0.57 | 0.45 | 0.42 | 0.38 | 0.65 | +0.0116 (0.0059) | +0.0336 | +0.0790 | 0.7587 | 0.7703 |
| | 40 | 15.2 | 0.72 | 0.62 | 0.25 | 0.20 | 0.78 | +0.0074 (0.0056) | +0.0274 | +0.0790 | 0.7629 | 0.7703 |
| | 160 | 19.5 | 0.75 | 0.60 | 0.25 | 0.23 | 0.82 | +0.0063 (0.0043) | +0.0272 | +0.0602 | 0.7641 | 0.7703 |
| | 40+RC | 21.9 | 0.62 | 0.47 | 0.35 | 0.25 | 0.78 | +0.0087 (0.0057) | +0.0321 | +0.0790 | 0.7617 | 0.7703 |

Pooled over sigma at the 40 s margin (120 UC1 instances), the hit rate is
0.96 / 0.99 / 0.97 / 1.00 at n = 0 / 10 / 40 / 160, with no material miss.

### Where the UC4 misses come from

Misses split into generation (the oracle is outside the candidate set) and
ranking (it is inside, and the M = 2000 selection picked another route).

| Setting | n | misses (generation / ranking) | mean generation gap | mean ranking gap | router time per instance |
|---|---|---|---|---|---|
| UC4 margin 60 | 0 | 37 (37 / 0) | 0.135 | 0.001 | 0.08 s |
| | 10 | 33 (33 / 0) | 0.107 | 0.001 | 0.38 s |
| | 40 | 29 (29 / 0) | 0.080 | 0.001 | 1.25 s |
| | 160 | 20 (20 / 0) | 0.049 | 0.002 | 4.69 s |
| | 40+RC | 1 (0 / 1) | 0.000 | 0.000 | 1.31 s |
| UC4 margin 0 | 0 | 20 (16 / 4) | 0.015 | 0.001 | 0.08 s |
| | 10 | 17 (11 / 6) | 0.008 | 0.004 | 0.37 s |
| | 40 | 11 (6 / 5) | 0.007 | 0.001 | 1.22 s |
| | 160 | 10 (6 / 4) | 0.005 | 0.002 | 4.55 s |
| | 40+RC | 15 (6 / 9) | 0.007 | 0.003 | 1.27 s |

Margin 60 s. The oracle starts on the narrowband terminal-to-HAPS link (the
timing-certain fallback) in 37 of 40 instances, then a HAPS uplink, a few ISLs
and a feeder pass; its held-out success is 0.92 in the median instance. The
selected route at n = 40 starts on a UAV hand-off in 30 of 40 instances. Two
mechanisms keep the safe route out of the candidate set:

- Earliest-arrival CGR on a realization never prefers a 200 s transfer while any
  UAV relay works, so sampled realizations propose it only when all three UAV
  paths fail in that draw (it enters slowly with n: in the set for 5%, 12%, 25%,
  47% of instances at n = 0, 10, 40, 160).
- The contact-suppression sweep does reach the narrowband link, but only after
  it has suppressed the HAPS uplink and feeder pass that the fast routes share
  with the safe one. On six instances checked, the direct-link route the sweep
  produces arrives 750 to 1625 s after release against deadlines of 322 to
  361 s (P = 0), while the best risk-class route through the same link has P
  of 0.91 to 0.93. In UC1 the backbone shares no contact with the air relays,
  which is why the sweep works there.

Margin 0 s. The oracle uses a UAV in 40 of 40 instances (no timing-certain
route meets the deadline), so risk-class exclusion has nothing to add; its
extra candidates only add near-ties to the ranking (ranking misses 9 against 5,
within selection noise). The generation misses here are mostly routes that
bounce through a second platform (satellite down to another HAPS and up again)
before the feeder, which earliest-arrival generation proposes only rarely: the
oracle bounces in all 6 generation misses at n = 40, which include the largest
gaps (0.079, 0.064, 0.041), and in 8 of the 40 margin-0 instances overall. Note for the scenario: `uc4_continuum` keeps downlinks as
point masses on the stated assumption that no gateway-bound bundle uses them;
the enumeration finds that they are useful as relay hops.

Oracle resolution. In 6 of 1200 comparisons (two instances) the selected route
measured significantly better than the oracle on the held-out draws, by 0.0065
(UC1, margin -30 s) and 0.0106 (UC4, margin 0 s). The oracle is the best
enumerated route up to about 0.01 of probability, which is the funnel's
resolution among near-tied routes.

### Infeasibility under the chance constraint (lower-bound admission)

UC1 at the 40 s margin: never infeasible (0 of 120 at every eps and budget;
the backbone has P = 1). Elsewhere, at the paper's n = 40:

| Setting | eps | router infeasible | of those: an enumerated route clears 1 - eps | of those: a candidate clears it at 20k draws | fallback = oracle | P fallback / P oracle (held out) |
|---|---|---|---|---|---|---|
| UC1 sigma 40, margin -30 | 0.01 | 40/40 | 0 | 0 | 0.85 | 0.904 / 0.905 |
| | 0.05 | 40/40 | 0 | 0 | 0.97 | 0.905 / 0.905 |
| | 0.10 | 32/40 | 12 | 12 | 0.88 | 0.898 / 0.899 |
| UC4 margin 60 | 0.01 | 40/40 | 0 | 0 | 0.25 | 0.828 / 0.909 |
| | 0.05 | 40/40 | 0 | 0 | 0.25 | 0.828 / 0.909 |
| | 0.10 | 33/40 | 28 | 1 | 0.12 | 0.809 / 0.906 |
| UC4 margin 60, n = 160 | 0.10 | 29/40 | 24 | 8 | 0.31 | 0.836 / 0.903 |
| UC4 margin 60, 40+RC | 0.10 | 17/40 | 12 | 12 | 0.82 | 0.889 / 0.889 |
| UC4 margin 0 | 0.01 to 0.10 | 40/40 | 0 | 0 | 0.55 to 0.60 | 0.762 / 0.770 |

(The 40+RC row at eps = 0.01 and 0.05 is 40/40 with no feasible enumerated
route.) Whenever the router admitted a route, its held-out success was at least
1 - eps: zero admitted routes over budget in every setting.

Three kinds of infeasibility show up:

- **Intrinsic**: no enumerated route clears the budget. All of UC4 at 0 s
  (best route 0.66 to 0.83), UC4 at 60 s for eps <= 0.05 (best 0.77 to 0.93),
  UC1 contested for eps <= 0.05 (best 0.86 to 0.95). Here `feasible=False` is
  the right answer, and the fallback route is on average within 0.01 of the
  oracle, except where generation misses (UC4 at 60 s: 0.83 against 0.91).
- **Generation**: the enumerated set has a feasible route that the candidate
  set lacks. UC4 at 60 s, eps = 0.10: 27 of 33 infeasible instances at n = 40.
  Risk-class exclusion removes this class (40+RC: in all 12 remaining cases a
  candidate is certifiably feasible).
- **Admission power**: a candidate clears 1 - eps at 20000 draws, but the
  Bonferroni lower bound on 2000 draws does not (true P within about 0.01 of
  the budget). UC1 contested, eps = 0.10: all 12 such cases; UC4 40+RC, eps =
  0.10: all 12. More admission draws would admit them.

## Reading

**R2, how often is the selected route the Monte-Carlo best?** On UC1 at the
paper's margin, 96 to 100% of instances (identical route in every hit), and the
misses are immaterial: at sigma = 20 s the router occasionally prefers an air
route whose held-out success is 0.998 to 0.9997 over the backbone's 1.0000,
a ranking effect of the 2000-draw selection between near-certain routes. On the
contested UC1 margin, 88 to 95%, again ranking noise among three air routes
with P near 0.90 (largest gap 0.030). On UC4 the answer depends on the margin:
28% at 60 s and 72% at 0 s for n = 40.

**R1 and R3, does candidate generation systematically miss better routes?** On
UC1, no: the best route was in the candidate set in all 160 instances at every
budget, including n = 0, because the envelope admits only three to five routes
and the generator proposes all of them. On UC4, yes, and the measurement points
at the mechanism. With room in the deadline, the best route uses a slow,
timing-certain access link that a realization's earliest-arrival route
rarely uses, and that the contact-suppression sweep reaches only after removing
the contacts it shares with the fast routes. More sampled realizations help
slowly (mean gap 0.137, 0.108, 0.081, 0.050 at n = 0, 10, 40, 160), at four
times the cost per step. Adding the mean-plan routes with a class of uncertain
contacts excluded closes the gap (hit 0.97, mean gap 0.0004, largest 0.0105) at
no measurable extra cost (1.31 s against 1.25 s per instance). At a 0 s margin
no timing-certain route meets the deadline, the residual gap is small on
average (0.007 at n = 40) but reaches 0.079, and it comes half from generation
(platform-bounce routes) and half from ranking noise.

**Cost of the oracle itself.** On UC4 the enumeration plus three-stage screening
took 2.3 to 4.0 s per instance on average (10.6 s at most), against 1.25 s for
the n = 40 router, and its held-out success is not optimistic (fresh draws).
At this plan size it is a deployable planning-plane router. It does not scale
the way candidate ranking does: the enumerated set grows by about 1.25 to
1.45x per extra hop of slack (6 500 to 188 000 routes at slack 24 here), and
with longer deadlines or denser meshes it would need the caps it reports.

**What this does and does not show.** The oracle is the best route among those
the support envelope allows within shortest + 24 hops, estimated to about 0.01
of probability. It is a route-level optimum: adaptive policies that change
route en route (the contingent policies of `run_uc4.py`) are outside its scope.
Two scenarios, one sigma for UC4, and the reliable-path deadline mode of UC4
(the one that anchors the deadline on the timing-certain path, which is
precisely the case where the default generator does worst); the nominal mode
was not run. The 40+RC variant is a measured suggestion and lives in the
runner; adopting it would mean adding it to `candidate_routes`, which changes
the published RNG streams and so needs a re-run of the paper's figures.

## The infeasibility policy, stated precisely

`pick_route_chance` scores every candidate on M draws, M sized from eps and the
candidate count by `stats.admission_draws`. A candidate is admissible when the
one-sided Clopper-Pearson lower bound on its P[T <= D], at level
delta / |candidates| (Bonferroni, delta = 0.05), is at least 1 - eps; under
the model, the probability that any admitted route truly violates the budget is
then at most delta. If at least one candidate is admissible, the router returns
the admissible candidate with the lowest mean delivery time and `feasible=True`.
If none is, the request is declared infeasible at budget eps: the router returns
the candidate with the highest estimated P[T <= D] as a best-effort route,
flagged `feasible=False`, and leaves the decision (forward best effort, refuse
the bundle, relax eps or the deadline, or replicate) to the caller.

## Sentences the paper could use

1. To measure what ranking a candidate set costs, we enumerated every
   node-simple route that the support envelope of the laws lets arrive by the
   deadline, a set that provably contains every route some realization
   delivers on time, scored all of them on common random numbers, and compared
   the router's choice with the best enumerated route on 20 000 fresh paired
   draws.
2. On UC1 the best route was in the candidate set in all 160 instances, and the
   router selected it in 96 to 100% of instances at the default margin, every
   miss costing less than 0.0021 of delivery probability; on UC4 with a 60 s
   margin, the default generator held the best route in only 25% of instances
   (mean gap 0.081), because the best route rides a slow, timing-certain access
   link that a realization's earliest-arrival route rarely uses and that contact
   suppression reaches only after removing the uplink and feeder it shares with
   the fast routes.
3. Adding the nominal-plan routes computed with each class of uncertain
   contacts excluded raises the UC4 hit rate to 97% and cuts the mean gap to
   0.0004 at no measurable cost, while at a 0 s margin, where the best route
   always relies on a UAV hand-off, the residual gap averages 0.007 (at most
   0.079).
4. When no candidate's Bonferroni-corrected Clopper-Pearson lower bound on
   P[T <= D] reaches 1 - eps, the router declares the request infeasible at that
   budget, returns the candidate with the highest estimated delivery probability
   as a flagged best-effort route, and leaves admission, renegotiation or
   replication to the caller; in our runs 305 of the 345 such declarations at
   the default candidate budget were intrinsic (no enumerated route met the
   budget either), and no admitted route exceeded its budget on held-out draws.

## Test count

`.venv/bin/pytest -q` at the time of writing: 163 passed (36 of them in `tests/test_enumerate.py`), 13 s.
