# BRUF / RUCoP against the stochastic contact graph (revision T1, R2.4)

Reviewer 2 asked for "a comparison table and at least one quantitative
comparison with BRUF/RUCoP, probabilistic CGR, or a related satellite-UAV MDP
router". This note documents a faithful reimplementation of BRUF and RUCoP
inside `scgraph`, its validation, and a paired comparison on UC1 and UC3 plus a
tractability study.

## Headline

1. **Timing uncertainty (UC1): BRUF loses to the SLA-aware router at
   σ ≥ 30 s, and its contingency adds nothing measurable.** (At σ = 15 s both
   are near zero; BRUF's 0 against 0.03% comes at 2.7 times the latency.) At the same slot width, BRUF-1 and
   the static Bernoulli path are statistically indistinguishable (paired
   differences within ±0.3 points). At σ = 60 s the SLA-aware router and its
   contingent table record 0 violations in 80 000 held-out runs (95% CP upper
   bound 4.6e-5). BRUF-1 records 0.55% at Δ = 10 s, 3.3% at Δ = 30 s and 9.2%
   at Δ = 60 s. Deterministic CGR records 29.2%.
2. **Correlated existence failures (UC3): RUCoP-2 beats every static route set,
   ours included.** At ρ = 0.5: RUCoP-2 73.3%, SCG correlation-aware 65.0%,
   Bernoulli-select 60.2%, naive 48.9%. The oracle that sees the realization
   reaches 75.8%. The gain comes from contingency (acknowledged retries after an
   observed failure). RUCoP's failure model is itself miscalibrated: its own
   predicted delivery probability is 98.2% against the 73.3% it achieves, and
   refining the grid to Δ = 10 s *lowers* the realized ratio (70.7%) while the
   model's value climbs to 100%.
3. **Tractability: on every scenario of the paper, exact BRUF/RUCoP is cheap.**
   States grow roughly linearly with the slot count; each extra copy
   multiplies them by 2 to 3.5 on UC1/UC3 and by 9 to 18 on a large multi-hop
   plan. RUCoP-3 at Δ = 1 s takes under 0.5 s on UC1 and UC3, and BRUF-1
   costs within a factor of about 5 of SCG's full route selection either way.
   The cost binds only through copies on long multi-hop plans: on a two-hour 48-satellite plan, RUCoP-3 needs 5.1 M
   states and 62 s at Δ = 10 s, and RUCoP-4 exceeds 6 M states from
   Δ = 60 s down. The paper's sentence that the corner's "own tractability
   forbids" fine grids is not supported as written and should be narrowed
   (§4(c)).

## 1. What was implemented

New module `scgraph/routing/rucop.py` (no shared file was modified):

| Piece | What it does | Maps to |
|---|---|---|
| `project(graph, ..., Δ)` → `SlotModel` | eq. (14) projection q_c[k] on the grid t_k = t0 + kΔ, k < ceil((D − t0)/Δ), via `bernoulli_projection` (latent state enters by its marginal) | the slotted uncertain contact plan BRUF/RUCoP consume |
| `solve(model, N)` | exact backward induction over (copy distribution, slot). State = sorted tuple of copy locations. Joint actions: every holder splits its copies among `stay` and its usable contacts. Contacts succeed independently with q_c[k]; copies sent over one contact move or stay together; a failure leaves them at the sender (acknowledged communication). Goal states (any copy at the destination) are absorbing. States pruned by forward reachability from the source state; contacts into nodes that can no longer reach the destination dropped (the rc_src / rc_dest pruning of the LSS/RUCoP tooling, value-preserving). Ties on probability broken by earliest expected delivery slot (exact lexicographic DP). Returns the table, the model's value, (state, slot) pairs, transitions, time; optional time/transition/state budgets | RUCoP [raverta2021rucop], global-information policy |
| `bruf(model)` | N = 1 with a complete (node, slot) table | BRUF [raverta2018bruf] |
| `rucop(model, N)` | N copies, pruned table | RUCoP-N |
| `solve(..., joint=False)` | one transmission per slot over the whole network | the action space of `uncertain-dtn/artifact/examples/toy_rucop.py`; kept only to reproduce the toy |
| `execute_policy(policy, realization)` | runs a policy on ONE realization of the full laws (copies jointly). A copy at node u at time a sits in slot floor((a − t0)/Δ) and takes the policy's contact if it exists and can carry the bundle from max(a, t_s) to t_e (the `_hop` rule). On failure the copy is back in play at the boundary after the attempt's slot and the policy is queried again | online forwarding |
| `pcgr_select` | Walter-style probabilistic CGR: rank candidates by the product of marginal contact existence probabilities, tie-break by nominal arrival | [walter2019probcgr], simplified |
| `realize_with_state`, `conditioned_graph` | a realization that also returns the latent-state index; the graph conditioned on one latent state | the latent-oracle reference in UC3 |

**Execution mode (a deliberate choice, reported both ways).** The default mode
`"asap"` queues a copy on the next contact its policy plans for the current
state (looking ahead over `wait` slots) and sends it as soon as that contact
can carry it. That is how a DTN node forwards along a chosen contact, and it is
exactly how the static routes in this study are executed, so static and
contingent policies differ only in contingency. The literal slot-locked mode
`"slot"` (a `wait` holds the copy until the next boundary) is kept as a
sensitivity row. It costs BRUF heavily at coarse grids on UC1: at Δ = 60 s the
violation rate goes from 0.08% to 17.6% at σ = 15 s, and from 9.2% to 21.7% at
σ = 60 s, because a copy that reaches the satellite mid-slot waits for the next
boundary and misses a deadline its own contacts would have met. On UC3 (loose
deadline) the two modes coincide.

**Two modelling facts of the BRUF/RUCoP line that matter here.**
(i) *Per-slot independence.* A contact spanning several slots enters as one
independent trial per slot, as in the published benchmarks (the ring-road
plans slice contacts into 60 s slots with independent failures). Under
persistent failures (a contact that does not exist stays absent for its whole
window) this makes retries look far better than they are, and the effect grows
as Δ shrinks. (ii) *Global information.* The executed RUCoP policy sees the
global copy distribution, which is RUCoP's optimal (global) policy; the
distributed L-RUCoP variant is restricted to local information, so its
optimum in the model is no higher. In UC3 the extra information
is mostly whether a relay copy already reached the destination, so we expect
the gap to be small, but it was not measured.

## 2. Validation

`tests/test_rucop.py` (23 tests, 1.3 s; full suite 121 passed):

- **Toy reproduced exactly.** With the toy's action space (`joint=False`),
  N = 1, 2, 3 all give 0.72 to 1e-12, and the test imports
  `toy_rucop.py` and checks agreement with its `compute_optimal_policy` for
  N = 1, 2. With RUCoP's joint actions, N = 2 gives
  1 − (1 − 0.8·0.9)(1 − 0.7·0.85) = 0.8866 exactly (two copies leave
  together over both relays), which the toy cannot express because it allows
  a single transmission per slot.
- BRUF-1 equals RUCoP-1; values are non-decreasing in N; the BRUF value is never
  below the static max-product path's model probability on the same q (UC1 and
  UC3 projections).
- Simulating a RUCoP-2 policy inside the slotted model reproduces its value
  within 4 standard errors.
- Execution against realizations of a stochastic graph built so that the
  slotted model is exact (each contact fills one slot) gives 0.72 (N = 1) and
  0.8866 (N = 2) within ±0.015 over 20 000 realizations.
- Execution semantics: failed attempts leave the copy and the policy acts again;
  copies on one contact move together; "asap" leaves when the planned contact
  opens and "slot" at the next boundary; `realize_with_state` reproduces
  `graph.realize` draw for draw.

## 3. How to run

```bash
.venv/bin/pytest -q tests/test_rucop.py
.venv/bin/python experiments/run_rucop.py --quick          # smoke test, ~1 min
.venv/bin/python experiments/run_rucop.py                  # full, see wall time
.venv/bin/python experiments/run_rucop.py --part uc1|uc3|tract   # one part
.venv/bin/python experiments/run_rucop.py --replot         # figures from CSVs
```

Parallelism: `experiments/_parallel.py` `pmap`, one seeded generator per
instance, 3 workers (SCGRAPH_WORKERS). The tractability part runs in a single
process so its timings are not contended by our own workers (other jobs were
sharing the machine, so timings are indicative; state and transition counts are
machine-independent).

Wall time of the full run (Apple M3 Pro, Python 3.14.4, shared machine):
UC1 126 s (120 instances), UC3 58 s (400 instances), tractability 909 s
(single process; the 4-copy and 3-copy multi-hop solves dominate).
Total 1093 s (18 min). Tests: 1.3 s.

Every policy is built on a planning stream and evaluated on a separate
held-out stream: per instance, 2000 realizations of the full laws (UC3:
including the latent state), all policies executed on the same realizations.
UC1: 40 instances per σ (80 000 runs per policy and σ). UC3: 40 instances per ρ.
Projections use m = 4000 draws per contact; SLA-aware scoring M = 2000 draws
per candidate; `plan_proxy` gets the same budget (m = 2000, 40 realizations,
depth 3, 4 buckets); UC3 static sets follow `run_uc3.py` (30 realizations,
greedy m = 2000). Planners see the graph restricted to [t0, D].

Outputs: `results/rucop_uc1.csv`, `results/rucop_uc3.csv`,
`results/rucop_tractability.csv`, `figures/rucop_uc1.{pdf,png}`,
`figures/rucop_uc3.{pdf,png}`, `figures/rucop_tractability.{pdf,png}`.
The UC1/UC3 CSVs also hold paired per-instance contrasts (`kind = contrast`,
mean difference ± 1.96 SE over instances).

## 4. Results

### (a) UC1, timing uncertainty (violation %, exact 95% CP; latency = mean delivery time of on-time bundles)

| policy | kind | σ = 15 s | σ = 30 s | σ = 60 s | latency σ = 15 / 30 / 60 [s] |
|---|---|---|---|---|---|
| deterministic CGR (mean plan) | static | 0.26 [0.22, 0.30] | 6.21 [6.05, 6.38] | 29.21 [28.90, 29.53] | 72 / 79 / 97 |
| probabilistic CGR (Walter-style) | static | identical to det. | identical | identical | 72 / 79 / 97 |
| Bernoulli path, Δ = 60 s | static | 0.02 | 2.49 [2.38, 2.60] | 9.56 [9.36, 9.77] | 408 / 319 / 451 |
| Bernoulli path, Δ = 30 s | static | 0.00 | 1.06 [0.99, 1.13] | 3.59 [3.46, 3.72] | 533 / 477 / 538 |
| Bernoulli path, Δ = 10 s | static | 0 (≤ 0.005) | 0.28 [0.25, 0.32] | 0.67 [0.61, 0.73] | 598 / 532 / 557 |
| SLA-aware route (SCG) | static | 0.03 [0.02, 0.04] | 0 (≤ 0.005) | 0 (≤ 0.005) | 220 / 567 / 570 |
| BRUF-1, Δ = 60 s | contingent | 0.08 [0.06, 0.11] | 2.45 [2.35, 2.56] | 9.17 [8.97, 9.37] | 400 / 315 / 449 |
| BRUF-1, Δ = 30 s | contingent | 0.00 | 0.89 [0.82, 0.95] | 3.31 [3.19, 3.44] | 532 / 474 / 537 |
| BRUF-1, Δ = 10 s | contingent | 0 (≤ 0.005) | 0.26 [0.23, 0.30] | 0.55 [0.50, 0.61] | 598 / 532 / 557 |
| SLA-aware ranked table (SCG, `plan_proxy`) | contingent | 0.02 [0.01, 0.04] | 0 (≤ 0.005) | 0 (≤ 0.005) | 197 / 567 / 570 |
| BRUF-1, Δ = 60 s, slot-locked | sensitivity | 17.63 | 9.95 | 21.67 | |

Paired contrasts (per-instance mean difference ± 1.96 SE, 40 instances):

- contingency at a fixed model, BRUF − Bernoulli path at the same Δ: between
  −0.39 and +0.07 points, every interval covering 0 (σ = 60 s, Δ = 30 s:
  −0.28 ± 0.29);
- contingency at the full laws, ranked table − SLA-aware route: 0.00 at
  σ = 30, 60 s; −0.01 ± 0.02 points at σ = 15 s;
- model at a fixed (contingent) structure, BRUF − ranked table:
  +0.55 (Δ = 10 s), +3.3 (30 s), +9.2 (60 s) points at σ = 60 s (the CSV
  stores BRUF − SLA-aware route; the ranked table and the route both record 0
  failures at σ = 60 s, so the two contrasts coincide).

BRUF's own predicted delivery probability (mean over instances) at σ = 60 s:
1.000 (Δ = 10 s, realized 0.9945), 0.952 (30 s, realized 0.967), 0.725 (60 s,
realized 0.908). In 21% of runs at Δ = 60 s BRUF recovered from a failed
attempt and still delivered; contingency fires, it just rarely changes the
outcome on UC1, because the relay's only onward contact is the one whose
jitter causes the failure.

### (b) UC3, existence + correlated collapse, N = 2 (delivery ratio %, 95% CP half-width ≤ 0.4 points)

| ρ | naive | PCGR | Bern.-select | SCG static | BRUF-1 | SCG table, 1 copy | RUCoP-2 Δ=30 | RUCoP-2 Δ=10 | RUCoP-2 + latent oracle | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 96.5 | 96.5 | 96.3 | 96.4 | 88.8 | 89.7 | 98.7 | 99.0 | 98.8 | 100.0 |
| 0.1 | 86.8 | 86.8 | 90.7 | 92.8 | 85.1 | 86.2 | 96.4 | 97.5 | 97.4 | 98.9 |
| 0.2 | 77.5 | 77.5 | 82.4 | 87.8 | 81.9 | 82.5 | 93.5 | 93.5 | 94.3 | 96.0 |
| 0.3 | 67.8 | 67.8 | 76.9 | 81.4 | 76.4 | 76.2 | 88.5 | 85.4 | 89.2 | 91.1 |
| 0.4 | 58.5 | 58.5 | 68.1 | 73.9 | 70.3 | 69.9 | 81.8 | 80.0 | 82.7 | 84.5 |
| 0.5 | 48.9 | 48.9 | 60.2 | 65.0 | 61.9 | 60.9 | 73.3 | 70.7 | 74.1 | 75.8 |
| 0.6 | 39.6 | 39.6 | 47.6 | 54.8 | 53.6 | 52.8 | 62.9 | 60.6 | 63.6 | 65.2 |
| 0.7 | 30.2 | 30.2 | 37.9 | 43.8 | 43.0 | 41.4 | 51.2 | 50.5 | 51.8 | 53.3 |
| 0.8 | 20.5 | 20.5 | 26.6 | 30.7 | 30.8 | 29.8 | 37.0 | 36.6 | 37.4 | 38.7 |
| 0.9 | 11.0 | 11.0 | 14.5 | 16.7 | 16.9 | 16.5 | 21.1 | 21.2 | 21.2 | 22.5 |

(BRUF-1 and the SCG table carry one copy; the other bounded policies two.
Slot-locked RUCoP-2 equals RUCoP-2 within 0.2 points everywhere.)

Paired contrasts at ρ = 0.5 (points, ± 1.96 SE over instances):

| contrast | what it isolates | Δ delivery |
|---|---|---|
| SCG static − Bernoulli-select | correlation awareness, both static | +4.8 ± 2.3 |
| RUCoP-2 − Bernoulli-select | contingency, both on the independent marginals | +13.1 ± 2.4 |
| RUCoP-2 − SCG static | contingent-independent vs static-correlated | +8.3 ± 0.4 |
| RUCoP-2 with latent oracle − RUCoP-2 | what knowing the latent state adds to a contingent policy (upper bound) | +0.8 ± 0.3 |
| RUCoP-2 Δ=10 − RUCoP-2 Δ=30 | finer grid under per-slot independence | −2.6 ± 1.4 |
| SCG table (1 copy) − BRUF-1 | full laws vs Bernoulli, both contingent single-copy | −1.0 ± 2.4 |
| RUCoP-2 − BRUF-1 | second copy | +11.5 ± 1.4 |

RUCoP-2 beats the SCG static set at every ρ (+2.3 ± 0.2 points at ρ = 0,
peaking at +8.3 at ρ = 0.5, +4.4 ± 0.2 at ρ = 0.9), and sits 1.3 to 2.7 points
below the per-realization oracle. The Δ = 10 s grid helps slightly at ρ ≤ 0.1
(+1.2 ± 0.6 at ρ = 0.1) and hurts for ρ in 0.3 to 0.6 (−1.8 to −3.1
points). The model's own value at Δ = 30 s is 98.2% at ρ = 0.5 (realized
73.3%); at Δ = 10 s it is 100.0% for every ρ ≤ 0.7.
PCGR coincides with naive on UC3 (both regions have the same marginal
existence, so the confidence product ties and nominal arrival decides) and
with deterministic CGR on UC1 (every contact exists).

### (c) Tractability

Exact backward induction, pure Python, medians over 3 instances per scenario.
"States" are forward-reachable (state, slot) pairs (BRUF's complete table is
in the CSV as `bruf1`; the RUCoP-1 count is the like-for-like N = 1 figure).
Budget per solve: 300 s or 6 million states in the forward pass.

| scenario | Δ [s] | slots | RUCoP-1 (= BRUF) | RUCoP-2 | RUCoP-3 | RUCoP-4 | projection (eq. 14) |
|---|---|---|---|---|---|---|---|
| UC1 (14 to 18 contacts, 10 to 14 nodes, H ≈ 600 s) | 60 | 10 | 45 st, < 1 ms | 99 st, 1 ms | 189 st, 4 ms | | < 5 ms |
| | 10 | 60 | 305 st, 2 ms | 699 st, 8 ms | 1 389 st, 27 ms | | 10 ms |
| | 1 | 596 | 3 081 st, 26 ms | 7 075 st, 94 ms | 14 085 st, 0.29 s | | 50 ms |
| UC3 (12 contacts, 8 nodes, H = 400 s) | 30 | 14 | 86 st, 1 ms | 286 st, 3 ms | 800 st, 11 ms | | < 5 ms |
| | 1 | 400 | 2 705 st, 21 ms | 9 215 st, 89 ms | 26 092 st, 0.40 s | | 30 ms |
| multi-hop: 48-satellite CPD plan, 2 h horizon, p_exist = 0.8 (4 530 to 4 918 contacts, 50 nodes) | 240 | 30 | 699 st, 15 ms | 10 890 st, 0.09 s | 126 523 st, 1.3 s | 1.16 M st, 18 s | 1.1 s |
| | 120 | 60 | 1 642 st, 31 ms | 27 956 st, 0.23 s | 344 571 st, 3.8 s | 3.36 M st, 62 s | 1.7 s |
| | 60 | 120 | 3 537 st, 68 ms | 62 340 st, 0.53 s | 817 892 st, 9.9 s | > 6 M st (censored) | 2.9 s |
| | 30 | 240 | 7 275 st, 0.14 s | 128 195 st, 1.1 s | 1.68 M st, 20 s | censored | 5.3 s |
| | 10 | 720 | 22 017 st, 0.44 s | 388 847 st, 3.3 s | 5.10 M st, 62 s | censored | 15.5 s |

SCG reference on the same instances (route_stats at M = 2000 per route;
full selection = candidate generation + scoring every candidate): UC1
0.84 ms per route, 28 ms in total (4 candidates); UC3 0.36 ms, 3.3 ms
(6 candidates); multi-hop 0.13 ms per route, 0.33 s in total (147 to 162
candidates). None of it depends on Δ.

Reading. For a fixed number of copies the reachable state space grows
roughly linearly with the slot count (×2 to ×2.7 per halving of Δ); each extra
copy multiplies it by 2 to 3.5 on UC1/UC3, where few nodes can reach the
destination, and by 9 to 18 on the multi-hop plan. On the paper's own scenarios
nothing is expensive: RUCoP-3 at Δ = 1 s takes 0.3 to 0.4 s, and BRUF-1
(excluding the projection) costs within a factor of about 5 of SCG's full
selection, cheaper at Δ ≥ 10 s on UC1/UC3 and at Δ ≥ 30 s on the multi-hop
plan. The cost binds only through copies on long, connected plans: on the two-hour
multi-hop plan RUCoP-3 needs 5.1 million states and a minute at Δ = 10 s, and
RUCoP-4 exceeds the 6-million-state budget from Δ = 60 s down. On that plan
the Monte-Carlo projection of eq. (14) itself (m = 2000 draws per contact)
costs 15.5 s at Δ = 10 s, more than BRUF and fifty times SCG's whole selection.
A compiled solver would move all curves down by a constant factor; the
slopes are properties of the model.

So the measurement supports a narrower claim than §VII-B makes: slotted MDPs
stay tractable at fine grids for one or two copies (and three on small
plans); what grows quickly is the copy dimension. The fine grid is also not
what the corner needs in UC3, where refining it makes the policy worse
(§4(b)).

## 5. Reading, honestly

Two effects are mixed in any "BRUF/RUCoP vs SCG" comparison: *static vs
contingent* (a committed route set against a policy that reacts to observed
failures), and *Bernoulli/independent vs full laws* (the model the decision is
computed on). The runs separate them.

**Where BRUF/RUCoP loses (UC1, timing).** With the decision structure held
fixed, the model is everything: contingent BRUF vs contingent SCG table, and
static Bernoulli path vs static SLA-aware route, show nearly the same gap
(+3.3 and +3.6 points at σ = 60 s and Δ = 30 s; +0.55 and +0.67 at
Δ = 10 s). Contingency, held at a fixed model, contributes nothing
measurable. The reason is structural: a timing
failure happens at a relay whose only onward contact is the one that slipped,
so there is nothing to fall back on once it is observed, and the right
decision is the one taken at the source with the full law in view. The
projection cannot see that a jittered contact which is rarely open at a slot
boundary is nonetheless almost surely usable within its window, and at
σ = 15 s it keeps BRUF (and the static Bernoulli path, which takes an air
route on 3% of instances at Δ = 10 s) on the backbone at 598 s mean delivery,
where the SLA-aware router takes the air relays on 78% of instances and
delivers in 220 s. The one cell where the corner does better is σ = 15 s, where BRUF/Bernoulli at fine grids record 0 violations against
0.03% for the SLA-aware route, at 2.7 times its latency.

**Where BRUF/RUCoP wins (UC3, existence).** Contingency is the dominant lever:
+13 points from contingency at a fixed (independent) model against +5 points
from correlation awareness at a fixed (static) structure. An acknowledged
failure at the source is observed and retried on a later contact, which no
static set can do. Telling RUCoP the latent state adds under 1 point, so on this topology the correlation
information that eq. (13) exploits is largely recovered by reacting to
failures: region A's contacts are tried first, and their failures hand the
copies to region B in time. RUCoP-2 also ends within 1.3 to 2.7 points of the
oracle at every ρ.

**Where the slotted model is wrong even when it wins.** RUCoP's predicted
delivery probability overstates the realized one by 25 points at ρ = 0.5 and by
up to 37 points (ρ = 0.7) at Δ = 30 s, and the error grows as the grid is
refined (up to 75 points at Δ = 10 s, ρ = 0.9), since per-slot independence
turns a contact that is absent for its whole window into many independent
retries. Refining the grid from 30 s to 10 s lowers realized delivery for
ρ between 0.3 and 0.6 (−2.6 points at ρ = 0.5, −3.1 at ρ = 0.3) while the
model's value goes to 1. For timing uncertainty a finer grid helps (UC1);
for persistent existence failures it hurts. In both cases the corner's own number is not a usable SLA estimate.

**Ties.** Single-copy contingent: BRUF-1 and the SCG ranked table are
indistinguishable on UC3 (−1.0 ± 2.4 points). Static: PCGR reduces to
deterministic CGR (UC1) and to naive (UC3) because its scalar confidence
carries no information in either scenario.

**What this means for the paper.**
- The comparison supports the paper's model claim (the projection loses
  timing information that decides the SLA in UC1, and its value is
  miscalibrated in both scenarios) and the correlation claim *among static
  sets* (H3 holds: +4.8 points).
- It does not support presenting the SCG static replica set as better than the
  uncertainty-aware line under existence failures: RUCoP-2 beats it by 2 to 8
  points. The honest framing is that contingency and the model are orthogonal.
  SCG supplies the law and the deadline objective; a contingent policy can be
  computed on top of it (the ranked table is the single-copy version and ties
  BRUF-1). A multi-copy contingent planner on the full laws does not exist in
  scgraph yet, and the UC3 numbers bound what it could add: at most the
  1.3 to 2.7 points between RUCoP-2 and the oracle. What the full laws would
  fix is the calibration: the policy's own estimate would become a usable SLA
  figure.
- The tractability sentence in §VII-B (main.tex, "The resolution the corner
  needs to close the gap is also the one its own tractability forbids. Slotted
  decision processes have state spaces that grow with the slot count...")
  needs rewriting: see §4(c).

## 6. Sentences the paper could use

> We reimplemented BRUF and RUCoP on the slotted projection of each scenario and
> executed every policy on the same held-out realizations of the full laws.
> Under timing uncertainty, BRUF's contingency adds nothing measurable over the
> static Bernoulli path at the same slot width, and both trail the SLA-aware
> router, which records no violation in 80 000 runs at σ = 60 s where BRUF
> violates 0.6% to 9.2% of deadlines for Δ = 10 to 60 s.
> Under correlated existence failures, RUCoP-2 delivers more than any static
> replica set, ours included (73% against 65% at ρ = 0.5), because acknowledged
> retries react to failures as they happen. Its own failure model overstates
> delivery by 25 points at ρ = 0.5 and degrades as the grid is refined, so the
> gain belongs to contingency, a decision structure that can equally be built on
> the stochastic contact graph (our single-copy ranked table ties BRUF).

> Solving these policies exactly is cheap at the scale of our scenarios: with
> forward-reachability pruning, BRUF and three-copy RUCoP take under half a
> second even at Δ = 1 s, because the reachable state space grows roughly
> linearly with the number of slots. The cost binds through the copy budget,
> about an order of magnitude per copy on large plans: on a two-hour plan over
> 48 satellites, RUCoP-3 needs 5.1 million states at Δ = 10 s and RUCoP-4
> exceeds six million already at Δ = 60 s, while the stochastic-contact-graph selection takes 0.33 s
> regardless of any grid.

## 7. Related-work comparison table (rows)

E = contact existence, T = timing (start/end), C = capacity/rate,
Corr = dependence across contacts. "†" marks cells taken from the paper's own
summary of a work whose PDF was not available locally; check before printing.

| Approach | E | T | C | Corr | Time model | Objective | Output | Scaling / cost |
|---|---|---|---|---|---|---|---|---|
| PRoPHET [lindgren2003prophet, rfc6693prophet] | learned from encounter history (no plan) | no | no | no | event-driven, no plan | heuristic delivery predictability | per-encounter forwarding rule | O(neighbours) state per node, fully online |
| Probabilistic CGR [walter2019probcgr] | scalar per-contact confidence | no | no | no (independent) | continuous (contact plan) | route delivery probability as a product of confidences† | route(s)† | CGR route search plus confidence bookkeeping |
| Satellite-UAV MDP routing, pcgUAV [pcgUAV] | contact probability from a semi-Markov mobility model | no | no | no† | discrete decision stages† | expected delivery (MDP)† | policy | MDP solved offline; grows with nodes × stages† |
| BRUF [raverta2018bruf] | yes, independent per (contact, slot) | no | no | no | slotted | max P[delivery by horizon] | single-copy policy | PRISM in the original; here ≤ 0.03 s at Δ = 1 s on UC1/UC3, 0.44 s at Δ = 10 s on the 2 h multi-hop plan |
| RUCoP [raverta2021rucop] | same | no | no | no | slotted | max P[≥ 1 of N copies delivered by horizon] | N-copy policy (global; L-RUCoP local) | worst case exponential in nodes, doubly exponential in copies; measured here (pruned): roughly linear in slots, 2 to 18× per extra copy; RUCoP-3 5.1 M states at Δ = 10 s on the 2 h plan |
| LSS [dargenio2020lss, budde2024modest] | same | no | no | no | slotted | P[delivery] by statistical model checking over sampled schedulers | distributed (local) scheduler | constant memory per sampled scheduler; scales past RUCoP's memory limit at a sampling-accuracy cost |
| Q-learning / GROGU [dargenio2025comparing, negrelli2025grogu] | same (probabilistic link availability, known contact bounds) | no | no | no | slotted | max P[delivery], learned | learned local policy (GROGU: one GNN policy shared by all nodes) | training cost (high for GROGU), cheap inference; GROGU adapts to outages without retraining |
| POMDP-DNF [stock2025dnf] | via node failures | no | no | yes: dependent node failures (CTMC node states) | discrete time | delivery ratio / delay, online POMCP planning | per-decision action from a belief | online Monte-Carlo tree search per decision |
| SOTA routing [fan2006sota, nie2009sota, samaranayake2012reliable] | no (fixed graph) | yes (random edge travel times) | n/a | independent edges | continuous (budget discretised for convolution) | max P[arrival by deadline] | adaptive policy or path | convolution-based, pseudo-polynomial in the time budget |
| This work (SCG) | yes | yes (start, end) | yes (rate, volume) | yes (latent state, common factors, within-contact copula) | continuous | max P[T ≤ D], or min cost s.t. P[T ≤ D] ≥ 1 − ε | route; ranked table (contingent placement); correlation-aware replica set | candidate generation (CGR cost) + per-route MC scoring O(M · hops), flat in plan size (0.13 to 0.9 ms per route at M = 2000); onboard: table lookup |

## 8. Caveats

- RUCoP is executed with global information (its optimal policy); a
  distributed L-RUCoP is restricted to local information and was not run.
- The "asap" execution mode is our reading of how a slotted policy is deployed;
  the literal slot-locked mode is reported and is much worse on UC1 at Δ = 60 s.
- The projection uses the paper's horizon convention (ceil((D − t0)/Δ) slots),
  so the last slot can admit a hop that completes after D; this is part of the
  corner's resolution loss and is mitigated by "asap" execution.
- The PCGR baseline is a simplification (product of marginal existence
  probabilities, tie-break by nominal arrival); it carries no information in
  UC1 (all contacts exist) or UC3 (symmetric regions).
- UC1 instances here use a new seed (20261013) and differ from `run_uc1.py`'s;
  numbers are comparable in kind, though individual cells differ (det at
  σ = 60 s: 29.2% here, 28.6% in the paper).
- Tractability timings come from a pure-Python solver on a shared machine;
  a C++/PRISM implementation would shift the curves down by a constant
  factor without changing their slopes. State and transition counts are exact.
- The SCG side of UC3 has no multi-copy contingent planner; the comparison
  therefore pits our static replica sets against RUCoP's contingent policy.
