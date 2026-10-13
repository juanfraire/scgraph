# Concurrent bundles, finite storage, residual volume (T8 / D3)

Answers R1.f ("Equation (3) assumes boundless storage and atomic bundle
forwarding ... intermediate buffer queues will experience severe congestion")
and R2.11 ("residual contact volume, fragmentation, and concurrent flows").

## What was implemented

| File | Content |
|---|---|
| `scgraph/flows.py` | `run_flows`: event-driven multi-bundle executor on one realization. `cgr_route_booked` / `book_routes`: CGR with SABR-style volume booking. `assign_contention_aware`: sequential SLA-aware assignment scored by Monte Carlo over the executor. |
| `tests/test_flows.py` | 14 tests: single bundle equals `arrival.delivery_time` (400 random chains and UC1 realizations), FIFO queueing, FIFO by arrival time, relay pipelining, buffer drop, slot reuse at equal times, unbounded source and destination, window cut-off, cut transfer holding the contact, missing contact, no route, route chaining, booked CGR, contention-aware split. |
| `experiments/run_flows.py` | The experiment (`--quick`, `--workers N`). |
| `results/flows_load.csv`, `results/flows_buffer.csv` | One row per (K, B, policy). |
| `figures/flows.{pdf,png}` | Left: on-time fraction vs K at B = inf. Right: on-time fraction vs B at K = 16. |

No shared file was modified. `.venv/bin/pytest -q`: 98 passed.

## Executor semantics (`run_flows`)

On one realized plan (a `Realization` from `graph.realize`, or a cid to
`DetContact` map):

- Each bundle has a source, destination, size, release time, deadline and a
  committed route (contact ids in hop order).
- A contact is a single server. It transmits one bundle at a time in FIFO
  order of arrival at its sender (ties by bundle index). Transmission starts at
  max(arrival at sender, contact start, contact busy-until): eq. (2) plus the
  backlog.
- Transfer is atomic at constant rate (eq. 3). A transfer that cannot end by
  the contact end is cut there: the contact stays busy until its end and the
  bundle is lost at that hop (cause `window`). A bundle whose turn comes after
  the window closed is lost the same way, immediately. The sender is assumed
  not to know that a jittered window will close early, so a doomed transfer
  wastes the tail of the window.
- On success the bundle reaches the next node one OWLT later (eq. 4).
- Storage: every node other than the bundles' sources and destinations holds
  at most B bundles. A bundle is resident from its arrival until its
  transmission out finishes, or until it is lost there. An arrival that finds B
  residents is dropped (cause `buffer`). Residency is tracked at B = inf too,
  which gives the peak storage a policy would need.
- A committed contact absent from the realization loses the bundle on arrival
  at its sender (cause `missing`); a bundle the policy could not route has
  cause `noroute`.
- Events run in global time order. At equal times departures go before
  arrivals and arrivals before window openings, so a slot freed at t serves a
  bundle arriving at t.
- Contacts are independent servers (a node may use two open contacts at once).
  There is no bundle lifetime and no re-routing: a late bundle is still
  delivered and counted late; a lost bundle stays lost.

With one bundle and unbounded storage this is `arrival.delivery_time`
exactly (tested).

## Policies

All four see only the selection stream. K bundles of size 100 are released
together at the instance's t0 from the event site, all with the instance's
deadline D (the favourable backbone pass's arrival + 40 s).

- **P1, CGR on the mean plan.** `cgr_route` once; all K bundles share the
  earliest-arrival route.
- **P2, CGR + volume booking (SABR-style).** Bundles routed in turn on the mean
  plan; each books its volume on the contacts it uses. A later bundle's
  earliest transmission opportunity on a contact is max(arrival, t_start +
  booked / rate), and the contact is usable only if the transfer still ends by
  t_end, which is the residual-volume test booked + size <= rate x duration.
  Deadline-blind, as CGR is.
- **P3, SLA-aware, contention-blind.** `pick_route` (argmax P[T <= D] over
  `candidate_routes`, M = 2000) for one bundle. The K bundles are identical,
  so routing each alone returns the same route; it is computed once.
- **P4, SLA-aware, contention-aware.** Bundles committed in index order.
  Bundle j takes the candidate (same candidate set as P3) that maximises its
  own on-time frequency over M_SEL = 500 Monte-Carlo plans, each replayed in
  the executor with bundles 1..j-1 on their committed routes and the storage B
  under test. Ties: larger expected number of on-time bundles in the set, then
  earlier mean delivery of bundle j. All candidates share the same plans
  (common random numbers). Exact pruning: other bundles can only delay or drop
  bundle j, so its frequency alone bounds it under contention, and candidates
  whose bound falls below the best found are skipped (about 4x fewer executor
  runs).
- Controls: **P4i** is the B = inf assignment of P4 played at finite B
  (contention-aware, storage-blind). **P2f / P4f** split each bundle into two
  fragments of 50 routed as independent bundles by P2 / P4; a bundle is on time
  iff both fragments are.

Sequential policies have the prefix property (the K-bundle assignment is the
first K of the 16-bundle one), so one assignment per B serves the load sweep.

## Protocol

- UC1 on the CPD backbone, `make_instance_cpd(rng, sigma=40)` (unmodified),
  restricted to contacts that end after t0.
- R = 30 instances; each has three independent streams spawned from its task
  seed: instance, selection, evaluation.
- Every policy is evaluated on the SAME N_EVAL = 1000 held-out realizations of
  its instance, drawn from the evaluation stream.
- Load sweep: K in {1, 2, 4, 8, 16} at B = inf. Buffer sweep: B in
  {inf, 8, 4, 2} at K = 16 (plotted) and K = 8.
- Metrics: fraction of bundles on time, with the exact two-sided 95%
  Clopper-Pearson interval on the pooled count (`scgraph/stats.py`); the mean
  and 95% t-interval of the per-instance on-time fraction (the figure's error
  bars); probability that the whole burst is on time; mean delivery time of
  delivered bundles (from t0); loss breakdown; mean peak storage at the busiest
  relay node; share of bundles routed through an air relay.

The pooled Clopper-Pearson interval treats the 30 x 1000 x K bundle outcomes
as independent. Bundles of one realization are correlated, so it is too
narrow; the per-instance t-interval is the honest measure of spread and is
what the figure shows. Both are in the CSVs.

## How to run

```bash
.venv/bin/python experiments/run_flows.py --quick    # 6 instances, ~7 s
.venv/bin/python experiments/run_flows.py            # 30 instances, ~2 min
```

Wall time of the full run: 121 s on 3 worker processes (351 s CPU), Apple
M3 Pro shared with other jobs (load average about 7). Per instance: 8.2 s
route selection (P3 plus five P4 assignments: four B values and P4f), 3.6 s
evaluation. One P4 assignment of 16 bundles at M = 500 over 17 candidates
takes 1.0 s single-process (33k executor runs after pruning, 145k without).

## Results

### Load sweep, B = inf: fraction of bundles on time

Pooled estimate, [Clopper-Pearson 95%], (+- per-instance 95% t half-width).

| K | P1 CGR | P2 CGR + booking | P3 SLA-aware, blind | P4 SLA-aware, contention-aware |
|---|---|---|---|---|
| 1 | 0.847 [0.843, 0.851] (+-0.020) | 0.847 [0.843, 0.851] (+-0.020) | 1.000 [1.000, 1.000] (+-0.000) | 1.000 [1.000, 1.000] (+-0.000) |
| 2 | 0.819 [0.816, 0.822] (+-0.022) | 0.836 [0.833, 0.838] (+-0.023) | 0.983 [0.982, 0.984] (+-0.034) | 0.998 [0.998, 0.998] (+-0.004) |
| 4 | 0.751 [0.748, 0.753] (+-0.026) | 0.817 [0.815, 0.819] (+-0.027) | 0.892 [0.890, 0.893] (+-0.076) | 0.989 [0.988, 0.989] (+-0.009) |
| 8 | 0.476 [0.474, 0.478] (+-0.031) | 0.790 [0.788, 0.792] (+-0.022) | 0.525 [0.523, 0.527] (+-0.057) | 0.938 [0.937, 0.939] (+-0.010) |
| 16 | 0.238 [0.237, 0.239] (+-0.015) | 0.744 [0.743, 0.745] (+-0.014) | 0.263 [0.261, 0.264] (+-0.028) | 0.848 [0.847, 0.849] (+-0.013) |

P4 at K = 1: 30 000 bundles, all on time; the Clopper-Pearson lower bound is
0.99988. P1 at K = 1 misses the deadline for 0.153 of bundles, close to the
paper's held-out deterministic violation at sigma = 40 (0.134 in
`results/uc1.csv`, over a different instance sample).

### Load sweep, where the bundles go and how they are lost

| K | policy | on time | late | window | buffer | whole burst on time | mean delivery [s] | peak storage | air share |
|---|---|---|---|---|---|---|---|---|---|
| 8 | P1 | 0.476 | 0.000 | 0.524 | 0.000 | 0.000 | 106 | 2.8 | 1.00 |
| 8 | P2 | 0.790 | 0.000 | 0.210 | 0.000 | 0.437 | 102 | 2.8 | 1.00 |
| 8 | P3 | 0.525 | 0.142 | 0.333 | 0.000 | 0.000 | 674 | 6.2 | 0.00 |
| 8 | P4 | 0.938 | 0.000 | 0.062 | 0.000 | 0.685 | 418 | 4.3 | 0.47 |
| 16 | P1 | 0.238 | 0.000 | 0.762 | 0.000 | 0.000 | 106 | 2.8 | 1.00 |
| 16 | P2 | 0.744 | 0.000 | 0.256 | 0.000 | 0.194 | 128 | 4.5 | 0.99 |
| 16 | P3 | 0.263 | 0.094 | 0.644 | 0.000 | 0.000 | 675 | 9.8 | 0.00 |
| 16 | P4 | 0.848 | 0.000 | 0.152 | 0.000 | 0.347 | 284 | 4.7 | 0.74 |

`missing` and `noroute` never occur (UC1 contacts have p_exist = 1, and the
mean plan always has a later pass for P2).

### Buffer sweep: fraction on time (buffer drops, window losses)

K = 16:

| B | P1 | P2 | P3 | P4 | P4i (storage-blind) |
|---|---|---|---|---|---|
| inf | 0.238 (0.00, 0.76) | 0.744 (0.00, 0.26) | 0.263 (0.00, 0.64) | 0.848 (0.00, 0.15) | 0.848 (0.00, 0.15) |
| 8 | 0.238 (0.00, 0.76) | 0.744 (0.00, 0.26) | 0.263 (0.23, 0.44) | 0.848 (0.00, 0.15) | 0.848 (0.00, 0.15) |
| 4 | 0.217 (0.03, 0.75) | 0.674 (0.07, 0.26) | 0.223 (0.39, 0.39) | 0.822 (0.01, 0.17) | 0.789 (0.06, 0.15) |
| 2 | 0.170 (0.08, 0.75) | 0.506 (0.24, 0.26) | 0.123 (0.49, 0.39) | 0.590 (0.19, 0.22) | 0.545 (0.30, 0.15) |

K = 8:

| B | P1 | P2 | P3 | P4 | P4i (storage-blind) |
|---|---|---|---|---|---|
| inf | 0.476 (0.00, 0.52) | 0.790 (0.00, 0.21) | 0.525 (0.00, 0.33) | 0.938 (0.00, 0.06) | 0.938 (0.00, 0.06) |
| 8 | 0.476 (0.00, 0.52) | 0.790 (0.00, 0.21) | 0.525 (0.00, 0.33) | 0.938 (0.00, 0.06) | 0.938 (0.00, 0.06) |
| 4 | 0.434 (0.06, 0.51) | 0.780 (0.01, 0.21) | 0.446 (0.32, 0.23) | 0.925 (0.00, 0.08) | 0.859 (0.08, 0.06) |
| 2 | 0.341 (0.16, 0.50) | 0.653 (0.14, 0.21) | 0.246 (0.53, 0.23) | 0.881 (0.00, 0.12) | 0.642 (0.30, 0.06) |

At K = 16, B = 2 the per-instance t half-widths are 0.004 to 0.012; P4
moves its air share from 0.74 to 0.88 to stay out of the satellite's storage.

### Proactive fragmentation (two fragments per bundle, B = inf)

On-time fraction (probability that the whole burst is on time):

| K | P2 | P2f | P4 | P4f |
|---|---|---|---|---|
| 1 | 0.847 (0.847) | 0.856 (0.856) | 1.000 (1.000) | 1.000 (1.000) |
| 2 | 0.836 (0.771) | 0.822 (0.782) | 0.998 (0.996) | 0.999 (0.997) |
| 4 | 0.817 (0.632) | 0.772 (0.663) | 0.989 (0.962) | 0.987 (0.963) |
| 8 | 0.790 (0.437) | 0.708 (0.476) | 0.938 (0.685) | 0.916 (0.708) |
| 16 | 0.744 (0.194) | 0.626 (0.219) | 0.848 (0.347) | 0.780 (0.393) |

## Reading

**Does the single-bundle SLA-aware router congest the backbone under load?
Yes.** It routes every bundle of the burst onto the one favourable pass. That
pass can deliver about four bundles between its arrival and the deadline (the
40 s margin is four transmission times), so P3 holds up to K = 2, slips at
K = 4 (0.892, with wide instance spread, +-0.076), and at K = 8 and 16 it falls
to 0.525 and 0.263: the level of plain CGR (0.476, 0.238) and well below
CGR with volume booking (0.790, 0.744). The excess bundles are cut at the
uplink or downlink window, or delivered late (0.142 at K = 8). The satellite
needs 9.8 bundles of storage on average at K = 16. This is the congestion R1
anticipated. One correction to R1's framing: in the paper's model the
backbone is chosen up front by the planner (the 69 s to 549 s latency is a
planned choice); there is no reactive fallback after a missed air window.
The congestion appears because every bundle is planned onto the same pass.

**Does contention-aware scoring fix it? Largely, at a latency cost.** Running
the executor inside the same Monte-Carlo scoring loop (P4) keeps the
on-time fraction at 0.989 for K = 4, 0.938 for K = 8 and 0.848 for K = 16, the
best of the four policies at every K, 0.10 to 0.15 above volume booking for
K >= 8. P4 fills the backbone pass up to its deadline capacity and sends the
rest to the air relays (air share 0.47 at K = 8, 0.74 at K = 16). It does not
reach 1: what remains at K = 16 are window losses on jittered air relays
(0.152), which P4 accepts once the backbone is full, and the whole burst is on
time in only 0.347 of realizations. Mean delivery is 284 s against 128 s for
booking, since a quarter of the bundles take the 600 s pass. We have no
optimal multi-bundle reference, so the gap between P4 and the best achievable
on-time fraction is unknown.

**What does a finite buffer do?** B = 8 changes no on-time fraction for any
policy. For P3 it relabels losses: 0.225 of bundles are now dropped at the
satellite, bundles that the downlink window would have cut anyway. At B = 4 and
B = 2 every policy loses on-time deliveries. P3 suffers most and ends lowest
at K = 16, B = 2 (0.123, two bundles of 16: the satellite stores two while it
waits for the downlink). Volume booking also degrades sharply (0.744 to 0.506),
since CGR does not see storage. P4 stays best (0.590), and scoring under the
true B matters: the storage-blind P4i gets 0.545 at K = 16 and 0.642 against
0.881 at K = 8, where P4 moves bundles from the satellite to the air relays and
keeps buffer drops at 0.002.

**Fragmentation.** A two-way proactive split routed fragment by fragment lowers
the on-time fraction from K = 4 on (K = 16: P2 0.744 to 0.626, P4 0.848 to
0.780), with a small gain at K = 1 for booking (0.847 to 0.856, from packing the
window tails). The probability that the whole burst is on time moves the other
way (P4: 0.347 to 0.393). We have not analysed that inversion; a plausible
reading is that spreading each bundle's fragments over several relays ties
bundle outcomes together, so losses concentrate in fewer realizations.

## Fragmentation and eq. (3)

- **Reactive fragmentation** keeps what crossed before the window closed and
  carries the residual forward. Eq. (3) then runs over the contacts that serve
  hop i in turn: E_i = inf{t : sum over c in C_i of the integral of R^c from
  tau_c to t >= omega_m}, with C_i the ordered contacts from the hop's sender to
  its receiver, and E_i is infinite only when the whole set cannot carry
  omega_m. Downstream, the fragments travel as separate bundles and the
  delivery time is the reassembly time, the maximum over fragments. In the
  executor this turns a `window` loss into a partial delivery plus a residual
  that needs a later contact on the same hop. In UC1 each air hop has a single
  window and the next backbone pass arrives after the deadline, so reactive
  fragmentation would mostly turn window losses into late deliveries. We did not
  implement it.
- **Proactive fragmentation** splits omega_m = sum_k omega_k at the source,
  sized to residual volumes, and sends piece k on its own route pi_k; each piece
  follows eqs. (2) to (4) with omega_k in place of omega_m, and T = max_k T_k.
- **The SLA objective keeps its form**, P[T <= D] with T the reassembly time.
  What changes is the event: success becomes the intersection over fragments,
  P[T_k <= D for every k], the AND counterpart of replication's eq. (13), which
  is an OR over copies. Under independence it is a product, so splitting across
  jittered routes multiplies their risks; under positive dependence the product
  is pessimistic. Joint Monte-Carlo sampling evaluates it directly
  (`joint_route_set_deliveries` with `all` over routes in place of `any`). The
  decision variable changes from a route to a split {(pi_k, omega_k)}, so
  candidate generation has to propose splits. The paper's sentence at
  main.tex:627 ("leaves the objective ... untouched") is right about the form
  and should add that the event becomes an intersection.
- The measured P2f / P4f rows above route fragments without a bundle-level
  objective, which is the cheap variant; a split-aware router would score
  P[all fragments on time] jointly.

## Limits

- One scenario (UC1 on the CPD backbone, sigma = 40 s), one source and
  destination, a burst release, equal sizes, a shared deadline, no background
  traffic, 30 instances.
- Committed routes: no re-forwarding at a node after a miss, no custody or
  retransmission, no lifetime expiry (late bundles keep occupying storage).
  A deployed node would hold a cut bundle and send it on a later pass; we did
  not model that, so `window` losses would partly become late deliveries.
- Contacts are independent FIFO servers with no per-node transmitter limit and
  no priorities; storage is counted in bundles (P2f / P4f count fragments,
  which is why their peak storage doubles).
- P4 is greedy in release order and myopic: it does not reserve capacity for
  later bundles, and its objective is bundle j's own probability. It scores
  with the true laws and the true B (no misspecification), on 500 plans.
  Its cost grows roughly with K^2 x candidates x M, since step j replays j
  bundles: about 1 s for 16 bundles here.
- The pooled Clopper-Pearson intervals ignore within-realization correlation
  (see Protocol); use the per-instance intervals for comparisons.
- `missing` and `noroute` are exercised by the tests only.

## Sentences the paper could use

1. To test the single-bundle assumption, we released K bundles at once from the
   UC1 event site (sigma = 40 s) and replayed each policy's routes on held-out
   realizations in an event-driven executor with FIFO contacts, atomic
   transfers cut at the window end, and storage for B bundles at every relay.
2. Routed one bundle at a time, the SLA-aware router sends the whole burst to
   the backbone pass, which can deliver about four bundles before the deadline:
   its on-time fraction falls from 1.00 at K = 1 to 0.53 at K = 8 and 0.26 at
   K = 16, below CGR with SABR-style volume booking (0.79 and 0.74).
3. Scoring candidates in the same Monte-Carlo loop with the already committed
   bundles replayed in the executor restores 0.94 and 0.85, the best of the
   four policies at every load, by filling the pass up to its deadline capacity
   and sending the rest over the air relays; selecting routes for 16 bundles
   takes about one second.
4. Finite storage tightens every policy (at K = 16 and two bundles per relay:
   0.59 for the contention- and storage-aware router, 0.51 for volume booking,
   0.12 for the contention-blind SLA-aware router), and scoring under the true
   storage is worth 0.24 at K = 8 (0.88 against 0.64).
