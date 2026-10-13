# T8: compute, forwarding and control-plane overhead (R1.d, R1.e)

Revision item T8 of `REVISION-PLAN.md`, answering Reviewer 1 on two points:
the "1000x slower than CGR" reading of the runtime figure (R1.d), and the
unquantified telemetry, uplink and staleness cost of a ground-central planner
(R1.e). Written 2026-10-13.

Status: the sidecar and staleness numbers are final (they are counts and
seeded Monte Carlo, and reproduce exactly). The timing numbers are
**provisional**: they were measured while other experiment tracks were running
on the same machine. See "Re-running the timing" below.

## Files and how to run

Three new runners. No shared file was modified.

| Runner | Measures | Outputs | Wall time (this run) |
|---|---|---|---|
| `experiments/run_overhead.py` | planning- and forwarding-plane wall time per routing decision, forwarding-table size | `results/overhead_compute.csv`, `figures/overhead_compute.{pdf,png}` | 326 s (`--quick`: 49 s) |
| `experiments/run_sidecar.py` | control-plane bytes per contact, per plan, per refresh; uplink time | `results/overhead_sidecar.csv` | 0.2 s |
| `experiments/run_staleness.py` | SLA violation and delivery time when the planner's information is Δa old | `results/staleness.csv`, `figures/staleness.{pdf,png}` | 52 s on 3 workers, 150 s CPU (`--quick`: 6 s) |

```bash
.venv/bin/python experiments/run_overhead.py            # timing; run on an idle machine
.venv/bin/python experiments/run_overhead.py --replot   # redraw the figure from the CSV
.venv/bin/python experiments/run_sidecar.py
.venv/bin/python experiments/run_staleness.py           # also --quick, --replot
```

`run_overhead.py` is single-process by construction (it does not use `pmap`).
`run_staleness.py` uses `_parallel.pmap` with at most 3 workers; its results
do not depend on the worker count.

### Re-running the timing

This run: Apple M3 Pro (12 cores), macOS 26.6.2, Python 3.14.4, numpy 2.5.1,
scipy 1.18.0, `SCGRAPH_WORKERS=1`, single process, with a load average of 7 to
8 from other jobs. Before quoting milliseconds in the paper, repeat it on an
idle machine:

```bash
.venv/bin/python experiments/run_overhead.py    # about 5.5 minutes
```

The ratios between stages are what the response relies on, and they should be
more stable than the absolute values, but they are still worth confirming on
the clean run. Per-route scoring came out at 0.79 ms here against the 0.75 ms
of the committed `runtime.csv`, which gives a rough idea of the contention.

## 1. Compute per routing decision (R1.d)

**Setup.** The committed CPD skeletons `scenarios/scale_{6,12,24,48}` (98 to
14,640 contacts) with UC1-style air injection (`make_instance_cpd`, σ = 40 s),
with the same seeds and the same 8 release times per plan as `run_runtime.py`.
Each quantity is the median of 7 repeats, then the median over release times.

- (i) one deterministic `cgr_route` search on the mean plan;
- (ii) SABR-style route-table construction: the deterministic part of
  `candidate_routes` (mean-plan route, one single-contact exclusion per hop,
  disjoint suppression sweep) with no realisations. The replica in
  `run_overhead.py` is asserted, on every instance, to return exactly the route
  set of `candidate_routes(..., n_realizations=0)`;
- (iii) full candidate generation, with the 40 realisations;
- (iv) per-route scoring at M = 2000 draws;
- (v) the full SLA-aware selection, `pick_route` timed end to end.

"Lean" repeats (iii) and (v) with realisations that redraw only the contacts
carrying a law and copy the point-mass backbone. The shipped `graph.realize`
re-samples every one of the 14,640 Dirac contacts per realisation, which is a
Python implementation choice; the lean variant is exact for UC1 (no latent
state, no common factor) and shows what the algorithm itself costs.

| Contacts | (i) one CGR | (ii) SABR table (routes) | (iii) candidates | (iv) score/route | (v) selection | (v)/(ii) | (v)/(i) | (v) lean | lean/(ii) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 98 | 0.012 ms | 0.11 ms (5) | 30.5 ms | 0.79 ms | 33.5 ms | 293 | 2,898 | 27.9 ms | 244 |
| 298 | 0.023 ms | 0.42 ms (12) | 42.3 ms | 0.79 ms | 46.1 ms | 109 | 2,031 | 29.9 ms | 71 |
| 1,110 | 0.074 ms | 9.2 ms (58.5) | 103.7 ms | 0.80 ms | 113.6 ms | 12.4 | 1,539 | 50.5 ms | 5.5 |
| 14,640 | 0.82 ms | 411 ms (228) | 1,348 ms | 0.79 ms | 1,374 ms | 3.3 | 1,672 | 522 ms | 1.27 |

**Reading.**

- The factor of about 1000 that Reviewer 1 quotes reproduces: (v)/(i) is
  1,539 to 2,898. It compares a complete offline selection with one route
  search.
- Measured against the alternative-route table a SABR router builds anyway,
  the full selection costs 293 times more on the 98-contact plan, 12 times at
  1,110 contacts and 3.3 times at 14,640. The ratio shrinks with the plan
  because route enumeration grows with the plan while the distributional
  increment grows more slowly. On the small plans the ratio is large and the
  cost small: (v) is 34 to 46 ms, a fixed overhead set mostly by drawing 40
  realisations through scipy (about 0.6 ms each, even on a 98-contact plan).
- The distributional increment, (v) minus (ii), is 33 ms on the smallest plan
  and 963 ms on the largest; with lean realisations it is 28 ms and 111 ms.
  On the largest plan the lean selection costs 1.27 times the SABR table.
- Per-route scoring is flat at 0.79 ms across a 150-fold range of plan size,
  as in the submitted figure.
- On the largest plan the median candidate set (228 routes) and the median
  SABR table (228 routes) coincide: the candidate-set growth the paper reports
  is route enumeration, the part a deterministic router already pays.

## 2. Forwarding plane (R1.d)

**Setup.** On each instance a `ProxyPlan` (Sec. VI: ranked table per node and
arrival bucket, depth 3, 4 buckets, `plan_proxy` defaults) is built offline on
the plan restricted to the bundle's lifetime, `graph.window(t0, D)`; without
the window the build on the largest plan would run every candidate search over
all 14,640 contacts for every node and bucket. The table is then executed
against 200 realisations; the 400 online decisions this produces,
`chooser()(node, a, avail, used)`, are replayed 20 times and timed. Table size
is the packed length of two binary encodings: "first hop" stores, per (node,
bucket) key, the first contact id of each ranked route, which is all the
chooser reads; "full" stores whole routes (u16 node, u8 bucket, u8 count per
key; u8 length plus u32 ids per route; 16-byte header).

| Contacts | Window contacts | Build (offline) | One resolution (onboard) | Keys | Ranked routes | Table bytes, first hop / full | Uplink at 9.6 kbit/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 98 | 14 | 173 ms | 0.53 µs | 14.5 | 19.5 | 154 / 212 B | 0.13 / 0.18 s |
| 298 | 16 | 254 ms | 0.55 µs | 14 | 16.5 | 138 / 181 B | 0.12 / 0.15 s |
| 1,110 | 40 | 748 ms | 0.59 µs | 19 | 29 | 212 / 297 B | 0.18 / 0.25 s |
| 14,640 | 253 | 1,923 ms | 0.61 µs | 23 | 47.5 | 292 / 486 B | 0.24 / 0.40 s |

**Reading.** An onboard decision is a dictionary lookup and a scan of at most
three ranked entries: 0.53 to 0.61 µs per decision in CPython, flat in plan
size. On the largest plan that is about 1,300 times cheaper than one onboard
CGR search over the same plan (0.82 ms). The absolute figures are for a
desktop core running an interpreter; the comparison that carries over to a
flight processor is the ratio, and even a processor 1,000 times slower would
resolve a decision in under a millisecond. The table is per flow (one source,
destination and deadline); the bytes scale with the number of flows, nodes and
buckets. Every table delivered all 200 executions on time (`proxy_pdr` = 1.0),
as expected at σ = 40 s with a 40 s backbone margin.

## 3. Control-plane size (R1.e)

**Records** (little-endian, packed; sizes from `struct.calcsize`):

| Record | Fields | Bytes per contact |
|---|---|---:|
| Classical, ION text | `a contact +start +end from to rate`, rendered from the plan | 33.8 to 34.1 (measured) |
| Classical, binary | u32 start, u32 end, u16 from, u16 to, u32 rate | 16 |
| Sidecar, timing | header (u32 contact index, u8 bitmap) + TruncNormal (μ, σ, lo, hi as f32) | 21 |
| Sidecar, volume | header + LogNormal (μ, σ) | 13 |
| Sidecar, existence | header + p | 9 |
| Sidecar, timing + volume + existence | | 33 |
| Sidecar, joint | the above + group/factor (u16 id, f32 loading) + copula (3 f32) | 51 |
| Telemetry (downlink) | u32 index + observed start, end, delivered volume (f32) | 16 |

A point-mass contact has no sidecar record at all, so the classical plan is
the sidecar-free special case on the wire as well. ION also needs `a range`
lines, one per node pair (252 lines for the 14,640-contact plan, under 2% of
its file). Times in f32 are offsets from the plan epoch (millisecond-scale
resolution over a 6 h plan).

**Per plan.** Uplink times are payload only, at 9.6 kbit/s / 100 kbit/s /
1 Mbit/s; bundle and frame headers add a few tens of bytes per refresh.
"Air/ground" counts every contact touching an air relay or a ground node (UC1
plans: 6 air contacts plus the satellite-ground passes; UC4: 774 HAPS contacts
plus 1,242 ground contacts).

| Plan | Contacts | ION text plan | UC1 sidecar as modelled | Sidecar, every contact joint (upper bound) | Refresh of air/ground contacts, joint |
|---|---:|---:|---:|---:|---:|
| scale_6 + UC1 air | 104 | 3.5 KiB | 126 B (0.10 s / 0.01 s / 1 ms) | 5.2 KiB (4.4 / 0.42 / 0.04 s) | 58 contacts, 2.9 KiB (2.5 / 0.24 / 0.02 s) |
| scale_12 + UC1 air | 304 | 10.1 KiB | 126 B | 15.1 KiB (12.9 / 1.24 / 0.12 s) | 138, 6.9 KiB (5.9 / 0.56 / 0.06 s) |
| scale_24 + UC1 air | 1,116 | 37.0 KiB | 126 B | 55.6 KiB (47 / 4.6 / 0.46 s) | 388, 19.3 KiB (16.5 / 1.6 / 0.16 s) |
| scale_48 + UC1 air | 14,646 | 484 KiB | 126 B | 729 KiB (622 / 60 / 6.0 s) | 2,386, 119 KiB (101 / 9.7 / 0.97 s) |
| uc4_cpd (ISL) | 3,606 | 119 KiB | n/a | 180 KiB (153 / 14.7 / 1.5 s) | 2,016, 100 KiB (86 / 8.2 / 0.82 s) |

Partial refreshes, joint records (timing-only records cost 21/51 = 0.41 of
these):

| Plan | f = 1% | f = 10% | f = 100% |
|---|---:|---:|---:|
| scale_48 + UC1 air | 7.3 KiB (6.3 / 0.60 / 0.06 s) | 73 KiB (62 / 6.0 / 0.60 s) | 729 KiB (622 / 60 / 6.0 s) |
| uc4_cpd | 1.8 KiB (1.6 / 0.15 / 0.02 s) | 18 KiB (15.3 / 1.5 / 0.15 s) | 180 KiB (153 / 14.7 / 1.5 s) |

UC4's air contacts alone (774) need 38.5 KiB with joint records (33 / 3.2 /
0.32 s) and 15.9 KiB with timing records (13.5 / 1.3 / 0.13 s). Telemetry
that observes every contact of the largest plan once is 229 KiB, the size of
its binary plan; observing 10% of it is 23 KiB (19.5 / 1.9 / 0.19 s).

**Reading.** What UC1 actually puts on the wire is 126 bytes, whatever the
size of the orbital plan, because the space layer is timing-certain (A1) and
point masses carry no record. The upper bound, every contact with a full
joint law, is 51 bytes per contact: 1.5 times the ION text line it
accompanies, 3.2 times a 16-byte binary contact record. A refresh of all the
air and ground contacts of the 3,606-contact multi-hop plan takes 8.2 s at
100 kbit/s with full joint records and 3.4 s with timing records; compared
with re-sending the classical records of the same contacts (31.5 KiB) that is
3.2 and 1.3 times as many bytes.

## 4. Staleness (R1.e)

**Setup.** UC1 (`make_instance_cpd`, CPD backbone, margin 40 s). Each air
contact's true start is nominal + jitter + drift: the jitter is the
instance's TruncNormal(0, σ); the drift is drawn once per air contact and
instance as d = κ√Δa · z, z ~ N(0, 1), a random walk with **κ = 1 s/√s**
(drift s.d. 7.7 s after 1 min, 17 s after 5 min, 30 s after 15 min, 42 s
after 30 min, 60 s after 1 h). The truth graph replaces each air start law by
TruncNormal(nominal + d, σ, ±3.5σ) through `StochasticContact.with_laws`; the
orbital layer does not drift. The same z is reused at every Δa, so each
instance moves along its own drift direction as its information ages (paired
design). Planners: deterministic CGR on the nominal plan; `pick_route`
(max-P) and `pick_route_chance` (ε = 0.05, `admission="lcb"`, δ = 0.05) on
the stale law σ ("unaware"), on σ_eff = √(σ² + κ²Δa) ("aware"), and on the
truth itself ("fresh", the drift known). 100 instances per σ, M = 2000
selection draws, 40 realisations; every committed route is scored on 10,000
held-out draws of the truth from a stream no planner touched (10⁶ draws
pooled per cell). Δa ∈ {0, 1, 5, 15, 30, 60} min.

σ = 40 s is the paper's UC1 jitter. At that level the SLA-aware routers
already avoid the air relays at ε = 0.05 (checked before the run: 0 of 20
instances used them), so a second level, σ = 10 s, covers the case where the
air segment is well characterised and the routers do commit to it.

Held-out violation (mean delivery time in s), mean over 100 instances:

**σ = 40 s**

| Δa | 0 | 1 min | 5 min | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|---:|---:|
| deterministic CGR | 0.141 (84) | 0.143 (84) | 0.160 (85) | 0.207 (87) | 0.270 (90) | 0.364 (95) |
| max-P and chance, unaware and aware (all four) | 0 (592) | 0 (592) | 0 (592) | 0 (592) | 0 (592) | 0 (592) |
| chance, fresh | 0 (592) | 0.001 (577) | 0.005 (500) | 0.005 (494) | 0.005 (491) | 0.004 (495) |

**σ = 10 s**

| Δa | 0 | 1 min | 5 min | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|---:|---:|
| deterministic CGR | 0.000 (70) | 0.000 (73) | 0.015 (77) | 0.062 (83) | 0.127 (89) | 0.252 (98) |
| max-P, unaware | 0.000 (71) | 0.000 (73) | 0.012 (78) | 0.056 (84) | 0.128 (90) | 0.259 (100) |
| max-P, aware | 0.000 (71) | 0.000 (85) | 0 (525) | 0 (554) | 0 (554) | 0 (554) |
| chance ε = 0.05, unaware | 0.000 (70) | 0.000 (72) | 0.015 (77) | 0.066 (83) | 0.137 (89) | 0.262 (99) |
| chance ε = 0.05, aware | 0.000 (70) | 0.000 (72) | 0.015 (77) | 0.010 (390) | 0 (554) | 0 (554) |
| chance ε = 0.05, fresh | 0.000 (70) | 0.000 (71) | 0.001 (72) | 0.002 (75) | 0.002 (83) | 0.002 (107) |

Instances whose admitted route is a significant breach (exact 95% lower bound
on its violation above ε), chance-constrained router at σ = 10 s:

| Δa | 0 | 1 min | 5 min | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|---:|---:|
| unaware | 0% | 0% | 5% | 13% | 27% | 32% |
| aware | 0% | 0% | 5% | 1% | 0% | 0% |

A "0" cell had no failure in 10⁶ held-out draws (exact one-sided 95% upper
bound 3.0 × 10⁻⁶); "0.000" is non-zero below 5 × 10⁻⁴. The across-instance
95% half-width on the means reaches 0.08 at σ = 10 s and 60 min (a few
instances fail with certainty once their hand-off is broken), so the
staleness curves are clear while the third digit is not. The chance
router's aggregate promise (its in-sample violation) was 0.0099 at 5 min and
0.0115 at 15 min for the aware planner, against 0.0150 and 0.0104 delivered;
the unaware planner promised about 0 at every age.

**Reading.**

1. At the paper's UC1 setting (σ = 40 s), stale air information costs the
   SLA-aware routers nothing: stale or aware, they had already moved the
   traffic onto the timing-certain backbone (zero failures in 10⁶ draws at
   every age). The deterministic router, which keeps the air relay, goes from
   0.141 to 0.364 violation over an hour. Fresh information is worth latency
   here: knowing the drift, the chance-constrained router takes an air relay
   in 19 to 25% of instances from 5 min on and delivers in about 495 s
   (592 s on the backbone).
2. At σ = 10 s the routers use the air relays, and staleness bites. A planner
   that ignores the age of its information picks the same routes as
   deterministic CGR and does no better than it: 0.26 violation at one hour,
   with a third of the instances in significant breach of a budget it
   believes it meets.
3. Widening the timing law by the age of the information restores the
   budget. The aware chance-constrained router stays at 0.015 or below at
   every age, and its promise matches its delivery in aggregate. It pays in
   delivery time: it keeps the air relay while the budget allows (77 s at
   5 min), mixes at 15 min (38% air, 390 s) and retreats to the backbone from
   30 min on (554 s). The max-P form retreats already at 5 min, since it
   maximises probability whatever the delay.
4. Fresh information gives both: 70 to 107 s at 0.002 or less. Staleness thus
   costs reliability when the planner ignores it and latency when it accounts
   for it, and the refresh rate is the operator's knob between the two.
5. At 5 min the aware router breaches ε on 5% of instances while averaging
   0.015. The chance constraint holds over the planner's own uncertainty,
   which now includes the drift; an instance whose drift landed in the tail
   can exceed ε, and the average over drifts stays within it.

## Caveats

- **Timing is provisional** (contended machine); rerun on an idle machine.
  The timings are CPython on a desktop core, so only ratios between stages
  transfer to flight hardware.
- **What "SABR" means here.** Stage (ii) is the deterministic part of this
  library's candidate generation, one particular k-route strategy (mean
  route, single-contact exclusions, disjoint suppression run to exhaustion
  over the 6 h plan, hence 228 routes on the largest plan). ION's CGR and
  A-SABR enumerate alternatives differently and usually stop earlier; a SABR
  that computes fewer routes would make (ii) cheaper and the (v)/(ii) ratio
  larger. The comparison isolates the distributional increment over the same
  enumeration, and that increment (33 ms to 963 ms shipped, 28 ms to 111 ms
  lean) is the honest headline.
- **Forwarding table** timings are for one flow; the ProxyPlan is built on the
  bundle-lifetime window.
- **Sidecar encoding** is a design sketch (f32 fields, no compression, no
  quantisation). Half-precision or delta coding would shrink it; ION's own
  encoding of the classical plan is not modelled beyond the text line.
- **Drift model.** Independent Gaussian random walk per air contact with
  κ = 1 s/√s chosen by hand (no telemetry fit); σ_eff assumes the
  planner knows κ, which it would have to estimate (T7 territory). A drift
  shared by both contacts of a relay would leave the hand-off slack intact
  and is milder for these two-hop air routes; it was not run.
- **No onboard contingency in the staleness sweep.** Routes are committed end
  to end (the ground-central placement). A proxy's ranked table could absorb
  part of the drift online; not measured here.
- **A timing-certain fallback exists** in UC1 (margin 40 s). Where no certain
  route meets the deadline, staleness would cost reliability for every
  planner, aware or not; not run.

## Sentences for the response to Reviewer 1

The factor of about 1000 in Fig. 11 compares a complete offline route
selection with a single deterministic route search, whereas a SABR router also
computes a table of alternative routes; against that table, the full
SLA-aware selection costs 3.3 times more on our largest plan (14,640
contacts), and 1.3 times once realisations stop re-sampling the deterministic
backbone. None of this runs on the forwarding plane: a node resolves the
precomputed ranked table in about 0.6 µs per decision in our reference
implementation, flat in plan size and roughly 1,300 times cheaper than one
onboard CGR search on the same plan, from a per-flow table of 140 to 490
bytes. On the control plane, point-mass contacts carry no stochastic
annotation, so the UC1 sidecar is 126 bytes whatever the size of the orbital
plan, and even with every contact fully annotated (51 bytes) the sidecar is
1.5 times the ION contact plan it accompanies; refreshing every air and ground
contact of our 3,606-contact multi-hop plan takes 8.2 s at 100 kbit/s. We also
aged the planner's information by up to one hour while the air relays drift:
a planner that ignores the age of its information falls back to the
deterministic router's violation (0.26 against 0.25 at one hour), while one
that widens its timing law by the drift accumulated since its last update
stays within the 5% budget at every age and pays in delivery time, which
fresh information recovers. At the paper's UC1 jitter, staleness costs
nothing at all, because the router has already moved the traffic onto the
timing-certain backbone.
