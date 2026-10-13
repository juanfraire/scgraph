# Generality sweep and the deep placement probe (R3, R1)

Answers R3 ("the main use cases appear to rely on a relatively specific orbital
setup. Additional experiments varying constellation density, topology depth,
source-destination locations, traffic volumes, deadline regimes, uncertainty
levels, and distributional assumptions would help establish whether the
reported improvements generalize") and R1 on the placement probe ("the
distributed planner is 'flattered' by the 2-hop setting").

## What was added

| File | Content |
|---|---|
| `scenarios/uc4_d40_skeleton.yaml`, `scenarios/uc4_d90_skeleton.yaml` | `uc4_skeleton.yaml` with only the Walker code (and the name) changed: `D:550:53:40/4/1` and `D:550:53:90/9/1`. |
| `scenarios/uc4_d40_cpd/`, `scenarios/uc4_d90_cpd/` | The CPD plans. |
| `experiments/run_generality.py` | One-at-a-time sensitivity around the UC4 base point. Flags: `--quick`, `--legacy-cands`, `--workers N` (capped at 3), `--replot`. |
| `experiments/run_placement_uc4.py` | The placement probe on UC4's deep topology. Flags: `--quick`, `--workers N`, `--replot`, `--proxy-diag`. |
| `results/generality.csv`, `results/generality_instances.csv` | One row per factor level; one row per instance and point. |
| `results/generality_legacy.csv`, `results/generality_legacy_instances.csv`, `figures/generality_legacy.{pdf,png}` | The same sweep with the default candidate generator. |
| `results/placement_uc4.csv`, `results/placement_uc4_instances.csv` | One row per margin; one row per instance. |
| `results/placement_uc4_proxydiag.csv` | Why the ranked-table proxy fails on a deep route (`--proxy-diag`). |
| `figures/generality.{pdf,png}` | Dot plot: one panel per factor, one row per level, held-out violation per policy, 95% across-instance intervals. |
| `figures/placement_uc4.{pdf,png}` | Left: the paper's UC1 probe replotted from `results/placement_uc1.csv`. Right: UC4. |

No shared file was modified (`scgraph/`, existing `experiments/`, `Makefile`,
`README.md`). `scripts/verify_results.py` checks a fixed list of CSVs, so the
new files do not affect `make verify`.

## How to run

```bash
cpd run scenarios/uc4_d40_skeleton.yaml -o scenarios/uc4_d40_cpd   # 3.0 s (plans are committed)
cpd run scenarios/uc4_d90_skeleton.yaml -o scenarios/uc4_d90_cpd   # 11.6 s
.venv/bin/python experiments/run_generality.py                     # 437 s
.venv/bin/python experiments/run_generality.py --legacy-cands      # 413 s
.venv/bin/python experiments/run_placement_uc4.py                  # 3190 s
.venv/bin/python experiments/run_placement_uc4.py --proxy-diag     # 213 s
```

Wall times at 3 workers on the M3 Pro, with other experiment tracks sharing the
machine. Instance CPU time: generality 1280 s for 325 instances (mean 3.9 s,
max 14.6 s); placement 9460 s for 100 instances (mean 95 s, max 118 s, of which
the every-node proxy table takes 75 s and the restricted one 16 s). The quick
modes take 38 s and 118 s. Both studies stayed well inside the 90-minute
budget at the instance counts asked for (25 per point, 20 per margin), so no
reduction was needed.

## Setup

### Constellation density

| Shell | Contacts (6 h) | Cross-plane ISLs | Closest same-rank adjacent-plane distance | Contacts per instance | Windows redrawn | Mean D - t0 |
|---|---|---|---|---|---|---|
| 40/4/1 | 1122 | none | 6647 km | 196 | 18 of 43 | 544 s |
| 60/6/1 (base) | 3606 | 930, median 482 s | 4792 km | 460 | 0 | 340 s |
| 90/9/1 | 11124 | 2146, median 2297 s | 3305 km | 1204 | 0 | 326 s |

The ISL range is 5000 km and the `plane-offset: 1` pattern links same-rank
satellites, so the 40-satellite shell gets no cross-plane ISLs: its mesh is
four in-plane rings, and the far terminal reaches the gateway by store, carry
and forward along one plane. In 18 of the 43 drawn 30-minute windows there was
no nominal route at all; those windows were redrawn (`redraws` column), so the
density-40 point is conditioned on a route existing. Both new plans load with
`uc4_continuum.load_skeleton(path)` and the far terminal reaches the gateway
in every accepted window (6.0 nominal hops at 40 satellites, 6.8 at 60 and 90).

### Design

One factor moves at a time around the UC4 base point: far terminal (Dakar),
60 satellites, sigma = 40 s, margin 60 s over the reliable path, bundle 100,
normal jitter. Levels: terminal {near, mid, far} (Barcelona, Rabat, Dakar:
4.0, 4.2 and 6.8 nominal hops), density {40, 60, 90}, margin {0, 60, 240} s,
sigma {20, 40, 60} s, bundle size {50, 100, 200}, jitter family {normal, t3,
uniform} (same standard deviation; planner and truth share the family). That is
13 distinct points of 25 instances each; the base point is computed once and
shared by all six factors.

Common random numbers: instance i uses the seed (SEED, i) at every point, so
the release time and UAV schedule are shared across the levels of a factor and
the level-to-level differences are paired.

### Policies

Copied from `run_uc4.py` (left untouched):

| Key | Policy | Scored on |
|---|---|---|
| `det` | static CGR on the nominal plan | 10 000 held-out draws of the route |
| `scg` | static SLA-aware, argmax P[T <= D] (`pick_route`, 2000 draws per candidate) | 10 000 held-out draws |
| `cgr_react` | reactive CGR (`execute_reactive`, no first route) | 1000 held-out realizations |
| `scg_policy` | SLA-aware with repair: first route by `pick_route_under_repair` (300 planning realizations), then `execute_reactive` | the same 1000 realizations |
| `scg_repair` | (extra) the static SLA-aware route, then CGR repair | the same 1000 realizations |

Selection and evaluation use separate streams; nothing is scored on draws used
to choose it.

Candidates come from `candidate_routes(..., risk_classes=True)`, the library
fix that `run_uc4.py` now uses. On UC4 the safe route starts on the slow
terminal-to-HAPS narrowband link and shares its space contacts with the UAV
routes, so the default disjoint sweep rarely offers it. I found this
independently while checking a puzzling number (the UAV-free route alone
violated about 0.075 on most base instances while the SLA-aware choice violated
about 0.19), and the coordinator's gap study reached the same conclusion. The
sweep was run with both generators (`--legacy-cands`); the legacy run
reproduces my first run cell for cell.

### Statistics

`*_viol` pools the held-out failures over the 25 instances (equal draws per
instance, so it equals the mean of the per-instance rates). Two intervals are
recorded. `*_cp_lo` / `*_cp_hi` are exact two-sided 95% Clopper-Pearson
intervals on the pooled count; they condition on the 25 drawn instances and
are narrow (about ±0.002 for static routes). `*_inst_ci` is the 95% t-interval
half-width over instances, which measures how much the rate depends on which
instances were drawn; it is the one that matters for generality and the one
drawn in the figure. `d_*` columns are instance-paired differences with their
95% t-interval and the count of instances favouring each side.

## Results: generality

### Held-out violation rate (risk-class candidates)

Mean over 25 instances, ± the 95% across-instance half-width. The last three
columns are paired differences (positive means the second-named policy
violates less), with the instance count favouring it out of 25.

| Factor | Level | CGR static | SLA static | CGR + repair | SLA + repair | CGR static minus SLA static | CGR+rep minus SLA+rep | CGR+rep minus SLA static |
|---|---|---|---|---|---|---|---|---|
| terminal | near | 0.352 ±0.095 | 0.085 ±0.011 | 0.108 ±0.011 | 0.072 ±0.006 | +0.268 ±0.092 (25) | +0.037 ±0.010 (23) | +0.024 ±0.016 (21) |
| terminal | mid | 0.313 ±0.066 | 0.086 ±0.010 | 0.112 ±0.012 | 0.078 ±0.010 | +0.228 ±0.065 (25) | +0.034 ±0.010 (23) | +0.027 ±0.012 (21) |
| all | base | 0.372 ±0.082 | 0.093 ±0.015 | 0.108 ±0.011 | 0.073 ±0.007 | +0.279 ±0.077 (25) | +0.035 ±0.009 (23) | +0.015 ±0.020 (19) |
| density | 40 | 0.296 ±0.055 | 0.094 ±0.017 | 0.124 ±0.017 | 0.075 ±0.007 | +0.202 ±0.060 (24) | +0.048 ±0.013 (23) | +0.029 ±0.022 (18) |
| density | 90 | 0.407 ±0.088 | 0.089 ±0.010 | 0.112 ±0.011 | 0.075 ±0.007 | +0.318 ±0.087 (25) | +0.037 ±0.007 (24) | +0.023 ±0.011 (20) |
| margin | 0 s | 0.379 ±0.080 | 0.260 ±0.045 | 0.166 ±0.036 | 0.123 ±0.025 | +0.119 ±0.076 (18) | +0.043 ±0.017 (21) | -0.094 ±0.018 (2) |
| margin | 240 s | 0.371 ±0.082 | 0.002 ±0.002 | 0.086 ±0.011 | 0.002 ±0.001 | +0.369 ±0.082 (25) | +0.084 ±0.011 (25) | +0.083 ±0.011 (25) |
| sigma | 20 s | 0.254 ±0.082 | 0.089 ±0.008 | 0.017 ±0.006 | 0.013 ±0.006 | +0.164 ±0.081 (24) | +0.004 ±0.003 (14, 8 ties) | -0.072 ±0.011 (0) |
| sigma | 60 s | 0.487 ±0.071 | 0.097 ±0.023 | 0.215 ±0.011 | 0.090 ±0.009 | +0.390 ±0.065 (25) | +0.125 ±0.015 (25) | +0.118 ±0.025 (24) |
| size | 50 | 0.316 ±0.070 | 0.021 ±0.021 | 0.107 ±0.014 | 0.016 ±0.009 | +0.295 ±0.074 (25) | +0.091 ±0.013 (25) | +0.086 ±0.020 (24) |
| size | 200 | 0.423 ±0.055 | 0.232 ±0.007 | 0.212 ±0.014 | 0.151 ±0.014 | +0.191 ±0.056 (25) | +0.060 ±0.015 (22) | -0.021 ±0.017 (6) |
| family | t3 | 0.332 ±0.077 | 0.092 ±0.014 | 0.082 ±0.008 | 0.061 ±0.008 | +0.240 ±0.072 (25) | +0.021 ±0.009 (21) | -0.010 ±0.015 (9) |
| family | uniform | 0.391 ±0.087 | 0.093 ±0.015 | 0.116 ±0.013 | 0.075 ±0.006 | +0.298 ±0.083 (25) | +0.041 ±0.010 (24) | +0.023 ±0.020 (20) |

The base row is the far terminal, 60 satellites, margin 60 s, sigma 40 s,
bundle 100, normal jitter, shared by all six factors. Static SLA-aware followed
by CGR repair (`scg_repair`) sits between the two repair policies at most
points (0.090 at the base) and is in the CSV.

### Latency, depth, and what the SLA-aware router picks

Mean delivery time of delivered bundles, relative to release (seconds). "Takes
the safe route" is the share of instances in which the static SLA-aware choice
is the UAV-free CGR route that defines the deadline.

| Factor | Level | D - t0 | CGR static | SLA static | CGR + repair | SLA + repair | Hops CGR / SLA | Uncertain hops CGR / SLA | Takes the safe route |
|---|---|---|---|---|---|---|---|---|---|
| terminal | near | 331 | 151 | 279 | 178 | 223 | 4.0 / 3.5 | 4.0 / 2.3 | 56% |
| terminal | mid | 331 | 152 | 280 | 178 | 223 | 4.2 / 3.8 | 4.0 / 2.4 | 80% |
| all | base | 340 | 156 | 289 | 185 | 213 | 6.8 / 7.2 | 5.0 / 3.7 | 64% |
| density | 40 | 544 | 397 | 494 | 414 | 447 | 6.0 / 5.3 | 4.7 / 2.8 | 88% |
| density | 90 | 326 | 154 | 274 | 180 | 199 | 6.8 / 6.5 | 5.2 / 3.6 | 48% |
| margin | 0 s | 280 | 156 | 187 | 186 | 174 | 6.8 / 8.2 | 5.0 / 5.2 | 0% |
| margin | 240 s | 520 | 156 | 310 | 186 | 301 | 6.8 / 7.5 | 5.0 / 3.5 | 28% |
| sigma | 20 s | 340 | 155 | 275 | 173 | 171 | 6.8 / 7.4 | 5.0 / 3.5 | 48% |
| sigma | 60 s | 340 | 155 | 289 | 198 | 274 | 6.8 / 7.2 | 5.0 / 3.5 | 68% |
| size | 50 | 194 | 107 | 140 | 123 | 141 | 7.4 / 6.8 | 5.2 / 3.2 | 52% |
| size | 200 | 615 | 275 | 537 | 331 | 343 | 6.8 / 6.9 | 4.8 / 3.5 | 72% |
| family | t3 | 340 | 155 | 289 | 180 | 182 | 6.8 / 7.2 | 5.0 / 3.5 | 68% |
| family | uniform | 340 | 156 | 289 | 187 | 218 | 6.8 / 7.5 | 5.0 / 3.5 | 64% |

### What the candidate generator changes

| Factor | Level | SLA static, default -> risk classes | SLA + repair, default -> risk classes | CGR+rep minus SLA static, default | Same, risk classes | Safe route offered, default |
|---|---|---|---|---|---|---|
| terminal | near | 0.188 -> 0.085 | 0.079 -> 0.072 | -0.076 ±0.018 | +0.024 ±0.016 | 4% |
| terminal | mid | 0.177 -> 0.086 | 0.085 -> 0.078 | -0.061 ±0.028 | +0.027 ±0.012 | 20% |
| all | base | 0.189 -> 0.093 | 0.079 -> 0.073 | -0.077 ±0.020 | +0.015 ±0.020 | 8% |
| density | 40 | 0.192 -> 0.094 | 0.097 -> 0.075 | -0.068 ±0.031 | +0.029 ±0.022 | 12% |
| density | 90 | 0.194 -> 0.089 | 0.075 -> 0.075 | -0.083 ±0.022 | +0.023 ±0.011 | 12% |
| margin | 0 s | 0.260 -> 0.260 | 0.119 -> 0.123 | -0.094 ±0.018 | -0.094 ±0.018 | 8% |
| margin | 240 s | 0.166 -> 0.002 | 0.049 -> 0.002 | -0.082 ±0.023 | +0.083 ±0.011 | 8% |
| sigma | 20 s | 0.124 -> 0.089 | 0.014 -> 0.013 | -0.107 ±0.009 | -0.072 ±0.011 | 0% |
| sigma | 60 s | 0.209 -> 0.097 | 0.125 -> 0.090 | +0.010 ±0.047 | +0.118 ±0.025 | 32% |
| size | 50 | 0.027 -> 0.021 | 0.016 -> 0.016 | +0.078 ±0.026 | +0.086 ±0.020 | 76% |
| size | 200 | 0.272 -> 0.232 | 0.150 -> 0.151 | -0.063 ±0.020 | -0.021 ±0.017 | 16% |
| family | t3 | 0.188 -> 0.092 | 0.059 -> 0.061 | -0.103 ±0.017 | -0.010 ±0.015 | 4% |
| family | uniform | 0.189 -> 0.093 | 0.083 -> 0.075 | -0.072 ±0.020 | +0.023 ±0.020 | 12% |

With risk classes the safe route is offered in every instance at every point.

### Reading

**Where the SLA-aware advantage holds.** Against the same execution model it
holds at all 13 points. Static SLA-aware beats static CGR by 0.12 to 0.39, and
SLA-aware with repair beats reactive CGR by 0.004 to 0.125, with the paired 95%
interval excluding zero at every point, under both candidate generators.
Topology depth and source-destination distance barely matter: the near, mid
and far terminals (4 to 7 hops) give SLA-aware static rates of 0.085, 0.086 and
0.093. Neither does constellation density (0.094, 0.093, 0.089 for 40, 60 and
90 satellites) or the jitter law (0.092 to 0.093 for normal, t3 and uniform).
What these factors move is the CGR baseline: static CGR goes from 0.30 at 40
satellites to 0.41 at 90 (its routes carry 4.7 and 5.2 uncertain hops). The
SLA-aware rate follows whether a timing-certain route exists with slack, which
the orbital geometry hardly changes.

**Where it shrinks.** Three regimes. (1) Zero margin: the deadline equals the
safe route's nominal arrival, so that route is never chosen (0%) and the
static gap falls from 0.28 to 0.12 (18 of 25 instances). (2) Large bundles
(200): the safe route's 400 s narrowband hop and 100 s feeder transfer become
fragile themselves; SLA static rises to 0.232 and the static gap to 0.19.
(3) Low jitter (sigma = 20 s): reactive CGR already delivers 98.3%, and
repair-aware selection adds only 0.004.

**Where it reverses.** Never against the same execution model. It does reverse
when a static SLA-aware source route is compared with reactive CGR, a
different execution model. Static planning loses to hop-by-hop CGR repair at
zero margin (by 0.094, 23 of 25 instances), at sigma = 20 s (by 0.072, all 25)
and at bundle 200 (by 0.021); it ties under t3 jitter and at the base point
(+0.015 ±0.020), and wins at sigma = 60 s (+0.118), bundle 50 (+0.086) and
margin 240 s (+0.083). Planning pays when jitter or slack is large; repair pays
when the deadline is tight or jitter is small. Combining them (SLA-aware with
repair) gives the lowest violation at all 13 points.

**Latency.** The static SLA-aware route pays for reliability in delivery time:
289 s against 156 s for CGR at the base (deadline 340 s), because it takes the
slow narrowband first hop in 64% of instances. With repair the premium is 28 s
(213 against 185 s), and at margin 0 SLA-aware with repair is 12 s faster than
reactive CGR (174 against 186 s).

**The candidate generator is material for one comparison.** With the default
generator the safe route was offered in 8% of base instances, the static
SLA-aware rate was 0.189 instead of 0.093, and static SLA-aware lost to
reactive CGR at 11 of the 13 points (one win, one tie). With risk classes it
wins significantly at 8 points, loses at 3 (margin 0, sigma 20, bundle 200)
and ties at 2. The other conclusions do not depend on the generator: SLA-aware
beats CGR under both execution models at every point either way, and the
SLA-aware-with-repair rates move by at most 0.047 (margin 240: 0.049 to 0.002)
and usually by less than 0.01, because repair recovers most of what a missing
candidate costs.

**Consistency.** The base point agrees with `results/uc4_margin.csv` at
margin 60 (CGR static 0.393, SLA static 0.085, CGR + repair 0.127, SLA + repair
0.081 on 30 different instances).

### Sentences for the paper (generality)

To test whether the gains depend on the orbital setup, we varied one factor at
a time around the UC4 base point, with 25 instances per point scored on
held-out draws: terminal distance (4 to 7 hops), constellation size (40, 60 and
90 satellites, the sparsest shell having no cross-plane ISLs), deadline margin
(0 to 240 s), UAV jitter (20 to 60 s), bundle size (50 to 200) and the jitter
law (normal, Student-t with 3 degrees of freedom, uniform). Against the same
execution model, SLA-aware routing lowered the violation rate at all 13 points,
by 0.12 to 0.39 for static source routes and by 0.004 to 0.125 when every node
repairs with CGR, with paired 95% intervals excluding zero throughout.
Distance, depth, density and the jitter law moved the SLA-aware rate by less
than 0.01 (0.085 to 0.094); the gain shrank when the deadline left no slack
over the timing-certain path and when large bundles made that path itself
fragile. Whether source-side planning or hop-by-hop repair matters more
depends on the regime: a static SLA-aware route beat reactive CGR at high
jitter (0.10 against 0.22 at sigma = 60 s) and lost to it at low jitter or
zero margin, while combining both was best at every point.

## Results: placement on the deep topology

Far terminal, 60 satellites, sigma = 40 s, 20 instances per margin, 1000
held-out realizations per instance shared by every policy. Ground, proxy and
distributed run through `placement.execute(..., max_hops=20)`; the references
through `execute_reactive`. Every candidate set uses risk classes; `plan_proxy`
calls the default generator internally, so the every-node proxy is a local copy
of it with the fix (same nodes, ranking and budget).

The restricted proxy keeps a table only for the nodes on the source's
candidate routes (34.5 nodes and about 133 table entries on average, against
about 271 entries for every node), and its table routes are drawn on the
sub-network those nodes span. A first version that restricted the nodes but
drew routes on the full plan sent bundles to nodes without an entry in up to
59% of realizations on a test instance; the sub-network version leaves the
bundle without an entry in at most 0.1%.

### Delivery ratio by the deadline

Mean over 20 instances, ± the 95% across-instance half-width.

| Margin | D - t0 | Ground | Proxy (restricted) | Proxy (every node) | Distributed | CGR + repair (ref.) | SLA + repair (ref.) | Ground + CGR repair |
|---|---|---|---|---|---|---|---|---|
| -60 s | 218 | 0.633 ±0.037 | 0.580 ±0.083 | 0.607 ±0.063 | 0.000 | 0.661 ±0.050 | 0.743 ±0.042 | 0.695 ±0.042 |
| 0 s | 278 | 0.748 ±0.038 | 0.662 ±0.105 | 0.661 ±0.117 | 0.000 | 0.825 ±0.037 | 0.876 ±0.028 | 0.849 ±0.031 |
| 60 s | 338 | 0.908 ±0.017 | 0.623 ±0.186 | 0.574 ±0.189 | 0.000 | 0.875 ±0.016 | 0.924 ±0.009 | 0.914 ±0.011 |
| 120 s | 398 | 0.974 ±0.014 | 0.786 ±0.188 | 0.587 ±0.230 | 0.000 | 0.889 ±0.015 | 0.976 ±0.011 | 0.976 ±0.013 |
| 240 s | 518 | 0.994 ±0.006 | 0.555 ±0.234 | 0.599 ±0.228 | 0.000 | 0.905 ±0.014 | 0.998 ±0.002 | 0.997 ±0.003 |

Placement order at every margin: ground > proxy (either) > distributed.

| Margin | Reached the gateway at all: ground / proxy / distributed (20 hops) / distributed (200 hops) | Proxy minus ground (paired; instances proxy better / worse) | CGR + repair minus ground (paired) |
|---|---|---|---|
| -60 s | 0.754 / 0.698 / 0.000 / 0.015 | -0.053 ±0.072 (8 / 11) | +0.028 ±0.034 (16 / 4) |
| 0 s | 0.768 / 0.696 / 0.000 / 0.015 | -0.086 ±0.105 (10 / 10) | +0.077 ±0.019 (19 / 1) |
| 60 s | 0.968 / 0.702 / 0.000 / 0.015 | -0.286 ±0.189 (2 / 14) | -0.033 ±0.025 (4 / 16) |
| 120 s | 0.988 / 0.795 / 0.000 / 0.015 | -0.189 ±0.188 (3 / 7) | -0.085 ±0.024 (1 / 19) |
| 240 s | 0.997 / 0.555 / 0.000 / 0.015 | -0.440 ±0.234 (1 / 11) | -0.089 ±0.018 (1 / 19) |

### Why the proxy fails here (`--proxy-diag`, 4 instances, 300 realizations)

| Margin | Ground | Proxy, 4 buckets (as in `placement.py`) | 4 buckets, loop-free | 16 buckets | 16 buckets, loop-free | 4-bucket walks stuck in a loop | 4-bucket walks stuck on closed entries |
|---|---|---|---|---|---|---|---|
| 60 s | 0.907 | 0.453 | 0.656 | 0.914 | 0.898 | 50.0% | 2.0% |
| 240 s | 0.973 | 0.228 | 0.487 | 0.740 | 0.969 | 27.3% | 50.0% |

Two mechanisms, both absent on a two-hop route. Each node's table is ranked on
its own, at the start of its time bucket, so node u can send the bundle to v
and v's best entry can send it back; after a few bounces every top-3 entry
starts with a contact already used and the bundle is stuck. And the four time
buckets span [t0, D], so at margin 240 s a bucket is 130 s wide while the
narrowband first hop alone takes 200 s: the entries at the HAPS and at the
first satellite, ranked at the bucket start, begin with contacts that have
closed by the time the bundle arrives. The
widening buckets explain why the proxy gets worse as the margin grows. Per
instance the outcome is bimodal (at 240 s, 9 instances below 0.2 and 11 above
0.8), hence the wide intervals. With 16 buckets and loop avoidance the proxy
recovers to the ground route's level, and on these instances it does not
exceed it.

### Reading

**The distributed placement does not survive depth.** It delivered no bundle
by the deadline at any margin, and only 1.5% of bundles reached the gateway at
all within 200 hops, against 0.96 to 0.97 on UC1. Ranking local contacts by
earliest arrival at the next hop, with no notion of the destination, turns
into a walk along 2 s in-plane ISL hops: a traced walk went terminal, satellite
259, 258, 257, ..., 254, back up to 259, down to the terminal, through a UAV
and the HAPS, and up again, without ever taking a 50 s feeder pass. On a
two-hop route every satellite's only useful outgoing contact goes to the
gateway, so the same rule cannot go wrong; R1's "flattered" is borne out.

**The UC1 inversion does not survive in its original form.** On UC1 the
distributed placement overtakes the global ones once the margin turns
negative. On UC4 it is last at every margin, and the order ground > proxy >
distributed never changes. A related inversion does survive among the
references: reactive CGR, which is distributed execution in which every node
holds the nominal plan, beats the committed ground route at margins of 0 s and
below (+0.077 at 0 s, 19 of 20 instances) and loses to it once there is slack
(-0.085 at 120 s, 19 of 20). Adaptivity in the forwarding plane wins at tight
deadlines, provided the nodes keep a global view of the plan.

**The proxy's contingency does not help on a deep route.** On UC3 the ranked
table beat the single ground route (0.67 against 0.49 at rho = 0.4). On UC4 it
fell below the ground route at every margin, for the reasons above, and the
unrestricted table was no better than the restricted one (the paired
difference is within ±0.05 except at 120 s, -0.20 ±0.19), so the restriction
is not what drives the result. The contingency that works on UC4 is CGR repair
on top of a good first route: ground plus CGR repair gains 0.06 to 0.10 over
the ground route at margins of 0 s and below, and SLA-aware with repair is the
best policy at every margin (0.743 to 0.998).

**Semantics.** Under `placement.execute` the proxy and distributed choosers
inspect the realized contact before committing (an absent contact, or one whose
window closes before the transfer ends, is skipped at no cost), while the
ground route and the reactive references discover a failure only when it
happens. The proxy and distributed placements lose despite that advantage.

### Sentences for the paper (placement)

The two-hop probe of Sec. VII-F does flatter the distributed placement. On UC4,
where routes span 6 to 9 contacts, a planner that ranks local contacts by
earliest arrival at the next hop delivered no bundle by the deadline at any
margin (1.5% reached the gateway at all within 200 hops), against 0.96 to 0.97
on UC1: with no downstream view it follows 2 s inter-satellite hops around the
ring and seldom reaches a feeder pass. The ranked-table proxy also fell below
the single ground route at every margin, because independently ranked
per-node tables send bundles round loops and four time buckets are too coarse
for a 200 s first hop; with 16 buckets and loop avoidance it only matches the
ground route. What survives from the two-hop result is that adaptivity wins at
tight deadlines when the forwarding nodes hold the nominal plan: reactive CGR
beat the committed ground route at zero margin (0.83 against 0.75) and lost to
it with 120 s of slack (0.89 against 0.97), and SLA-aware selection combined
with hop-by-hop repair was the best policy at every margin.

## Caveats

- One factor at a time: interactions (for example zero margin with high
  jitter) are not explored.
- 25 instances per point and 20 per margin. The across-instance intervals are
  the relevant uncertainty; the exact Clopper-Pearson intervals in the CSVs
  condition on the drawn instances and are far narrower.
- The density axis changes the topology class as well as the density: the
  40-satellite shell has no cross-plane ISLs, needs store, carry and forward,
  and 18 of 43 windows without any route were redrawn.
- Bundle size also moves the deadline, since the reliable path's narrowband
  hop takes size / 0.5 s. It is the volume of one bundle; concurrent flows are
  in `results/notes/flows.md`.
- The jitter family applies to the UAV start law only, at equal standard
  deviation; the duration and rate laws and the ISL acquisition law keep their
  defaults.
- The static SLA-aware numbers depend on the candidate generator (both
  versions reported above); everything here uses risk classes unless marked
  "default".
- The proxy diagnostic covers 4 instances and two margins and is indicative.
  The restricted proxy is my construction (nodes on the source's candidate
  routes, routes on the sub-network they span); the every-node variant checks
  it.
