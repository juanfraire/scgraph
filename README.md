# scgraph: stochastic contact graphs

Companion library and reproduction artefact for

> J. A. Fraire, I. Guérin Lassous, T. Begin, E. Lavinal,
> "Stochastic Contact Graphs: Rethinking the Contact Plan for the 6G
> Space-Air-Ground Continuum", IEEE Open Journal of the Communications
> Society, Special Issue on Delay-Tolerant Networking for 6G (under review).

The paper generalises the DTN contact plan so that each contact carries a joint
law over its start time, end time and capacity, and turns routing into a
chance-constrained deliver-by-deadline problem. Deterministic CGR/SABR (point
masses) and the discrete on/off "uncertain DTN" model (a Bernoulli marginal)
are corners of that object. This repository implements the object, both
corners, and every measurement the paper reports.

Everything in the paper regenerates with one command, in about 11 minutes:

```bash
make install-repro     # pinned dependency versions
make reproduce         # 9 CSVs, and all 7 of the paper's result figures
make verify            # bit-for-bit diff against the recorded reference
```

---

## Table of contents

- [Revision (OJCOMS-07102-2026)](#revision-ojcoms-07102-2026)
- [Quick start](#quick-start)
- [Paper → code map](#paper--code-map) (start here if you are holding the PDF)
- [Reproducing the results](#reproducing-the-results)
- [The correctness anchor](#the-correctness-anchor-a-sabr)
- [Repository layout](#repository-layout)
- [Before you change anything](#before-you-change-anything)
- [How to cite](#how-to-cite)

---

## Revision (OJCOMS-07102-2026)

The submission received a Reject-Resubmit decision on 2026-10-12. Everything
below was added for the resubmission. The submitted experiments keep their
selection streams draw for draw, so every movement in their numbers comes from
the one methodological change listed first. `results/reference-v1/` holds the
CSVs of the submitted manuscript.

```bash
make reproduce            # the submitted figures, with held-out evaluation
make reproduce-revision   # every new experiment below
```

### Method changes

| Change | Why | Where |
|---|---|---|
| Held-out evaluation of every policy | the submitted runners scored the SLA-aware route on the draws that selected it, so its violation was optimistic and Fig. 5's "achieved <= budget" held by construction | `verify/montecarlo.py` `evaluate_route`; `run_uc1.py`, `run_eps.py`, `run_ablation.py`, `run_uc2.py` (columns `*_insample` / `*_v1` keep the old numbers) |
| Exact confidence bounds and admission on a lower bound | "certification" is now a statement at confidence 1 - delta: a route is admitted when its Bonferroni-corrected Clopper-Pearson lower bound clears 1 - eps | `scgraph/stats.py`; `pick_route_chance(admission="lcb")` |
| Joint law within a contact; dependence across contacts | the paper's object is a joint law; the submitted experiments exercised one coordinate at a time | `contact.py` (`corr` copula over start/duration/rate; `factor` common factor), `distributions.py` (`ppf`, `TruncStudentT`) |
| Risk-class candidates | on a meshed backbone the default generator misses the safe route in most instances | `candidate_routes(..., risk_classes=True)`; `results/notes/gap.md` |
| Hop-by-hop repair | a deployed CGR node re-routes when a contact fails; deep routes need that baseline | `routing/reactive.py` (`execute_reactive`, `pick_route_under_repair`) |
| Batched realizations, graph windows | speed on large plans | `graph.py` (`realize_batch`, `window`, indexed `by_id`) |

### New experiments

| Reviewer request | Experiment | Outputs | Note |
|---|---|---|---|
| Deeper multi-hop continuum, combined facets (R1, R2, R3) | `run_uc4.py` on `scenarios/uc4_cpd` (60 satellites with ISLs) | `uc4_sigma.csv`, `uc4_margin.csv`, `uc4.pdf` | this README |
| Jitter compounding along a route (R1) | `run_compound.py` | `compound.csv`, `compound.pdf` | |
| Fair-design artefact (R2) | `run_uc1.py --skeleton scenarios/uc1_raw --tag nonfair` | `uc1_nonfair.csv`, `uc1_h1_nonfair.pdf` | |
| Coupled example for Fig. 1 (R2) | `plot_joint_law.py` | `joint_law.csv`, `joint_law.pdf` | |
| BRUF / RUCoP comparison (R2) | `run_rucop.py` (`routing/rucop.py`) | `rucop_uc1.csv`, `rucop_uc3.csv`, `rucop_tractability.csv` | `results/notes/rucop.md` |
| Concurrent flows, residual volume, buffers (R1, R2) | `run_flows.py` (`flows.py`) | `flows_load.csv`, `flows_buffer.csv`, `flows.pdf` | `results/notes/flows.md` |
| Misspecified laws (R2) and dependence (R2, R3) | `run_misspec_timing.py`, `run_misspec_capacity.py`, `run_dependence.py` | `robust_*.csv`, `robust_*.pdf` | `results/notes/robustness.md` |
| Learning the laws (R2) | `run_estimation.py` | `estimation.csv`, `estimation.pdf` | `results/notes/robustness.md` |
| Compute, forwarding cost, control plane (R1) | `run_overhead.py`, `run_sidecar.py` | `overhead_*.csv`, `overhead_compute.pdf` | `results/notes/overhead.md` |
| Planner staleness (R1) | `run_staleness.py` | `staleness.csv`, `staleness.pdf` | `results/notes/overhead.md` |
| Optimality gap of candidate generation (R1, R2, R3) | `run_gap.py` (`routing/enumerate.py`) | `gap_uc1*.csv`, `gap_uc4*.csv`, `gap.pdf` | `results/notes/gap.md` |
| Generality and deep-topology placement (R1, R3) | `run_generality.py`, `run_placement_uc4.py` | `generality.csv`, `placement_uc4.csv` | `results/notes/generality.md` |

New CPD plans: `scenarios/uc1_raw` (UC1 without fair design), `scenarios/uc4_cpd`
and the density variants used by the generality sweep. `scripts/generate_skeletons.sh revision`
regenerates them.

---

## Quick start

```bash
git clone <this repo> && cd scgraph
make install            # venv + package, loose dependency bounds
make test               # 66 tests; 21 are the A-SABR anchor and skip by default
make reproduce          # regenerate results/*.csv and figures/*.{pdf,png}
make verify             # confirm nothing moved
```

Python 3.11+ (the results were recorded on 3.14.4). `make help` lists every
target. Reproducing the paper needs neither a Rust toolchain nor a Contact Plan
Designer install, because the orbital skeletons are committed.

---

## Paper → code map

Every equation, figure and hypothesis, and where it lives.

### Equations

| Eq. | Paper, what it says | Implemented in | Exercised by |
|---|---|---|---|
| (1) | The stochastic contact: joint law over $(T_s, T_e, V)$, possibly defective | [`scgraph/contact.py`](scgraph/contact.py) `StochasticContact`; laws in [`scgraph/distributions.py`](scgraph/distributions.py) | [`tests/test_distributions.py`](tests/test_distributions.py) |
| (2) | $\tau_i=\max(A_{i-1}, T_s^{c_i})$, transmission may start | [`scgraph/arrival.py:21`](scgraph/arrival.py#L21) (scalar), [`:57`](scgraph/arrival.py#L57) (vectorised) | [`tests/test_arrival.py`](tests/test_arrival.py) |
| (3) | $E_i$, completion when cumulative volume reaches $\omega_m$ | [`scgraph/arrival.py:22`](scgraph/arrival.py#L22) | [`tests/test_arrival.py`](tests/test_arrival.py) |
| (4) | $A_i=E_i+\ell_{c_i}$, $\mathsf{T}_\pi=A_k$; $\infty$ on infeasibility | [`scgraph/arrival.py:17`](scgraph/arrival.py#L17) `delivery_time`, [`:25`](scgraph/arrival.py#L25) | [`tests/test_arrival.py`](tests/test_arrival.py) |
| (5) | $S_\pi(m)$, $P_{\mathrm{succ}}(\pi,m)=\Pr[\mathsf{T}_\pi\le D_m]$ | [`scgraph/routing/quantile.py:69`](scgraph/routing/quantile.py#L69) `route_stats` | [`tests/test_quantile.py`](tests/test_quantile.py) |
| (6) | $\pi^\star=\arg\max_\pi \Pr[\mathsf{T}_\pi\le D_m]$ | [`scgraph/routing/quantile.py:91`](scgraph/routing/quantile.py#L91) `pick_route` | UC1, UC3, ablation |
| (7) | Chance-constrained form: $\min \mathrm{cost}$ s.t. $\Pr\ge 1-\varepsilon$ | [`scgraph/routing/quantile.py:118`](scgraph/routing/quantile.py#L118) `pick_route_chance` | $\varepsilon$-sweep (Fig. 5) |
| (8) | $Q_{1-\varepsilon}(\pi,m)$, the quantile delivery time | [`quantile.py:82`](scgraph/routing/quantile.py#L82) (MC); [`scgraph/analytic.py:76`](scgraph/analytic.py#L76) `route_quantile` (convolution) | [`tests/test_quantile.py`](tests/test_quantile.py); the two agree within CI |
| (9) | $\Pr\ge 1-\varepsilon \iff Q_{1-\varepsilon}\le D_m$ | the equivalence `pick_route_chance` relies on: it filters on $P_{\mathrm{succ}}$ and minimises latency | Fig. 5 (left) measures this claim |
| (10) | $\mathrm{cost}_\varepsilon(\pi,m)=Q_{1-\varepsilon}$, the SABR drop-in key | [`quantile.py`](scgraph/routing/quantile.py) `route_stats["quantile"]` | [`tests/test_quantile.py`](tests/test_quantile.py) |
| (11) | Capacity chance constraint $\Pr[\sum_{c}V^c\ge\omega]\ge 1-\varepsilon$ | [`scgraph/scenarios/uc2_logistics.py`](scgraph/scenarios/uc2_logistics.py) `provisioning_cpd` | UC2 (Fig. 6) |
| (12) | Independent replication $1-\prod(1-P_{\mathrm{succ}})$ | [`scgraph/routing/replication.py:34`](scgraph/routing/replication.py#L34) `independent_success` | [`tests/test_replication.py`](tests/test_replication.py) |
| (13) | Correlated replication $\mathbb{E}_z[\,\cdot\,]$ under a shared latent state | [`replication.py:18`](scgraph/routing/replication.py#L18) `joint_success`, via [`arrival.py:64`](scgraph/arrival.py#L64) `joint_route_set_deliveries`; latent model in [`graph.py:18`](scgraph/graph.py#L18) | UC3 (Fig. 7) |
| (14) | Bernoulli projection $q_c[k]$ | [`scgraph/routing/bernoulli.py:54`](scgraph/routing/bernoulli.py#L54) `slot_usability` | ablation (Fig. 3), [`tests/test_bernoulli.py`](tests/test_bernoulli.py) |

### Figures

| Fig. | Paper | Produced by | Data | Image |
|---|---|---|---|---|
| 1 | The stochastic contact and its two corners | TikZ in `main.tex` (schematic, no code) | n/a | n/a |
| 2 | Populating the graph (pipeline) | TikZ in `main.tex` (schematic) | n/a | see [`docs/PIPELINE.md`](docs/PIPELINE.md) |
| 3 | Bernoulli corner vs slot width $\Delta$ | [`experiments/run_ablation.py`](experiments/run_ablation.py) | `results/ablation_slots.csv` | `figures/subsumption.pdf` |
| 4 | UC1 / H1: violation + latency vs jitter $\sigma$ | [`experiments/run_uc1.py`](experiments/run_uc1.py) | `results/uc1.csv` | `figures/uc1_h1.pdf` |
| 5 | The $\varepsilon$ knob | [`experiments/run_eps.py`](experiments/run_eps.py) | `results/eps.csv` | `figures/eps.pdf` |
| 6 | UC2 / H2: provisioned overhead vs fade | [`experiments/run_uc2.py`](experiments/run_uc2.py) | `results/uc2.csv` | `figures/uc2_h2.pdf` |
| 7 | UC3 / H3: delivery ratio vs collapse $\rho$ | [`experiments/run_uc3.py`](experiments/run_uc3.py) | `results/uc3.csv` | `figures/uc3_h3.pdf` |
| 8 | Placement probe | [`experiments/run_placement.py`](experiments/run_placement.py) | `results/placement_uc1.csv`, `results/placement_uc3.csv` | `figures/placement.pdf` |
| 9 | Route-selection runtime vs plan size | [`experiments/run_runtime.py`](experiments/run_runtime.py) | `results/runtime.csv` | `figures/runtime.pdf` |
| n/a | §V-F's MC-vs-convolution crossover (no paper figure) | [`experiments/run_mc_scaling.py`](experiments/run_mc_scaling.py) | `results/mc_scaling.csv` | `figures/mc_scaling.pdf` |

Figures land in `figures/` and are copied into the paper's `img/` by hand.

### Hypotheses

| | Claim | Scenario | Experiment | Headline in the paper |
|---|---|---|---|---|
| H1 | Distributional $T_s,T_e$ + max-$\Pr[\text{deliver by }D]$ routing lowers the SLA-violation rate vs a deterministic plan, and the gap widens with jitter | [`scenarios/uc1_event.py`](scgraph/scenarios/uc1_event.py) `make_instance_cpd` | `run_uc1.py` | at $\sigma{=}60$ s: det 0.2856 → SLA-aware 0.0; latency 69 → 549 s; Bernoulli corner at $\Delta{=}30$ s 0.0277 |
| H2 | Chance-constrained capacity sizing meets the SLA at lower provisioned redundancy than worst-case | [`scenarios/uc2_logistics.py`](scgraph/scenarios/uc2_logistics.py) `provisioning_cpd` | `run_uc2.py` | mean-case SLA 0.63–1.00 (violates); worst-case overhead 1.20 → 5.02; chance-constrained 1.20 → 1.99 ($2.5\times$ saving) |
| H3 | Replication budgeted by existence *and correlation* beats a stale plan and correlation-blind flooding at equal overhead | [`scenarios/uc3_emergency.py`](scgraph/scenarios/uc3_emergency.py) | `run_uc3.py` | at $\rho{=}0.5$: stale 0.41, naive 0.49, Bernoulli 0.59, SCG 0.65, epidemic 0.76 |

Three cross-cutting studies accompany them. The subsumption ablation (§VII-B)
traces the path between the two corners: at $\sigma{=}60$ s it sits near zero on
fine grids and rises to 0.137 at $\Delta{=}90$ s, against deterministic 0.275.
The $\varepsilon$-sweep (§VII-D) finds achieved violation at or below budget at
every $\varepsilon$, with latency falling from 520 s to 82 s. The placement
probe (§VII-F) gives 0.49 / 0.67 / 0.71 for UC3 at $\rho{=}0.4$, a factor of
1.45, and the UC1 ordering inverts once the deadline margin reaches zero. The
runtime study (§VII-H) spans 98 to 14 640 contacts, with per-route scoring flat
at about 0.75 ms while candidate generation grows from 31 ms to 1.39 s.

### Model assumptions

| | Assumption | Where it lives in the code |
|---|---|---|
| A1 | Deterministic orbital backbone (point-mass timing) | [`scgraph/io/cpd.py`](scgraph/io/cpd.py) loads CPD exports as all-`Dirac` contacts |
| A2 | Parametric marginals: TruncNormal timing, LogNormal volume | [`scgraph/distributions.py`](scgraph/distributions.py); laws sit on start and duration, so A5 holds by construction |
| A3 | Factored existence × timing | `StochasticContact.p_exist` alongside the laws ([`contact.py`](scgraph/contact.py)) |
| A4 | Independence baseline; latent-state correlation where it is the object of study | [`graph.py`](scgraph/graph.py) `LatentModel`; shared draws in [`arrival.py`](scgraph/arrival.py) `joint_route_set_deliveries` |
| A5 | Feasibility $T_s\le T_e$, transfer fits the window | enforced by construction (duration law) and by the `e > c.t_end` test in [`arrival.py`](scgraph/arrival.py) |
| A6 | Local stationarity | no re-estimation inside a horizon; laws are fixed per instance |

---

## Reproducing the results

```bash
make reproduce      # runs all eight experiments in order
make verify         # diffs results/*.csv against results/reference/
```

Or one at a time (`make run-uc1`, `make run-placement`, …). Every runner takes
`--quick` for a reduced sweep. That is handy for smoke-testing a change, though
it does not produce the paper's numbers.

Measured wall time, single process, on an Apple M3 Pro running macOS 26.5.2,
Python 3.14.4 and numpy 2.5.1:

| Experiment | Time | Output |
|---|---|---|
| `run_uc1.py` | 53 s | `uc1.csv`, `uc1_h1.{pdf,png}` |
| `run_uc2.py` | 3 s | `uc2.csv`, `uc2_h2.{pdf,png}` |
| `run_uc3.py` | 13 s | `uc3.csv`, `uc3_h3.{pdf,png}` |
| `run_ablation.py` | 97 s | `ablation_slots.csv`, `subsumption.{pdf,png}` |
| `run_eps.py` | 22 s | `eps.csv`, `eps.{pdf,png}` |
| `run_placement.py` | 346 s | `placement_uc1.csv`, `placement_uc3.csv`, `placement.{pdf,png}` |
| `run_runtime.py` | 135 s | `runtime.csv`, `runtime.{pdf,png}` |
| `run_mc_scaling.py` | 5 s | `mc_scaling.csv`, `mc_scaling.{pdf,png}` |
| total | ≈11 min | |

`make verify` compares against `results/reference/`, which holds the CSVs the
manuscript was written from. On the recorded environment the match is
bit-identical, and any drift should be treated as a bug until proven otherwise.
On a different platform, use `python scripts/verify_results.py --tol 1e-9`.

Two runners time the machine as well as the model, so their wall-clock columns
are exempt from the comparison and will differ on every run: `run_runtime.py`
and `run_mc_scaling.py`. What they support is the shape of a curve (scoring
flat in plan size; MC linear in $M$ against a flat convolution), so the
millisecond values carry no claim. Both take `--replot` to redraw from the
committed CSV without re-measuring, which keeps `figures/runtime.pdf` identical
to the copy in the paper.

---

## The correctness anchor (A-SABR)

The paper's subsumption claim has two halves. The Bernoulli half is a lossy
projection, so it is measured (Fig. 3). The deterministic half is an equality,
and the test suite enforces it: on point-mass graphs `scgraph`'s router must
return the same routes and delivery times as
[A-SABR](https://github.com/DTN-MTP/A-SABR), compared over 20 randomised plans.

```bash
make anchor    # build the Rust CLI, then run the cross-check for real
```

The anchor is optional. Nothing in `results/` or `figures/` depends on it, so a
clone without a Rust toolchain still reproduces the whole paper, and the anchor
tests skip with a message saying how to enable them. Setting
`SCGRAPH_REQUIRE_ANCHOR=1` turns that skip into a failure. CI does this, so the
anchor cannot be silently lost.

### Which A-SABR, exactly

A-SABR is an ordinary pinned git dependency
([`rust/asabr-cli/Cargo.toml`](rust/asabr-cli/Cargo.toml)), with nothing
vendored or patched. The anchor was checked against this revision:

| | |
|---|---|
| Repository | `https://github.com/DTN-MTP/A-SABR.git` |
| Revision | `1277e360f2d61a593c336c6f6eb2e583758d7580` |
| Dated | 2026-06-30 (`ops(ci): add manual workflow`) |
| Recorded in | `Cargo.toml` (`rev =`) and the committed `Cargo.lock` |

Upstream has since moved on, and the pin stays where it is on purpose. As of
2026-07-28, A-SABR's `main` is 73 commits ahead and has restructured the API
this CLI uses: `a_sabr::routing` gave way to `pathfinding::top_level`,
`route_stage` is gone, and `Bundle` lost its `source` and `destinations`
fields. `src/main.rs` fails to compile against `main` with 8 errors, so
re-anchoring means porting the CLI.

For a paper artefact that is the normal condition, and it costs the paper
nothing. The anchor is a correctness cross-check that no result depends on: no
CSV and no figure changes whether the pin moves or is never touched again. Its
value is that "identical to A-SABR" names one specific A-SABR, which a reviewer
can go and reproduce.

To re-anchor against a newer A-SABR, update `rev` in `Cargo.toml`, port
`src/main.rs` to the new API, then run `make anchor`. If the 21 anchor tests
still pass, the claim holds for that revision too.

Two caveats:

- **Rust ≥ 1.97 is required**
  ([`rust-toolchain.toml`](rust/asabr-cli/rust-toolchain.toml) pins it). A-SABR
  uses standard-library features stabilised up to 1.97, so an older toolchain
  fails the build with a wall of E0658 "use of unstable library feature"
  errors. That looks like a nightly requirement, but A-SABR targets stable and
  its CI runs on stable. `rustup update` is the fix.
- **A-SABR's plan lexer parses numbers as `i64`.** Plans written for the
  cross-check must therefore have integer-valued attributes.
  `Realization.write_asabr` raises on anything else instead of rounding
  silently. Real-valued plan support would be a natural item for a future
  A-SABR branch.

---

## Repository layout

```
scgraph/
  distributions.py      Dirac / Uniform / TruncNormal / LogNormal          (A2)
  contact.py            StochasticContact, DetContact                     (eq. 1)
  graph.py              StochasticContactGraph, Realization, LatentModel  (A4)
  arrival.py            store-carry-forward recursion, scalar + vectorised (eqs. 2-4)
  analytic.py           convolution CDF / quantile / p_success            (eq. 8, §V-F a)
  routing/
    deterministic.py    CGR/SABR-style earliest-arrival Dijkstra (the baseline)
    quantile.py         candidate_routes, route_stats, pick_route,
                        pick_route_chance                                 (eqs. 5-10)
    replication.py      joint_success, greedy_select                      (eqs. 12-13)
    bernoulli.py        Bernoulli-TEG projection + router                 (eq. 14)
    placement.py        three planning-plane placements + execution driver (§VI)
    asabr.py            subprocess wrapper around the Rust CLI (the anchor)
  verify/montecarlo.py  P_hat with Wilson CIs; full-graph rerouting reference
  io/cpd.py             reads Contact Plan Designer exports               (§IV-A, A1)
  scenarios/            uc1_event.py, uc2_logistics.py, uc3_emergency.py

experiments/            one runner per figure, plus _style.py (shared plot style)
scenarios/              uc1_skeleton.yaml + committed CPD plans (uc1_cpd, scale_*)
results/                the paper's CSVs; reference/ is the verification baseline
figures/                the paper's figures, copied into main.tex's img/
rust/asabr-cli/         Level-0 A-SABR CLI (A-SABR is a pinned git dependency)
scripts/                build_anchor.sh, generate_skeletons.sh, verify_results.py
docs/                   PIPELINE.md (the §IV walkthrough), ARCHIVAL.md
tests/                  66 tests
```

The CPD skeletons under `scenarios/` are committed (about 7 MB in the working
tree, 190 kB packed) so that `run_runtime.py` works from a bare clone.
`scripts/generate_skeletons.sh` regenerates them if you have CPD installed.

---

## Before you change anything

The paper quotes exact numbers, so the experiments have to stay
bit-reproducible. Three choices in the runners exist only to protect that, and
none of them is a tidiness opportunity:

- `run_uc1.py` and `run_uc3.py` score the Bernoulli ablation on a separate RNG
  stream (`SEED + 1`), so adding it left every pre-existing series
  bit-identical.
- `run_runtime.py` takes a fresh stream per plan size (`SEED + 1000*n + i`),
  because the timing loop itself consumes draws.
- Anything that changes the order of RNG draws inside an instance loop moves
  the published results.

After any change, run `make reproduce && make verify`. A difference is a bug in
the change until proven otherwise. If a value genuinely has to move, say so and
update the manuscript alongside it; quietly re-recording the reference defeats
the point of having one.

**The analytic convolution is the slower path here, and that is correct.** At
the $\varepsilon$ values the paper uses it runs about 200× slower than Monte
Carlo (§V-F). MC costs $O(M)$, 0.76 ms at $M{=}2000$, while the convolution
pays a fixed grid cost of 169 ms at 4000 points; they cross near
$M\approx 7\times10^5$. `run_mc_scaling.py` is that measurement. The paper
states this explicitly, having first assumed the opposite, so "optimising" it
would contradict the text.

**`route_stats["quantile"]` is a ranking key over an enumerated candidate set,
not a shortest-path label.** The quantile of a sum is not the sum of quantiles,
so it has no optimal substructure (§V-C). Candidate generation is the
deterministic route search, and the disjoint-suppression sweep in
`candidate_routes` matters more than it looks: without it the safe backbone
never enters the candidate set, and H1 stops reproducing.

---

## How to cite

Cite the paper for the model and results, and the software for the artefact.
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff), which GitHub,
GitLab and Zenodo all read.

```bibtex
@article{fraire2026scgraph,
  author  = {Fraire, Juan A. and Gu{\'e}rin Lassous, Isabelle and
             Begin, Thomas and Lavinal, Emmanuel},
  title   = {Stochastic Contact Graphs: Rethinking the Contact Plan for the
             {6G} Space-Air-Ground Continuum},
  journal = {IEEE Open Journal of the Communications Society},
  year    = {2027},
  note    = {Special Issue on Delay-Tolerant Networking for 6G. To appear.},
  doi     = {10.1109/OJCOMS.2026.XXXXXX}
}

@software{scgraph,
  author  = {Fraire, Juan A. and Gu{\'e}rin Lassous, Isabelle and
             Begin, Thomas and Lavinal, Emmanuel},
  title   = {scgraph: stochastic contact graphs and {SLA}-aware {DTN} routing
             under uncertainty},
  year    = {2026},
  version = {0.1.0},
  url     = {https://gitlab.inria.fr/jfraire/scgraph}
}
```

Volume, pages and DOIs are placeholders until the paper is published. See
[`docs/ARCHIVAL.md`](docs/ARCHIVAL.md) for the release and DOI plan.

## Licence

MIT, see [`LICENSE`](LICENSE). A-SABR and the Contact Plan Designer are both
MIT-licensed, and both are fetched or invoked without being vendored.

## Acknowledgement

Funded by the French government under the France 2030 ANR programme "PEPR
Networks of the Future" (DONUTS project, ANR-24-PEFT-0001), and by the ANR
STEREO project (ANR-22-CE25-0014-01).
