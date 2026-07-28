# scgraph — stochastic contact graphs

Companion library and reproduction artefact for

> J. A. Fraire, I. Guérin Lassous, T. Begin, E. Lavinal,
> **"Stochastic Contact Graphs: Rethinking the Contact Plan for the 6G
> Space-Air-Ground Continuum"**, IEEE Open Journal of the Communications
> Society, Special Issue on Delay-Tolerant Networking for 6G (under review).

The paper generalises the DTN **contact plan** so that each contact carries a
joint law over its start time, end time and capacity, and turns routing into a
chance-constrained *deliver-by-deadline* problem. Deterministic CGR/SABR
(point masses) and the discrete on/off "uncertain DTN" model (a Bernoulli
marginal) are corners of that object; this repository implements the object,
both corners, and every measurement the paper reports.

**Everything in the paper regenerates with one command**, in about 11 minutes:

```bash
make install-repro     # pinned dependency versions
make reproduce         # 9 CSVs, and all 7 of the paper's result figures
make verify            # bit-for-bit diff against the recorded reference
```

---

## Table of contents

- [Quick start](#quick-start)
- [Paper → code map](#paper--code-map) ← **start here if you are holding the PDF**
- [Reproducing the results](#reproducing-the-results)
- [The correctness anchor](#the-correctness-anchor-a-sabr)
- [Repository layout](#repository-layout)
- [Things worth knowing before you change anything](#things-worth-knowing-before-you-change-anything)
- [How to cite](#how-to-cite)

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
target. No Rust toolchain and no Contact Plan Designer install are needed to
reproduce the paper: the orbital skeletons are committed.

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
| (8) | $Q_{1-\varepsilon}(\pi,m)$, the quantile delivery time | [`quantile.py:82`](scgraph/routing/quantile.py#L82) (MC); [`scgraph/analytic.py:76`](scgraph/analytic.py#L76) `route_quantile` (convolution) | [`tests/test_quantile.py`](tests/test_quantile.py) — the two agree within CI |
| (9) | $\Pr\ge 1-\varepsilon \iff Q_{1-\varepsilon}\le D_m$ | the equivalence `pick_route_chance` relies on: it filters on $P_{\mathrm{succ}}$ and minimises latency | Fig. 5 (left) is this claim measured |
| (10) | $\mathrm{cost}_\varepsilon(\pi,m)=Q_{1-\varepsilon}$, the SABR drop-in key | [`quantile.py`](scgraph/routing/quantile.py) `route_stats["quantile"]` | [`tests/test_quantile.py`](tests/test_quantile.py) |
| (11) | Capacity chance constraint $\Pr[\sum_{c}V^c\ge\omega]\ge 1-\varepsilon$ | [`scgraph/scenarios/uc2_logistics.py`](scgraph/scenarios/uc2_logistics.py) `provisioning_cpd` | UC2 (Fig. 6) |
| (12) | Independent replication $1-\prod(1-P_{\mathrm{succ}})$ | [`scgraph/routing/replication.py:34`](scgraph/routing/replication.py#L34) `independent_success` | [`tests/test_replication.py`](tests/test_replication.py) |
| (13) | Correlated replication $\mathbb{E}_z[\,\cdot\,]$ under a shared latent state | [`replication.py:18`](scgraph/routing/replication.py#L18) `joint_success`, via [`arrival.py:64`](scgraph/arrival.py#L64) `joint_route_set_deliveries`; latent model in [`graph.py:18`](scgraph/graph.py#L18) | UC3 (Fig. 7) |
| (14) | Bernoulli projection $q_c[k]$ | [`scgraph/routing/bernoulli.py:54`](scgraph/routing/bernoulli.py#L54) `slot_usability` | ablation (Fig. 3), [`tests/test_bernoulli.py`](tests/test_bernoulli.py) |

### Figures

| Fig. | Paper | Produced by | Data | Image |
|---|---|---|---|---|
| 1 | The stochastic contact and its two corners | TikZ in `main.tex` (schematic, no code) | — | — |
| 2 | Populating the graph (pipeline) | TikZ in `main.tex` (schematic) | — | see [`docs/PIPELINE.md`](docs/PIPELINE.md) |
| 3 | Bernoulli corner vs slot width $\Delta$ | [`experiments/run_ablation.py`](experiments/run_ablation.py) | `results/ablation_slots.csv` | `figures/subsumption.pdf` |
| 4 | UC1 / H1: violation + latency vs jitter $\sigma$ | [`experiments/run_uc1.py`](experiments/run_uc1.py) | `results/uc1.csv` | `figures/uc1_h1.pdf` |
| 5 | The $\varepsilon$ knob | [`experiments/run_eps.py`](experiments/run_eps.py) | `results/eps.csv` | `figures/eps.pdf` |
| 6 | UC2 / H2: provisioned overhead vs fade | [`experiments/run_uc2.py`](experiments/run_uc2.py) | `results/uc2.csv` | `figures/uc2_h2.pdf` |
| 7 | UC3 / H3: delivery ratio vs collapse $\rho$ | [`experiments/run_uc3.py`](experiments/run_uc3.py) | `results/uc3.csv` | `figures/uc3_h3.pdf` |
| 8 | Placement probe | [`experiments/run_placement.py`](experiments/run_placement.py) | `results/placement_uc1.csv`, `results/placement_uc3.csv` | `figures/placement.pdf` |
| 9 | Route-selection runtime vs plan size | [`experiments/run_runtime.py`](experiments/run_runtime.py) | `results/runtime.csv` | `figures/runtime.pdf` |
| — | *(no figure)* §V-F's MC-vs-convolution crossover | [`experiments/run_mc_scaling.py`](experiments/run_mc_scaling.py) | `results/mc_scaling.csv` | `figures/mc_scaling.pdf` |

Figures land in `figures/` and are copied into the paper's `img/` by hand.

### Hypotheses

| | Claim | Scenario | Experiment | Headline in the paper |
|---|---|---|---|---|
| **H1** | Distributional $T_s,T_e$ + max-$\Pr[\text{deliver by }D]$ routing lowers the SLA-violation rate vs a deterministic plan, and the gap widens with jitter | [`scenarios/uc1_event.py`](scgraph/scenarios/uc1_event.py) `make_instance_cpd` | `run_uc1.py` | at $\sigma{=}60$ s: det **0.2856** → SLA-aware **0.0**; latency **69 → 549** s; Bernoulli corner at $\Delta{=}30$ s **0.0277** |
| **H2** | Chance-constrained capacity sizing meets the SLA at lower provisioned redundancy than worst-case | [`scenarios/uc2_logistics.py`](scgraph/scenarios/uc2_logistics.py) `provisioning_cpd` | `run_uc2.py` | mean-case SLA **0.63–1.00** (violates); worst-case overhead **1.20 → 5.02**; chance-constrained **1.20 → 1.99** ($2.5\times$ saving) |
| **H3** | Replication budgeted by existence *and correlation* beats a stale plan and correlation-blind flooding at equal overhead | [`scenarios/uc3_emergency.py`](scgraph/scenarios/uc3_emergency.py) | `run_uc3.py` | at $\rho{=}0.5$: stale **0.41**, naive **0.49**, Bernoulli **0.59**, SCG **0.65**, epidemic **0.76** |

Supporting studies: the **subsumption ablation** (§VII-B) traces the path
between the two corners — at $\sigma{=}60$ s, ~0 at fine grids rising to
**0.137** at $\Delta{=}90$ s against deterministic **0.275**. The
**$\varepsilon$-sweep** (§VII-D) shows achieved violation at or below budget at
every $\varepsilon$, with latency **520 → 82** s. The **placement probe**
(§VII-F) gives UC3 at $\rho{=}0.4$: **0.49 / 0.67 / 0.71** (a factor **1.45**),
with the UC1 ordering inverting once the deadline margin drops to 0 or below.
The **runtime study** (§VII-H) spans **98 → 14 640** contacts with per-route
scoring flat at **≈0.75 ms** and generation growing **31 ms → 1.39 s**.

### Model assumptions

| | Assumption | Where it lives in the code |
|---|---|---|
| A1 | Deterministic orbital backbone (point-mass timing) | [`scgraph/io/cpd.py`](scgraph/io/cpd.py) loads CPD exports as all-`Dirac` contacts |
| A2 | Parametric marginals: TruncNormal timing, LogNormal volume | [`scgraph/distributions.py`](scgraph/distributions.py); laws placed on start + **duration** so A5 holds by construction |
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
`--quick` for a reduced sweep, which is useful for smoke-testing a change but
does **not** produce the paper's numbers.

Measured wall time, single process, **Apple M3 Pro / macOS 26.5.2 / Python
3.14.4 / numpy 2.5.1**:

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
| **total** | **≈11 min** | |

`make verify` compares against `results/reference/`, which holds the CSVs the
manuscript was written from. On the recorded environment the match is
**bit-identical**; that is the check, and drift is a bug until proven
otherwise. On a different platform use
`python scripts/verify_results.py --tol 1e-9`.

Two runners measure the machine rather than the model, so their wall-clock
columns are exempt from the comparison and will differ on every run:
`run_runtime.py` and `run_mc_scaling.py`. The claims they support are shapes
(scoring flat in plan size; MC linear in $M$ against a flat convolution), not
millisecond values. Both take `--replot` to redraw from the committed CSV
without re-measuring, which is how `figures/runtime.pdf` stays identical to the
copy in the paper.

---

## The correctness anchor (A-SABR)

The paper's subsumption claim has two halves. The Bernoulli half is lossy and
is *measured* (Fig. 3). The deterministic half is an **equality**, and it is
checked in code: on point-mass graphs `scgraph`'s router must return the same
routes and delivery times as [A-SABR](https://github.com/DTN-MTP/A-SABR),
compared over 20 randomised plans.

```bash
make anchor    # build the Rust CLI, then run the cross-check for real
```

This is **optional**. Nothing in `results/` or `figures/` depends on it, so a
clone without a Rust toolchain reproduces the whole paper and the anchor tests
simply skip, with a message saying how to enable them. Setting
`SCGRAPH_REQUIRE_ANCHOR=1` turns the skip into a failure; CI does this, so the
anchor cannot be silently lost.

### Which A-SABR, exactly

A-SABR is a normal pinned git dependency
([`rust/asabr-cli/Cargo.toml`](rust/asabr-cli/Cargo.toml)); nothing is vendored
or patched. The anchor was checked against exactly this revision:

| | |
|---|---|
| Repository | `https://github.com/DTN-MTP/A-SABR.git` |
| Revision | `1277e360f2d61a593c336c6f6eb2e583758d7580` |
| Dated | 2026-06-30 (`ops(ci): add manual workflow`) |
| Recorded in | `Cargo.toml` (`rev =`) **and** the committed `Cargo.lock` |

**Upstream has moved on, and the pin is deliberate.** As of 2026-07-28 A-SABR's
`main` is 73 commits ahead and has restructured the API the CLI uses:
`a_sabr::routing` was removed in favour of `pathfinding::top_level`,
`route_stage` is gone, and `Bundle` no longer has `source` / `destinations`
fields. `src/main.rs` does not compile against `main` (8 errors), so **bumping
the pin is a port, not a version bump.**

That is the normal condition for a paper artefact, and it costs the paper
nothing: the anchor is a *correctness cross-check*, not a dependency of any
result. No CSV or figure changes if the pin moves, or if it is never touched
again. What the pin buys is that "identical to A-SABR" names one specific
A-SABR, so a reviewer can reproduce the comparison rather than chase a moving
target.

To re-anchor against a newer A-SABR: update `rev` in `Cargo.toml`, port
`src/main.rs` to the new API, then `make anchor`. If the 21 anchor tests still
pass, the claim holds against that revision too.

Two caveats, both real and both worth knowing:

- **It needs Rust ≥ 1.97** ([`rust-toolchain.toml`](rust/asabr-cli/rust-toolchain.toml)
  pins it). A-SABR uses standard-library features stabilised up to 1.97, so on
  an older toolchain the build fails with a wall of E0658 "use of unstable
  library feature" errors. That reads like A-SABR is nightly-only; it is not,
  and its CI runs on stable. `rustup update` is the fix.
- **A-SABR's plan lexer parses numbers as `i64`.** Plans written for the
  cross-check must therefore have integer-valued attributes;
  `Realization.write_asabr` raises rather than rounding silently. Real-valued
  plan support is a natural item for a future A-SABR branch.

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

The CPD skeletons under `scenarios/` are committed (~7 MB working tree, ~190 kB
packed) so that `run_runtime.py` works from a bare clone.
`scripts/generate_skeletons.sh` regenerates them if you have CPD installed.

---

## Things worth knowing before you change anything

**The paper quotes exact numbers, so the experiments must stay bit-reproducible.**
Three deliberate choices exist purely to protect that, and are not tidiness
opportunities:

- `run_uc1.py` and `run_uc3.py` run the Bernoulli ablation on a **separate RNG
  stream** (`SEED + 1`), so that adding it left every pre-existing series
  bit-identical.
- `run_runtime.py` takes a **fresh stream per plan size** (`SEED + 1000*n + i`),
  because the timing loop itself consumes draws.
- Anything that changes the **order** of RNG draws inside an instance loop
  changes the published results.

After any change: `make reproduce && make verify`. A difference is a bug in the
change until proven otherwise. If a number genuinely must move, say so and
update the manuscript deliberately — do not quietly re-record the reference.

**The analytic convolution is slower than Monte Carlo here, and that is
correct.** At the $\varepsilon$ values the paper uses it is ~200× slower
(§V-F): MC is $O(M)$ (0.76 ms at $M{=}2000$) while the convolution pays a fixed
grid cost (169 ms at 4000 points); they cross near $M\approx 7\times10^5$.
`run_mc_scaling.py` is that measurement. The paper states this explicitly,
having first assumed the opposite. Do not "optimise" this into a contradiction
with the text.

**`route_stats["quantile"]` is a ranking key, not a shortest-path label.** The
quantile of a sum is not the sum of quantiles, so it has no optimal
substructure; it ranks an *enumerated* candidate set (§V-C). Candidate
generation is the deterministic route search, and the disjoint-suppression
sweep in `candidate_routes` is load-bearing: without it the safe backbone never
enters the candidate set and H1 does not reproduce.

---

## How to cite

Cite the paper for the model and results, and the software for the artefact.
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (GitHub, GitLab
and Zenodo all read it).

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

Volume, pages and DOIs are placeholders until the paper is published; see
[`docs/ARCHIVAL.md`](docs/ARCHIVAL.md) for the release and DOI plan.

## Licence

MIT, see [`LICENSE`](LICENSE). A-SABR and the Contact Plan Designer are both
MIT-licensed and are fetched or invoked, not vendored.

## Acknowledgement

Funded by the French government under the France 2030 ANR programme "PEPR
Networks of the Future" (DONUTS project, ANR-24-PEFT-0001), and by the ANR
STEREO project (ANR-22-CE25-0014-01).
