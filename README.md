# scgraph — stochastic contact graphs

Companion library for the paper *"Stochastic Contact Graphs: Rethinking the
Contact Plan for the 6G Space-Air-Ground Continuum"* (IEEE OJ-COMS, DONUTS
project). Equation numbers in docstrings refer to the paper's `main.tex`.

## What it does

- **Model**: stochastic contacts with laws over start / duration / rate,
  existence probability, and an optional shared latent state for correlated
  failures (`scgraph/contact.py`, `scgraph/graph.py`, `scgraph/distributions.py`).
- **Routing**: earliest-arrival CGR/SABR-style router on realizations
  (`routing/deterministic.py`), SLA-aware route selection maximizing
  `P[deliver by deadline]` with quantile cost (`routing/quantile.py`),
  correlation-aware replication (`routing/replication.py`).
- **Verification**: per-route Monte Carlo, full-graph Monte Carlo with Wilson
  CIs (`verify/montecarlo.py`), and analytic CDF propagation for the timing-
  uncertainty regime (`analytic.py`).
- **Anchor**: on point-mass graphs, routes and delivery times are asserted to
  match the [A-SABR](https://github.com/DTN-MTP/A-SABR) framework exactly, via
  a thin Rust CLI (`rust/asabr-cli`, path-dependency on the A-SABR clone
  expected at `../a-sabr`).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
(cd rust/asabr-cli && cargo build --release)   # needs ../../../a-sabr checkout
.venv/bin/pytest -q                             # 40 tests, incl. the anchor
```

Note: A-SABR's contact-plan lexer parses numbers as `i64`, so plans written
for the cross-check must have integer-valued attributes (see
`Realization.write_asabr`).

## Reproducing the paper figures (H1/H2/H3)

```bash
.venv/bin/python experiments/run_uc1.py   # UC1: SLA violation vs timing jitter
.venv/bin/python experiments/run_uc2.py   # UC2: provisioned overhead vs fade
.venv/bin/python experiments/run_uc3.py   # UC3: delivery ratio vs collapse
```

Outputs land in `results/*.csv` and `figures/*.{pdf,png}`; all runs are
seeded (`SEED` at the top of each script). `--quick` runs a reduced sweep.
