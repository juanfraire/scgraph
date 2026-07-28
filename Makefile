# scgraph: reproduction entry points for the OJ-COMS paper.
#
#   make install     create .venv and install the package (loose bounds)
#   make install-repro   same, but with the exact pinned versions used for the paper
#   make test        run the test suite (anchor skipped if not built)
#   make anchor      build the A-SABR CLI, then run the anchor for real
#   make reproduce   regenerate every CSV and figure          (~11 min, M3 Pro)
#   make verify      diff regenerated CSVs against the reference; non-zero on drift
#   make all         reproduce + verify
#
# Wall times measured on an Apple M3 Pro / macOS 26.5 / Python 3.14.4, single
# process. Only `placement` and `runtime` are slow enough to notice.

PY      := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest

# In dependency-free order; `reproduce` runs them all.
EXPERIMENTS := uc1 uc2 uc3 ablation eps placement runtime mc_scaling
# Figure-only scripts: they draw the scenario rather than measure it, so they
# write to figures/ but produce no CSV for `verify` to diff.
SCHEMATICS  := scenario

.PHONY: all install install-repro test anchor reproduce verify update-reference \
        skeletons clean-figures help \
        $(addprefix run-,$(EXPERIMENTS)) $(addprefix run-,$(SCHEMATICS))

help:
	@sed -n '3,12p' Makefile | sed 's/^# \{0,1\}//'
	@echo
	@echo "single experiments (time, on the machine above):"
	@grep -E '^run-[a-z0-9_]+:.*##' Makefile \
	  | awk -F'##' '{split($$1,t,":"); printf "  make %-14s %s\n", t[1], $$2}'

# ---------------------------------------------------------------- environment

.venv:
	python3 -m venv .venv

install: .venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[dev]"
	@echo "installed. next: make test"

# The exact versions behind the numbers in the paper. Use this to reproduce,
# and `install` to develop.
install-repro: .venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements-repro.txt
	$(PIP) install --quiet -e . --no-deps
	@echo "installed pinned versions. next: make reproduce"

# ---------------------------------------------------------------------- tests

test:
	$(PYTEST) -q

# The anchor is optional for reproducing the paper but required for the
# subsumption claim's equality half, so this target makes it hard-fail.
anchor:
	scripts/build_anchor.sh
	SCGRAPH_REQUIRE_ANCHOR=1 $(PYTEST) tests/test_anchor.py -q

# ---------------------------------------------------------------- experiments
#
# Each runner writes results/<name>.csv and figures/<name>.{pdf,png} and is
# individually re-runnable; the per-target timings are the measured medians.

run-uc1:                      ##  53 s  H1: SLA violation vs timing jitter
	$(PY) experiments/run_uc1.py
run-uc2:                      ##   3 s  H2: provisioned overhead vs fade
	$(PY) experiments/run_uc2.py
run-uc3:                      ##  13 s  H3: delivery ratio vs collapse severity
	$(PY) experiments/run_uc3.py
run-ablation:                 ##  97 s  subsumption: Bernoulli corner vs slot width
	$(PY) experiments/run_ablation.py
run-eps:                      ##  22 s  the epsilon knob
	$(PY) experiments/run_eps.py
run-placement:                ## 346 s  placement probe (the slow one)
	$(PY) experiments/run_placement.py
run-runtime:                  ## 135 s  route-selection runtime vs plan size
	$(PY) experiments/run_runtime.py
run-mc_scaling:               ##   5 s  MC vs convolution cost against M (Sec. V-F)
	$(PY) experiments/run_mc_scaling.py
run-scenario:                 ##   1 s  scenario figure: skeleton geometry + passes
	$(PY) experiments/plot_scenario.py

reproduce: $(addprefix run-,$(EXPERIMENTS)) $(addprefix run-,$(SCHEMATICS))
	@echo
	@echo "all experiments done. figures/ now holds the paper's img/ contents."
	@echo "run 'make verify' to confirm nothing moved."

verify:
	$(PY) scripts/verify_results.py

# Only after deliberately accepting that a number changed, and updating the
# manuscript to match.
update-reference:
	$(PY) scripts/verify_results.py --update

all: reproduce verify

# ------------------------------------------------------------------ scenarios

# The CPD skeletons are committed, so this is only needed to change the
# scenario. Requires the Contact Plan Designer on PATH.
skeletons:
	scripts/generate_skeletons.sh

clean-figures:
	rm -f figures/*.pdf figures/*.png
