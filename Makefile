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
        skeletons clean-figures help reproduce-revision \
        $(addprefix run-,$(EXPERIMENTS)) $(addprefix run-,$(SCHEMATICS)) \
        $(addprefix run-,$(REVISION))

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

# ------------------------------------------------- revision (OJCOMS-07102-2026)
#
# New experiments added for the resubmission. They write new CSVs and figures
# and leave the submitted ones alone; `reproduce-revision` runs them all.
# Wall times are at 3 worker processes (SCGRAPH_WORKERS) on the machine above.

REVISION := uc4 compound uc1_nonfair joint_law rucop flows misspec_timing misspec_capacity \
            dependence estimation overhead sidecar staleness gap generality placement_uc4

run-uc4:                      ## UC4 multi-hop continuum: static vs repair, joint law, compounding
	$(PY) experiments/run_uc4.py
run-compound:                 ## jitter compounding along a k-hop chain, isolated (R1)
	$(PY) experiments/run_compound.py
run-uc1_nonfair:              ## UC1 on the plan WITHOUT fair contact-plan design (R2)
	$(PY) experiments/run_uc1.py --skeleton scenarios/uc1_raw --tag nonfair
run-joint_law:                ## coupled-contact figure (R2: Fig. 1 coupled example)
	$(PY) experiments/plot_joint_law.py
run-rucop:                    ## BRUF-1 / RUCoP-N against SLA-aware routing (R2)
	$(PY) experiments/run_rucop.py
run-flows:                    ## concurrent bundles, residual volume, finite storage (R1, R2)
	$(PY) experiments/run_flows.py
run-misspec_timing:           ## heavy-tailed true timing (R2)
	$(PY) experiments/run_misspec_timing.py
run-misspec_capacity:         ## fade variance and inter-pass correlation (R2, R3)
	$(PY) experiments/run_misspec_capacity.py
run-dependence:               ## strength and structure of latent dependence (R3)
	$(PY) experiments/run_dependence.py
run-estimation:               ## learning the laws from n observations (R2)
	$(PY) experiments/run_estimation.py
run-overhead:                 ## compute re-baseline, forwarding cost, sidecar size (R1); time on an idle machine
	SCGRAPH_WORKERS=1 $(PY) experiments/run_overhead.py
run-sidecar:                  ## control-plane bytes and uplink time of the stochastic sidecar (R1)
	$(PY) experiments/run_sidecar.py
run-staleness:                ## planner information age (R1)
	$(PY) experiments/run_staleness.py
run-gap:                      ## candidate-set optimality gap against an exhaustive oracle (R1, R2, R3)
	$(PY) experiments/run_gap.py
run-generality:               ## one-at-a-time sensitivity on UC4 (R3)
	$(PY) experiments/run_generality.py
run-placement_uc4:            ## placement probe on the deep topology (R1)
	$(PY) experiments/run_placement_uc4.py

reproduce-revision: $(addprefix run-,$(REVISION))
	@echo "revision experiments done."

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
