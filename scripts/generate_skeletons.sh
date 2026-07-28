#!/usr/bin/env bash
#
# Regenerate the CPD orbital skeletons under scenarios/ (Stage 1 of the
# populate-and-learn pipeline, paper Sec. IV-A).
#
# YOU DO NOT NEED THIS TO REPRODUCE THE PAPER. Every skeleton it produces is
# committed, so `make reproduce` runs from a bare clone with no CPD install.
# This script is here so the generated data is not a black box: it records
# exactly which scenario and which constellation sizes produced the plans, and
# lets a reviewer regenerate them or change the scenario.
#
# Requires the Contact Plan Designer on PATH:
#   https://gitlab.inria.fr/jfraire/contact-plan-designer
#
# Usage:  scripts/generate_skeletons.sh [uc1|scale|all]     (default: all)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCEN="$REPO/scenarios"
WHAT="${1:-all}"

command -v cpd >/dev/null || {
  echo "error: 'cpd' not found on PATH." >&2
  echo "       Install the Contact Plan Designer, or just use the committed" >&2
  echo "       plans under scenarios/ (nothing else needs this script)." >&2
  exit 1
}

# The UC1 skeleton: 12 satellites, 3 planes, 600 km, 53 deg, event site and
# gateway 390 km apart, 10 deg mask, 6 h horizon -> 298 contacts. This is the
# backbone behind UC1, UC2, the ablation, the eps-sweep and the placement probe.
gen_uc1() {
  echo "==> uc1_cpd  (12 sats, the scenario in the paper)"
  cpd run "$SCEN/uc1_skeleton.yaml" -o "$SCEN/uc1_cpd"
}

# The runtime study (Sec. VII-H) needs the SAME scenario at several
# constellation sizes, so that plan size is the only thing varying: 6/12/24/48
# satellites give the 98/298/1110/14640 contacts that results/runtime.csv
# reports. Only the satellite count in the Walker code changes; every other
# scenario parameter is held fixed, so the resulting curve says something about
# plan size and nothing about geometry.
#
# The plane count is held at 3 throughout, matching uc1_skeleton.yaml (so
# scale_12 reproduces the UC1 skeleton exactly, which is verifiable: its ION
# export is byte-identical to uc1_cpd's). CPD's export records the scenario
# name but not the Walker code, so 3 planes at n=6 is a reconstruction from
# that identity and from the committed satellite ids, not a value read back
# out of the plan.
gen_scale() {
  for n in 6 12 24 48; do
    code="D:600:53:${n}/3/1"
    out="$SCEN/scale_${n}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN
    echo "==> scale_${n}  (${code})"
    sed -e "s|code: \".*\"|code: \"${code}\"|" \
        -e "s|^name: .*|name: scgraph-scale-${n}|" \
        "$SCEN/uc1_skeleton.yaml" > "$tmp/scenario.yaml"
    cpd run "$tmp/scenario.yaml" -o "$out"
  done
}

case "$WHAT" in
  uc1)   gen_uc1 ;;
  scale) gen_scale ;;
  all)   gen_uc1; gen_scale ;;
  *)     echo "usage: $0 [uc1|scale|all]" >&2; exit 1 ;;
esac

cat <<'EOF'

Regenerated. Note that this REPLACES the committed plans, so every experiment
that consumes them will produce different numbers unless CPD reproduces the
previous output byte for byte. Re-run `make reproduce && make verify` and treat
any drift as a change to be justified, not as a new baseline.
EOF
