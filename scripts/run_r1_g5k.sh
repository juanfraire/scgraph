#!/bin/bash
# Second-revision experiments (R1) on one Grid'5000 node.
# Usage (from the repo root on the frontend):
#   oarsub -l host=1,walltime=2:00:00 ./scripts/run_r1_g5k.sh              # E1, E2 (+ --uc1), E4
#   oarsub -l host=1,walltime=1:00:00 "./scripts/run_r1_g5k.sh run_r1_bernoulli 'run_r1_bernoulli --uc1'"
# Each argument is an experiment script name, optionally followed by its flags.
set -euo pipefail
cd "$(dirname "$0")/.."
CORES=$(nproc)
export SCGRAPH_WORKERS=$CORES
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs results
if [ $# -eq 0 ]; then
  set -- run_r1_admission run_r1_bernoulli "run_r1_bernoulli --uc1" run_r1_uc3grid
fi
echo "host $(hostname), $CORES workers, $(date -Is)"
for e in "$@"; do
  name=${e%% *}
  log=logs/$(echo "$e" | tr -s ' -' '_' | sed 's/_$//').log
  echo "== $e $(date -Is) -> $log"
  # shellcheck disable=SC2086
  .venv/bin/python experiments/$name.py ${e#"$name"} > "$log" 2>&1
  tail -1 "$log"
done
echo "all done $(date -Is)"
