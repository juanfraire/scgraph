"""Check that regenerated results still match the numbers the paper quotes.

Every experiment is seeded, so re-running one must reproduce its CSV exactly.
This script is the guard on that: it compares the CSVs in results/ against a
reference copy and exits non-zero on any drift.

Why this exists at all, given `git diff results/`: the paper's text, abstract and
conclusion cite specific values, so a refactor that perturbs the ORDER in which
random numbers are drawn will silently move a figure and invalidate a sentence
that is already written. Bit-identical output is the only cheap way to notice.
Two RNG choices in the runners exist for exactly this reason:

  * run_uc1.py / run_uc3.py score the Bernoulli ablation on a separate stream
    (SEED + 1), so adding it left every pre-existing series untouched.
  * run_runtime.py takes a fresh stream per plan size (SEED + 1000*n + i),
    because the timing loop itself consumes draws.

Usage:
    python scripts/verify_results.py                 # results/ vs results/reference/
    python scripts/verify_results.py --update        # adopt current results as reference
    python scripts/verify_results.py --tol 1e-9      # allow a numeric tolerance

Default tolerance is 0: on one machine, with pinned dependencies, the runs are
bit-identical. Cross-platform (different BLAS, different libm) the last digits
of a float can move without any modelling change, and --tol covers that.
A run that needs --tol on the SAME machine is a bug, not a rounding artefact.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REFERENCE = RESULTS / "reference"

# The CSVs the paper draws on, and the experiment that writes each.
EXPECTED = {
    "uc1.csv": "run_uc1.py",
    "uc2.csv": "run_uc2.py",
    "uc3.csv": "run_uc3.py",
    "ablation_slots.csv": "run_ablation.py",
    "eps.csv": "run_eps.py",
    "placement_uc1.csv": "run_placement.py",
    "placement_uc3.csv": "run_placement.py",
    "runtime.csv": "run_runtime.py",
    "mc_scaling.csv": "run_mc_scaling.py",
}

# Wall-clock timings are measurements of the machine, not of the model, so they
# differ run to run by design. The runtime study's *shape* is the claim (scoring
# flat, generation growing), not the millisecond values, so these columns are
# compared for presence only. Everything else must match exactly.
TIMING_COLUMNS = {
    "t_det_ms", "t_cand_ms", "t_mc_per_route_ms",
    "t_conv_per_route_ms", "t_total_ms",
    "t_mc_ms", "t_conv_ms",          # run_mc_scaling.py
}


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def cells_differ(a: str, b: str, tol: float) -> bool:
    """Compare two CSV cells, numerically when both parse as numbers."""
    if a == b:
        return False
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return True
    if math.isnan(fa) and math.isnan(fb):   # NaN != NaN, but two NaNs agree here
        return False
    if math.isinf(fa) or math.isinf(fb):
        return fa != fb
    if tol == 0.0:
        # Exact, but on the parsed values: "1.0" and "1.00" are the same number,
        # and a repr change in a future Python should not read as result drift.
        return fa != fb
    return abs(fa - fb) > tol * max(1.0, abs(fa), abs(fb))


def compare(name: str, tol: float) -> list[str]:
    """Differences between results/<name> and results/reference/<name>."""
    cur, ref = RESULTS / name, REFERENCE / name
    if not ref.exists():
        return [f"{name}: no reference (run --update to record one)"]
    if not cur.exists():
        return [f"{name}: MISSING -- run experiments/{EXPECTED[name]}"]

    cur_cols, cur_rows = read_csv(cur)
    ref_cols, ref_rows = read_csv(ref)
    problems: list[str] = []

    if cur_cols != ref_cols:
        return [f"{name}: columns changed\n    reference: {ref_cols}\n    current:   {cur_cols}"]
    if len(cur_rows) != len(ref_rows):
        return [f"{name}: {len(ref_rows)} rows expected, {len(cur_rows)} found"]

    timing_only = 0
    for i, (c, r) in enumerate(zip(cur_rows, ref_rows)):
        for col in ref_cols:
            if cells_differ(c[col], r[col], tol):
                if col in TIMING_COLUMNS:
                    timing_only += 1
                    continue
                problems.append(
                    f"{name}: row {i + 1}, column '{col}': "
                    f"expected {r[col]}, got {c[col]}"
                )
    if timing_only:
        print(f"  note: {name}: {timing_only} wall-clock cell(s) differ (expected)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true",
                    help="copy current results/ into results/reference/")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="relative tolerance (default 0 = bit-identical)")
    args = ap.parse_args()

    if args.update:
        REFERENCE.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in EXPECTED:
            src = RESULTS / name
            if src.exists():
                shutil.copy2(src, REFERENCE / name)
                n += 1
            else:
                print(f"  skipped {name} (not generated yet)")
        print(f"recorded {n}/{len(EXPECTED)} reference CSVs in {REFERENCE}")
        return 0

    problems: list[str] = []
    for name in EXPECTED:
        problems.extend(compare(name, args.tol))

    if problems:
        print("\nRESULTS DRIFTED from the values the paper quotes:\n")
        for p in problems[:40]:
            print(f"  {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        print(
            "\nTreat this as a bug in the change that caused it, not as a new "
            "baseline.\nThe paper's abstract, results and conclusion cite these "
            "numbers verbatim.\nIf a value genuinely must move, say so explicitly "
            "and update the manuscript\ntogether with results/reference/."
        )
        return 1

    print(f"all {len(EXPECTED)} result files match the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
