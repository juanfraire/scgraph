#!/usr/bin/env bash
#
# Build the Level-0 A-SABR CLI, which is the correctness anchor: on point-mass
# graphs scgraph's router must return the same routes and delivery times as
# A-SABR (paper Sec. VII-A, the equality half of the subsumption claim).
#
# The anchor is OPTIONAL. Nothing in results/ or figures/ depends on it, so a
# reader without a Rust toolchain can reproduce every number in the paper
# without ever running this script. What the anchor buys is the check that our
# deterministic corner really is A-SABR's, and running it is the only way to
# see that check actually pass.
#
# A-SABR is a normal pinned git dependency in rust/asabr-cli/Cargo.toml, so
# this script is a thin wrapper over `cargo build --release`. It exists to give
# a clear message for the one failure mode that is not obvious:
#
#   A-SABR uses standard-library features stabilised up to Rust 1.97. On an
#   older toolchain the build fails with a list of E0658 "use of unstable
#   library feature" errors, which reads like A-SABR is nightly-only. It is
#   not: it targets stable, and its CI runs on stable. The toolchain is simply
#   out of date, and `rustup update` fixes it.
#
# Usage:  scripts/build_anchor.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$REPO/rust/asabr-cli"
MIN_RUST="1.97"

command -v cargo >/dev/null || {
  echo "error: cargo not found. Install Rust from https://rustup.rs and re-run." >&2
  echo "       The anchor is optional; 'make reproduce' works without it." >&2
  exit 1
}

# rust-toolchain.toml pins the version, so rustc reports the pinned one if it
# is installed and rustup fetches it if not. Warn before the build rather than
# after, because the E0658 wall is genuinely misleading.
have="$(rustc --version 2>/dev/null | cut -d' ' -f2 || echo 0)"
lowest="$(printf '%s\n%s\n' "$MIN_RUST" "$have" | sort -V | head -1)"
if [ "$lowest" != "$MIN_RUST" ]; then
  echo "warning: rustc $have is older than the $MIN_RUST A-SABR needs." >&2
  echo "         If the build fails with E0658 'unstable library feature'," >&2
  echo "         run 'rustup update' -- that is the fix, not a patch." >&2
fi

echo "==> building asabr-cli (A-SABR fetched at the pin in Cargo.toml)"
( cd "$CLI" && cargo build --release )

BIN="$CLI/target/release/asabr-cli"
[ -x "$BIN" ] || { echo "error: build reported success but $BIN is missing" >&2; exit 1; }
echo "==> ok: $BIN"
echo "    run the anchor with:  SCGRAPH_REQUIRE_ANCHOR=1 pytest tests/test_anchor.py -q"
