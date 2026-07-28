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
# see that check pass rather than take it on trust.
#
# Why this is a script and not plain `cargo build`:
#
#   A-SABR at the pinned revision does not compile on current stable Rust. Its
#   parser uses `MaybeUninit::uninit().into()` to build a `[MaybeUninit<T>; N]`,
#   which stable rustc has never accepted; rustc 1.90 rejects it with E0277.
#   The fix is three call sites rewritten to the stable uninit-array idiom,
#   carried here as patches/0001-stable-rust-uninit-array.patch and applied to a
#   clean upstream checkout. It touches parsing.rs only, so no routing
#   behaviour changes, and the anchor test is what proves the routes still agree.
#
#   The patch belongs upstream. Until it lands, this script is what makes the
#   anchor reproducible off the original author's machine (docs/ARCHIVAL.md).
#
# Usage:  scripts/build_anchor.sh
# Env:    ASABR_SRC=/path/to/a-sabr   use an existing checkout (linked, not cloned;
#                                     you are then responsible for its revision
#                                     and for having the patch applied)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$REPO/rust/asabr-cli"
PATCH="$CLI/patches/0001-stable-rust-uninit-array.patch"
VENDOR="$REPO/rust/vendor/a-sabr"

# Pinned so the anchor is checked against one known A-SABR, not "whatever main
# is today". Bump this and the patch together.
ASABR_URL="https://github.com/DTN-MTP/A-SABR.git"
ASABR_REV="1277e360f2d61a593c336c6f6eb2e583758d7580"

command -v cargo >/dev/null || {
  echo "error: cargo not found. Install Rust from https://rustup.rs and re-run." >&2
  echo "       The anchor is optional; 'make reproduce' works without it." >&2
  exit 1
}

mkdir -p "$REPO/rust/vendor"

# rust/asabr-cli/Cargo.toml has a fixed relative path dependency on
# ../vendor/a-sabr/asabr. Both modes below make that path resolve, so the
# manifest never has to be edited and the working tree stays clean.
if [ -n "${ASABR_SRC:-}" ]; then
  echo "==> linking rust/vendor/a-sabr -> $ASABR_SRC"
  echo "    (ASABR_SRC set: revision and patch state are yours to manage)"
  rm -rf "$VENDOR"
  ln -s "$ASABR_SRC" "$VENDOR"
else
  if [ -L "$VENDOR" ]; then          # drop a stale ASABR_SRC symlink
    rm -f "$VENDOR"
  fi
  if [ ! -d "$VENDOR/.git" ]; then
    echo "==> cloning A-SABR into rust/vendor/a-sabr"
    git clone --quiet "$ASABR_URL" "$VENDOR"
  fi
  echo "==> checking out ${ASABR_REV:0:12}"
  git -C "$VENDOR" fetch --quiet origin "$ASABR_REV" 2>/dev/null || true
  # --force resets the tree, so the patch below always applies to pristine
  # upstream and re-running this script is idempotent.
  git -C "$VENDOR" checkout --quiet --force "$ASABR_REV"
  echo "==> applying $(basename "$PATCH")"
  git -C "$VENDOR" apply "$PATCH"
fi

echo "==> building asabr-cli"
( cd "$CLI" && cargo build --release )

BIN="$CLI/target/release/asabr-cli"
[ -x "$BIN" ] || { echo "error: build reported success but $BIN is missing" >&2; exit 1; }
echo "==> ok: $BIN"
echo "    run the anchor with:  SCGRAPH_REQUIRE_ANCHOR=1 pytest tests/test_anchor.py -q"
