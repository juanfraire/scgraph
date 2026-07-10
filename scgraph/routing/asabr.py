"""Subprocess wrapper around the Level-0 A-SABR CLI (the correctness anchor).

The CLI lives in rust/asabr-cli and depends on the A-SABR clone at
../a-sabr (path dependency). Build once with:
    cargo build --release   (in rust/asabr-cli)
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

from scgraph.graph import Realization

_CLI = Path(__file__).resolve().parents[2] / "rust" / "asabr-cli" / "target" / "release" / "asabr-cli"


def available() -> bool:
    return _CLI.exists()


def route_arrival(
    real: Realization, src: int, dst: int, size: float, t0: float
) -> tuple[float, list[int]]:
    """Route one bundle with A-SABR; returns (arrival, node path). inf if unreachable."""
    if not available():
        raise RuntimeError(
            f"asabr-cli not built: run `cargo build --release` in {_CLI.parents[2]}"
        )
    with tempfile.NamedTemporaryFile("w", suffix=".cp", delete=False) as f:
        path = f.name
    try:
        real.write_asabr(path)
        out = subprocess.run(
            [str(_CLI), path, str(src), str(dst), str(size), str(t0)],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(out.stdout.strip())
        if not data.get("found"):
            return math.inf, []
        return float(data["arrival"]), [h["to_node"] for h in data["hops"]]
    finally:
        os.unlink(path)
