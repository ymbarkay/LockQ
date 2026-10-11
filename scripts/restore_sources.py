#!/usr/bin/env python3
"""Restore the simulation source used by the LockQ experiment scripts."""
from pathlib import Path
import gzip

repo = Path(__file__).resolve().parents[1]
archive = repo / "experiments" / "simulation.py.gz"
target = repo / "experiments" / "simulation.py"
if not archive.exists():
    raise SystemExit(f"Missing bundled simulation source: {archive}")
target.write_bytes(gzip.decompress(archive.read_bytes()))
print(f"Restored {target.relative_to(repo)}")
