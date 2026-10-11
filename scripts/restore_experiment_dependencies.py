#!/usr/bin/env python3
"""Restore the archived v2.6 simulation dependency from its gzip source blob."""
import gzip
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
source = HERE / "experiments" / "v26" / "lockq_base.py.gz"
target = source.with_suffix("")

if not source.exists():
    raise SystemExit(f"Missing source archive: {source}")
data = gzip.decompress(source.read_bytes())
if not data.startswith(b"from __future__ import annotations"):
    raise SystemExit("Unexpected archive contents")
target.write_bytes(data)
print(f"Restored {target.relative_to(HERE)} ({len(data)} bytes)")
