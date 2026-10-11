#!/usr/bin/env python3
"""Recalculate published v2.6 estimator metrics from archived CSVs.

The record-level VQC export is optional. If absent, verification of 22/32
uses the original *process-level* averages and counts, rather than claiming
an independent audit of each measurement record.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "v26"
main = pd.read_csv(ROOT / "primary320" / "v26_independent_infamily_sessions.csv")
panel = pd.read_csv(ROOT / "primary320" / "v26_targeted_registered_dynamic_sessions.csv")

def recovery(table):
    return float(table.v26_gain.sum() / table.oracle_gain.sum() * 100)

record_path = ROOT / "vqc" / "v26_vqc_record_results.csv"
if record_path.exists():
    records = pd.read_csv(record_path)
    vqc = records[
        (records.reference_margin <= 0.10)
        & records.true_family.isin(
            ["persistent_markov", "abrupt_single_change", "smooth_drift"]
        )
    ]
    record_count = len(vqc)
    model_gain = int(vqc.v26_order_gain_record.sum())
    oracle_gain = int(vqc.oracle_order_gain_record.sum())
    vqc_scope = "archived individual measurement records"
else:
    by_process = pd.read_csv(ROOT / "vqc" / "v26_vqc_process_results.csv")
    vqc = by_process[
        (by_process.scope == "low_margin")
        & by_process.true_family.isin(
            ["persistent_markov", "abrupt_single_change", "smooth_drift"]
        )
    ]
    record_count = int(vqc.n_records.sum())
    model_gain = int(round((vqc.n_records * vqc.v26_gain).sum()))
    oracle_gain = int(round((vqc.n_records * vqc.oracle_gain).sum()))
    vqc_scope = "archived process-level aggregates, not individual records"

measured = {
    "five_model_bma_error_percent": float(main.bma5_ordered_error.mean() * 100),
    "lockq_error_percent": float(main.v26_ordered_error.mean() * 100),
    "dynamic_recovery_percent": recovery(panel),
    "abrupt_recovery_percent": recovery(
        panel[panel.true_family == "abrupt_single_change"]
    ),
    "coverage_percent": float(main.action_coverage.mean() * 100),
    "low_margin_dynamic_vqc_records": record_count,
    "vqc_model_net_corrections": model_gain,
    "vqc_oracle_net_corrections": oracle_gain,
    "vqc_recovery_percent": model_gain / oracle_gain * 100,
}
expected = {
    "five_model_bma_error_percent": 8.56,
    "lockq_error_percent": 7.74,
    "dynamic_recovery_percent": 83.333333333,
    "abrupt_recovery_percent": 87.5,
    "coverage_percent": 86.08,
    "low_margin_dynamic_vqc_records": 9888,
    "vqc_model_net_corrections": 22,
    "vqc_oracle_net_corrections": 32,
    "vqc_recovery_percent": 68.75,
}
checks = {k: abs(v - expected[k]) < 0.011 for k, v in measured.items()}
print(json.dumps({
    "scope": "archived results, not a fresh simulation",
    "vqc_evidence": vqc_scope,
    "measured": measured, "checks": checks,
    "all_pass": all(checks.values())
}, indent=2))
if not all(checks.values()):
    raise SystemExit("Numerical audit FAILED")
