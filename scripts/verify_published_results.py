#!/usr/bin/env python3
"""Recalculate published v2.6 estimator metrics from archived output CSVs.

Requires the CSV files from LockQ_CameraReady_Code_And_Results.zip.
Not a fresh simulation and not an audit of the targeted safeguarded rule.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "v26"
main = pd.read_csv(ROOT / "primary320" / "v26_independent_infamily_sessions.csv")
panel = pd.read_csv(ROOT / "primary320" / "v26_targeted_registered_dynamic_sessions.csv")
records = pd.read_csv(ROOT / "vqc" / "v26_vqc_record_results.csv")

vqc = records[
    (records.reference_margin <= 0.10)
    & records.true_family.isin(["persistent_markov", "abrupt_single_change", "smooth_drift"])
]
model_gain = int(vqc.v26_order_gain_record.sum())
oracle_gain = int(vqc.oracle_order_gain_record.sum())

def recovery(table):
    return float(table.v26_gain.sum() / table.oracle_gain.sum() * 100)

measured = {
    "five_model_bma_error_percent": float(main.bma5_ordered_error.mean() * 100),
    "lockq_error_percent": float(main.v26_ordered_error.mean() * 100),
    "dynamic_recovery_percent": recovery(panel),
    "abrupt_recovery_percent": recovery(panel[panel.true_family == "abrupt_single_change"]),
    "coverage_percent": float(main.action_coverage.mean() * 100),
    "low_margin_dynamic_vqc_records": int(len(vqc)),
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
passed = {k: abs(v - expected[k]) < 0.011 for k, v in measured.items()}
print(json.dumps({"scope": "archived outputs, not simulation rerun",
                  "measured": measured, "checks": passed}, indent=2))
if not all(passed.values()):
    raise SystemExit("Numerical audit FAILED")
