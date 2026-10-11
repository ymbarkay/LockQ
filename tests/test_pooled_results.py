"""Check the reported classifier gains against the saved process results.

The denominator is the number of measurement records, not the number
of classifier configurations.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = {"persistent_markov", "abrupt_single_change", "smooth_drift"}


def test_record_pooled_classifier_gains():
    df = pd.read_csv(ROOT / "results" / "classifiers" / "processes.csv")
    df = df[(df["scope"] == "low_margin") & df["true_family"].isin(DYNAMIC)]
    n = int(df["n_records"].sum())
    model_gain = float((df["n_records"] * df["v26_gain"]).sum()) / n
    oracle_gain = float((df["n_records"] * df["oracle_gain"]).sum()) / n

    checks = json.loads((ROOT / "docs" / "number_checks.json").read_text())
    observed = checks["vqc_low_margin"]
    assert n == observed["gain_record_count"] == 9888
    assert abs(model_gain - observed["v26_gain"][0]) < 1e-12
    assert abs(oracle_gain - observed["oracle_gain"][0]) < 1e-12
    assert observed["lockq_net_corrected_decisions"] == 22
    assert observed["oracle_net_corrected_decisions"] == 32
    assert abs(observed["pooled_recovery_percent"] - 68.75) < 1e-12
