# LockQ v0.1.0

Initial public release accompanying **LockQ: When Does Batch Order Matter in Finite-Shot Quantum Classification? An Exact Unordered Comparator** (IEEE QAI 2026).

## Included

- Exact unordered likelihood implementation for two- and three-state models.
- Drift simulation and estimator source in readable Python files.
- Saved model-comparison, dynamic-panel, and classifier process results.
- Tests for exact permutation agreement and record-pooled classifier gains.
- A numerical verification script and GitHub Actions workflow.
- MIT software license and citation metadata.

## Key measurements

| Experiment | Result |
| --- | ---: |
| Ordered reference-decision error | 7.74% |
| Dynamic-panel order-advantage recovery | 83.3% |
| Abrupt-change order-advantage recovery | 87.5% |
| Low-margin trained-classifier order advantage | 0.2225 percentage points |
| Corresponding oracle advantage | 0.3236 percentage points |
| Trained-classifier pooled recovery | 68.8% |

The gains in the last three rows are calculated across 9,888 measurement records from time-dependent processes.

## Verify

```bash
python -m pip install -e '.[verify]'
python -m pytest -q tests
python scripts/verify_published_results.py
```

All measurements were obtained by simulation.
