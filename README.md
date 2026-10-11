# LockQ

**When Does Batch Order Matter in Finite-Shot Quantum Classification? An Exact Unordered Comparator**  
IEEE QAI 2026 · Youssouf Barkay Molimi

LockQ evaluates whether time order improves the reference decision of a finite-shot quantum classifier. The exact unordered comparator retains the same multiset of measurement batches but discards their time labels. Six noise-drift models are combined by cross-fitted predictive stacking.

## Install

```bash
python -m pip install -e '.[verify]'
python -m pytest -q tests
python scripts/verify_published_results.py
```

The verification script recomputes the paper's headline estimator and trained-classifier metrics, the five- versus six-model BMA confidence intervals, and confidence intervals over independent classifier configurations, using the supplied experiment outputs. **These checks recalculate saved outputs; they do not rerun the simulations.**

## Regenerate simulations

```bash
python -m pip install -e '.[full]'
python experiments/run_estimator.py --mode full --out runs/estimator
```

The trained-classifier experiments require the original four-qubit classifier input probabilities/labels (or checkpoints to reconstruct them) and the corresponding evaluation runner. The supplied classifier aggregates support numerical verification; they are not substitutes for generating new measurement streams. All results in the paper are from simulated measurement records, not quantum hardware.

The separately calibrated permutation safeguard in the camera-ready manuscript was evaluated under its own experimental protocol, distinct from the original 320-record estimator experiment. Its original complete confirmatory reproduction is not yet included in this small repository and is **not** asserted by the verification script.

## Results and numerical conventions

- Six-model BMA versus LockQ on the factorial study: **0.82 percentage points (95% CI [0.02, 1.72])**; the five-model comparison has a different CI.
- Trained-classifier low-margin threshold: **`abs(p_star - 0.5) <= 0.10`** (72 configurations).
- LockQ recovers **22/32 = 68.75%** of the record-pooled dynamic oracle gain on **9,888** low-margin measurement records.
- Recomputed dataset-label accuracy gain relative to five-model BMA is **1.04 points** (95% CI [0.51, 1.59]); the camera-ready text currently says 1.24 and needs correction.

## Structure

- `lockq/` — exact unordered likelihood.
- `experiments/` — model and experiment implementations.
- `results/` — saved evaluation tables needed by the verification script.
- `scripts/verify_published_results.py` — reproduction of reported statistics from saved outputs.
- `tests/` — exactness and pooled-number tests.

## Citation

See `CITATION.cff`. Licensed under MIT.