# Reproducibility and provenance

This repository accompanies the camera-ready LockQ paper. The research implementation is **v2.6.2**, not subsequent experimental variants.

## Numerical audit (runs from the repository)

```bash
python -m pip install -e '.[verify]'
python -m pytest -q tests/test_exact.py
python scripts/verify_published_results.py
```

`tests/test_exact.py` checks the exact unordered likelihood against exhaustive enumeration for 2- and 3-state hidden-state models, including repeated observed batch values.

`scripts/verify_published_results.py` recalculates six-model estimator metrics from the saved primary-session tables and VQC process aggregates. It checks 8.56% BMA error, 7.74% LockQ error, 83.3% dynamic recovery, 87.5% abrupt recovery, 86.08% primary coverage, and the pooled trained-classifier result **22 / 32 = 68.75%**.

The VQC process table retains 9,888 low-margin dynamic observations across the three time-dependent generating families. Those aggregates allow an independent check of the numerical ratio but **do not provide all underlying individual decisions**; the complete 7-MB record-level file is retained in the separately downloadable experiment archive.

## Source organization

- `lockq/exact.py`: independently tested NumPy reference implementation.
- `experiments/v26/LockQ_v26_2_FinalStack.py`: archived six-family estimator.
- `experiments/v26/LockQ_v25_Robust_Extension.py`: archived estimator dependency.
- `experiments/v26/lockq_base.py.gz`: compressed simulation helper. Run `python scripts/restore_experiment_dependencies.py` to restore the importable Python file.
- `results/v26/primary320/`: saved primary session and registered-panel outputs.
- `results/v26/vqc/v26_vqc_process_results.csv`: saved classifier-by-process aggregates.
- `docs/LOCKQ_OFFICIAL_V26_2_METHOD.json`: frozen method description and artifact manifest.

The archived evaluator for the 80 previously trained four-qubit classifiers and their original `.npz` checkpoints are not both available from the public repository at present. The numerical audits therefore **do not claim end-to-end re-execution of circuit training or measurement simulation**.

## Separate safeguard evaluation

The final paper also describes a safeguard with a permutation certificate, an exact-unordered fallback, and independent selective-risk calibration. This **targeted experiment is not the same as** the original broad estimator study. Results from one protocol must not be presented as if they were generated under the other.

The later exploratory v2.9 research variant is not part of this camera-ready repository.

All experiments described here are simulations, not device-level hardware measurements.
