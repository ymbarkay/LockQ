# LockQ

**LockQ: When Does Batch Order Matter in Finite-Shot Quantum Classification? An Exact Unordered Comparator**  
Youssouf Barkay Molimi · IEEE QAI 2026

Research code for the **camera-ready paper**. The core estimator is the original six-model predictive stack (v2.6.2). LockQ compares the ordered sequence of measurement batches against the *exact unordered multiset* of those same batches, preserving batch structure while removing time labels.

## Reproduce the reported estimator numbers

```bash
python -m pip install -e '.[verify]'
python -m pytest -q tests/test_exact.py
python scripts/verify_published_results.py
```

The first command tests the standalone NumPy exact likelihood against all distinct permutations for two- and three-state models. The second verifies archived estimator outputs (not a new simulation run).

| Published estimator result | Recomputed |
| --- | ---: |
| Five-model BMA error | **8.56%** |
| LockQ ordered error | **7.74%** |
| Registered dynamic recovery | **83.3%** |
| Abrupt-change recovery | **87.5%** |
| Mean primary-study coverage | **86.08%** |
| Low-margin trained-classifier recovery | **68.75%** (22/32 net corrections) |

The GitHub VQC audit uses the archived *process-level counts*, not independent per-record reconstruction. The larger record-level CSV is in the separately retained experiment archive. The trained-circuit checkpoints are unavailable in this release; it is therefore **not a full from-scratch VQC reproduction**.

## Experiment code

The frozen estimator scripts are in `experiments/v26/`. The original simulator dependency is provided as a gzip archive to keep it bit-for-bit intact. To restore:

```bash
python scripts/restore_experiment_dependencies.py
```

Then see `REPRODUCIBILITY.md` for prerequisites and data availability. Large-scale simulations are much more expensive than the numerical audit.

## Scope

All reported measurements are simulated, not physical quantum hardware observations. The paper's **targeted safeguarded procedure** uses a separate calibration and evaluation protocol; **the broader v2.6 estimator results above must not be interpreted as evaluations of that safeguard**. Later exploratory or confirmatory versions are not included.

The current repository contains core estimator source, exact-comparator source and tests, original primary-session records, process-level VQC evidence, and audits. It does not yet contain every archived record-level VQC file or the complete frozen classifier checkpoint bundle.

**License:** MIT (see `LICENSE`). Please cite the associated IEEE QAI 2026 paper when using the implementation.
