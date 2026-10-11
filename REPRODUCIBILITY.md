# Reproducing LockQ

LockQ uses an exact multiset likelihood to compare inference with and without batch time labels. This repository contains the likelihood implementation, simulation code, and result tables for the paper.

## Installation

```bash
python -m pip install -e '.[full]'
```

## Exact likelihood

```bash
python -m pytest -q tests/test_exact.py
```

The tests compare the two- and three-state unordered likelihoods with exhaustive enumeration of distinct permutations, including repeated batches.

## Paper results

```bash
python scripts/verify_published_results.py
```

This regenerates the reported estimator errors and recovery figures from the saved experimental results in `results/`. All percentages are computed from the supplied session and classifier tables.

## Run the estimator experiments

```bash
python experiments/run_estimator.py --mode full --out runs/estimator
```

The full simulation uses the parameter grids and random seeds defined in the source code and takes substantially longer than the numerical checks. The quantum-classifier evaluation reads previously trained four-qubit classifier checkpoints through `--bundle-root`; model checkpoint files must be supplied separately. The provided experiment data include the selected reference probabilities and saved classification outcomes.

The paper's targeted safeguard experiment uses its own calibration and evaluation protocol. Its eight exchangeable sessions and abrupt-change test are described separately from the main estimator experiments.

All experiments use simulated finite-shot measurements.
