# Reproducing LockQ

LockQ computes an exact likelihood for an unordered multiset of measurement batches, allowing batch order to be evaluated without discarding batch structure.

## Install

```bash
python -m pip install -e ".[full]"
python scripts/restore_sources.py
```

## Test the likelihood

```bash
python -m pytest -q tests/test_exact.py
```

The tests compare the exact two- and three-state unordered likelihoods with exhaustive enumeration of distinct batch permutations.

## Reproduce the paper's numerical results

```bash
python scripts/verify_published_results.py
```

The result tables in `results/estimator/` and `results/classifiers/` contain the model errors and order advantages used to compute the paper's reported figures.

## Run the simulation

```bash
python experiments/run_estimator.py --mode full --out runs/estimator
```

The simulation uses the parameter grid and random seeds in the experiment code and may take substantial CPU time.

The trained-classifier evaluation additionally requires the saved four-qubit classifier model checkpoints, supplied separately from the result tables. Their original training procedure is not included here.

The estimator study and the targeted safeguard evaluation follow the protocols described in the paper. All measurement streams are simulated.
