# LockQ

**When Does Batch Order Matter in Finite-Shot Quantum Classification? An Exact Unordered Comparator**  
Youssouf Barkay Molimi · IEEE QAI 2026

LockQ isolates the information in batch order by comparing a time-ordered sequence of finite-shot measurements against the **exact unordered multiset** of the same batches. The estimator combines six drift models using cross-fitted stacking; the safeguard separately tests for temporal structure and calibrates abstention.

## Install and verify

```bash
python -m pip install -e '.[verify]'
python -m pytest -q tests
python scripts/verify_published_results.py
```

The last command independently **recalculates** the paper's main numbers and confidence intervals from the supplied experiment tables; it does not rerun the simulations.

## Run experiments

Install simulation dependencies first:

```bash
python -m pip install -e '.[full]'
```

```bash
# Six-model estimator and registered dynamic/robustness panels (320 fitting records)
python experiments/run_estimator.py --mode full --out runs/estimator

# Independent calibration and permutation safeguard (80 fitting + 80 risk records)
python experiments/run_safeguard.py --mode full --out runs/safeguard \
  --fit-records 80 --risk-records 80 --permutations 499 --max-workers 4

# Four-qubit classifier evaluation using the original frozen selected-input probabilities
# and dataset labels. Run worker indices 0 through 19; then aggregate.
python experiments/run_classifiers.py --selected-inputs inputs/vqc_selected_inputs.csv \
  --out runs/classifiers --worker-index 0
python experiments/run_classifiers.py --selected-inputs inputs/vqc_selected_inputs.csv \
  --out runs/classifiers --summarize-only
```

For the trained-classifier evaluation, repeat `--worker-index` for **every index 0–19 before summarizing**. This uses archived reference probabilities and labels, not retraining the original quantum circuits. Simulation runs may take substantial CPU time. All measurement records are simulated; no hardware measurements are included.

## Conventions

- Reference margin: **`abs(p_star - 0.5)`**. The low-margin trained-classifier subset uses `<= 0.10` (72 independent configurations; 9,888 dynamic-process measurement records).
- The six-model BMA comparison is **0.82 percentage points, 95% CI [0.02, 1.72]**. The five-model BMA confidence interval is different.
- Record-pooled dynamic VQC recovery is **22 / 32 = 68.75%**.
- Dataset-label accuracy improvement over five-model BMA is **1.04 points, 95% CI [0.51, 1.59]** from the original records. The manuscript's earlier 1.24-point value requires correction.

**Scope:** The original estimator and safeguarded experiments use different calibration protocols. The saved tables verify the estimator and trained-classifier numbers, while reproducing the targeted safeguard numbers requires running its experiment; the repository does not claim that its full simulation results are already independently reproduced. The original trained-classifier checkpoints are not included.

MIT licensed. See `CITATION.cff`.