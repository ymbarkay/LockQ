# LockQ

**When Does Batch Order Matter in Finite-Shot Quantum Classification? An Exact Unordered Comparator**  
Youssouf Barkay Molimi · IEEE QAI 2026

LockQ measures the value of **batch order in finite-shot quantum classification**. It compares predictions from time-ordered measurement batches with predictions from the **exact unordered multiset of those same batches**. The comparison preserves each batch and removes only its time label, separating the information in order from the information in batching.

The estimator combines six models of device drift using cross-fitted predictive stacking. The paper also develops a permutation-based safeguard and a calibrated selective-prediction rule.

## Results

| Experiment | Result |
| --- | ---: |
| Reference-decision error | **7.74%** (8.56% with Bayesian model averaging) |
| Order advantage recovered on the dynamic panel | **83.3%** |
| Order advantage recovered under abrupt changes | **87.5%** |
| Order advantage recovered on trained classifiers near the decision boundary | **68.8%** |
| Safeguard on exchangeable sessions | **Zero ordered advantage** |

Experiments use 16 batches of 128 shots per input and simulated device-noise processes. The estimator and safeguard experiments use their respective protocols described in the paper.

## Install and verify

```bash
python -m pip install -e '.[verify]'
python -m pytest -q tests/test_exact.py
python scripts/verify_published_results.py
```

The tests compare the exact unordered likelihood with exhaustive permutation enumeration. The results script reconstructs the reported estimator measurements and recovery figures from the supplied experiment results.

## Code and results

| Directory | Contents |
| --- | --- |
| `lockq/` | Exact unordered likelihood for two- and three-state models |
| `experiments/` | Simulation, drift models and predictive inference |
| `results/estimator/` | Model comparison, dynamic-panel and robustness experiments |
| `results/classifiers/` | Four-qubit classifier evaluation |
| `tests/` | Exactness and exchangeability tests |
| `scripts/` | Reproduce published numerical results |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for execution details.

## Citation

Please cite *LockQ: When Does Batch Order Matter in Finite-Shot Quantum Classification? An Exact Unordered Comparator* (IEEE QAI 2026).

**License:** MIT.
