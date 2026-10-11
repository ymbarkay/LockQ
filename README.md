# LockQ

**LockQ: When Does Batch Order Matter in Finite-Shot Quantum Classification? An Exact Unordered Comparator**  
Youssouf Barkay Molimi · IEEE QAI 2026

This repository accompanies the camera-ready research paper. The published version is based on the **v2.6 estimator**. Its core contribution is an exact likelihood for the unordered multiset of measurement batches, preserving the batches while removing time labels.

## Reproducibility scope

The archived experiment outputs have been independently audited: LockQ error **7.74%**, dynamic recovery **83.3%**, abrupt-change recovery **87.5%**, and trained-classifier record-pooled dynamic recovery **68.75%** (22 net corrected decisions out of 32 oracle corrections).

These are recalculations of the original simulation outputs, **not new end-to-end simulation runs**. All measurements are simulated, not hardware results. The final manuscript also contains a separately evaluated, targeted safeguard protocol; the original v2.6 estimator study should not be misrepresented as that later protocol.

The repository will contain the numerical tests and original estimator code. See `REPRODUCIBILITY.md` for scope and limitations.
