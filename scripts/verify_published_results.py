#!/usr/bin/env python3
"""Compute LockQ's paper metrics from frozen experiment outputs.

This reads experiment outputs, not reruns quantum simulations. See README for
commands and prerequisites for fresh simulation runs.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'results'
DYNAMIC = {'persistent_markov', 'abrupt_single_change', 'smooth_drift'}


def seed(*parts):
    return int.from_bytes(hashlib.sha256('::'.join(map(str, parts)).encode()).digest()[:8], 'little') % (2**63-1)


def boot_ci(x, seed_value, draws=20_000):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed_value)
    idx = rng.integers(0, len(x), (draws, len(x)))
    return tuple(np.quantile(x[idx].mean(axis=1), [0.025, 0.975]))


def check(label, measured, expected, tolerance):
    assert abs(measured-expected) <= tolerance, f'{label}: {measured} != {expected}'
    print(f'{label:35s} {measured:8.4f}   OK')


def main():
    sessions = pd.read_csv(ROOT/'estimator'/'sessions.csv')
    panel = pd.read_csv(ROOT/'estimator'/'dynamic_panel.csv')
    processes = pd.read_csv(ROOT/'classifiers'/'processes.csv')
    factorial = pd.read_csv(ROOT/'estimator'/'factorial_sessions.csv')
    units = pd.read_csv(ROOT/'classifiers'/'configurations.csv')

    assert len(sessions) == len(factorial) == 20
    check('LockQ ordered error (%)', 100*sessions.v26_ordered_error.mean(), 7.74, .006)
    check('Five-model BMA error (%)', 100*sessions.bma5_ordered_error.mean(), 8.56, .006)
    check('Six-model BMA error (%)', 100*factorial.bma6_ordered_error.mean(), 8.56, .006)
    check('Five-model stacking error (%)', 100*factorial.stack5_ordered_error.mean(), 7.60, .006)

    dynamic = 100*panel.v26_gain.sum()/panel.oracle_gain.sum()
    abrupt = panel[panel.true_family.eq('abrupt_single_change')]
    abrupt_ratio = 100*abrupt.v26_gain.sum()/abrupt.oracle_gain.sum()
    check('Dynamic-panel recovery (%)',dynamic,83.3333333,.05)
    check('Abrupt recovery (%)', abrupt_ratio,87.5,.05)
    check('Coverage (%)',100*sessions.action_coverage.mean(),86.08,.006)

    bm6 = 100*(factorial.bma6_ordered_error - factorial.stack6_ordered_error)
    bm5 = 100*(factorial.bma5_ordered_error - factorial.stack6_ordered_error)
    check('Six-model BMA improvement (pp)',bm6.mean(),.82,.001)
    assert np.allclose(boot_ci(bm6,seed('bma6_stack6')),[.02,1.72],atol=.005)
    assert np.allclose(boot_ci(bm5,seed('bma5_stack6')),[0,1.72],atol=.005)
    print('Factorial six-model BMA 95% CI       [0.02, 1.72] pp   OK')

    low = units.query("scope == 'low_margin'")
    assert len(low)==72 and low.n_records.sum()==9888
    assert (low.max_reference_margin<=.1).all()
    # Frozen classifier code computes abs(p_star - 0.5), not abs(2*p_star - 1).
    gain = 100*low.label_gain_vs_bma5
    ci = boot_ci(gain,seed('low_margin','label'))
    check('Dataset-label gain (pp)',gain.mean(),1.0416666667,.001)
    assert np.allclose(ci,[.50636574,1.59155575],atol=.005)
    print('Dataset-label gain 95% CI            [0.51, 1.59] pp   OK')

    dyn = processes[processes.scope.eq('low_margin') & processes.true_family.isin(DYNAMIC)]
    n = int(dyn.n_records.sum())
    num = int(round((dyn.n_records*dyn.v26_gain).sum()))
    den = int(round((dyn.n_records*dyn.oracle_gain).sum()))
    assert (n,num,den) == (9888,22,32)
    check('Trained-classifier recovery (%)',100*num/den,68.75,.01)
    print('All saved-output checks passed.')

if __name__=='__main__':
    main()