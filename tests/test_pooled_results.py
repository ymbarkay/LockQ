"""Check the saved low-margin VQC record-pooled gains."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DYNAMIC={'persistent_markov','abrupt_single_change','smooth_drift'}

def test_record_pooled_classifier_gains():
    df=pd.read_csv(ROOT/'results/classifiers/processes.csv')
    df=df[(df.scope=='low_margin') & df.true_family.isin(DYNAMIC)]
    n=int(df.n_records.sum())
    ng=int(round((df.n_records*df.v26_gain).sum()))
    dg=int(round((df.n_records*df.oracle_gain).sum()))
    assert (n,ng,dg)==(9888,22,32)
    assert ng/dg==0.6875