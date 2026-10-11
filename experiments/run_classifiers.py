#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, sys, re, os, math, pickle, hashlib, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--bundle-root', default='inputs/vqc_bundle')
parser.add_argument('--selected-inputs', default='inputs/vqc_selected_inputs.csv',
                    help='Original selected reference probabilities and dataset labels (bypasses checkpoint loading)')
parser.add_argument('--out',default='runs/classifiers')
parser.add_argument('--worker-index',type=int,default=None)
parser.add_argument('--summarize-only',action='store_true')
args=parser.parse_args()
OUT=Path(args.out); (OUT/'cache').mkdir(parents=True,exist_ok=True)

# Import official LockQ implementation without executing its main.
saved=sys.argv[:]
sys.argv=['run_estimator.py','--mode','full','--out',str(OUT/'_method_import'),'--skip-targeted']
spec=importlib.util.spec_from_file_location('lockq26',str(Path(__file__).resolve().parent / 'run_estimator.py'))
L=importlib.util.module_from_spec(spec); sys.modules['lockq26']=L; spec.loader.exec_module(L)
sys.argv=saved
L.torch.set_num_threads(1)
try: L.torch.set_num_interop_threads(1)
except RuntimeError: pass

SEED=20260729
REPEATS=4
BINS=L.MARGINS
DYNAMIC={'persistent_markov','abrupt_single_change','smooth_drift'}
EXCHANGEABLE={'latent_static','iid_mode_mixture'}

def seed(*parts):
    return int.from_bytes(hashlib.sha256('::'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**63-1)

def load_selected_inputs():
    rows=[]
    root=Path(args.bundle_root)/'models'
    for p in sorted(root.glob('vqc22_*.npz')):
        m=re.match(r'vqc22_(.+)_d(\d+)_s(\d+)_',p.name)
        if not m: continue
        dataset,depth,train_seed=m.group(1),int(m.group(2)),int(m.group(3))
        z=np.load(p)
        pc=np.asarray(z['pclean'],float); yt=np.asarray(z['ytest'],int)
        margin=np.abs(pc-.5)
        config=f'{dataset}_d{depth}_s{train_seed}'
        for b,(lo,hi) in enumerate(BINS):
            idx=np.where((margin>=lo-1e-12)&(margin<=hi+1e-12))[0]
            if len(idx)==0: continue
            midpoint=(lo+hi)/2
            j=int(idx[np.argmin(np.abs(margin[idx]-midpoint))])
            for r in range(REPEATS):
                rows.append(dict(config=config,dataset=dataset,depth=depth,train_seed=train_seed,
                                 input_index=j,margin_bin=b,reference_probability=float(pc[j]),
                                 reference_margin=float(margin[j]),dataset_label=int(yt[j]),repeat=r))
    df=pd.DataFrame(rows)
    # Stable order is scientifically frozen before measurement streams.
    df=df.sort_values(['config','margin_bin','repeat']).reset_index(drop=True)
    return df

META = pd.read_csv(args.selected_inputs) if Path(args.selected_inputs).is_file() else load_selected_inputs()
if META.empty:
    raise ValueError('No selected classifier inputs found; provide --selected-inputs or original model bundle.')
(OUT/'vqc_selected_inputs.csv').write_text(META.to_csv(index=False))


def worker(i:int):
    path=OUT/'cache'/f'vqc_session_{i}.pkl'
    if path.exists(): return
    specs=L.true_specs_infamily()
    family,j,s=specs[i]
    theta=META.reference_probability.to_numpy(float)
    # 320 known calibration records with the official stratified design.
    tc=L.make_cal_thetas(L.N_CAL,seed('vqc_cal_theta',i))
    sc=L.base.simulate_sequences(s,tc,f'v26_vqc_cal_{i}',shared_seed=seed('vqc_cal',i))
    st=L.base.simulate_sequences(s,theta,f'v26_vqc_test_{i}',shared_seed=seed('vqc_test',i))
    truth=torch.tensor(theta>=.5,dtype=torch.long,device=L.DEVICE)
    inf=L.infer_session(sc,tc,st)
    oo=L.oracle_binary(st,s,'ordered'); ou=L.oracle_binary(st,s,'unordered')
    policy=inf['action_policy']
    act=L.apply_action_policy(inf['v26_o'],inf['v26_u'],1.0,policy,truth)
    payload={
      'session':i,'true_family':family,'within_family_index':j,'spec':s,
      'truth':truth.cpu().numpy(),
      'v26_o':inf['v26_o'].cpu().numpy(),'v26_u':inf['v26_u'].cpu().numpy(),
      'bma5_o':inf['bma5_o'].cpu().numpy(),'bma5_u':inf['bma5_u'].cpu().numpy(),
      'oracle_o':oo.cpu().numpy(),'oracle_u':ou.cpu().numpy(),
      'action_policy':policy,'action':act,'stack_weights':inf['stack_weights'],
      'retained_counts':inf['retained_counts']
    }
    path.write_bytes(pickle.dumps(payload))
    print(f'done {i} {family}',flush=True)


def bootstrap(x,B=20000,seed0=SEED):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    rng=np.random.default_rng(seed0)
    idx=rng.integers(0,len(x),(B,len(x)))
    b=x[idx].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=int(len(x)))

def summarize():
    records=[]; sessions=[]
    for p in sorted((OUT/'cache').glob('vqc_session_*.pkl')):
        o=pickle.loads(p.read_bytes()); i=o['session']; fam=o['true_family']; truth=o['truth']
        row=META.copy(); row['session']=i; row['true_family']=fam; row['reference_decision']=truth
        for name in ['v26_o','v26_u','bma5_o','bma5_u','oracle_o','oracle_u']:
            prob=o[name]; row[name+'_prob1']=prob[:,1]; row[name+'_pred']=prob.argmax(1); row[name+'_error']=(prob.argmax(1)!=truth).astype(int)
        row['v26_order_gain_record']=row['v26_u_error']-row['v26_o_error']
        row['bma5_order_gain_record']=row['bma5_u_error']-row['bma5_o_error']
        row['oracle_order_gain_record']=row['oracle_u_error']-row['oracle_o_error']
        row['v26_label_correct']=(row['v26_o_pred']==row['dataset_label']).astype(int)
        row['reference_label_correct']=(row['reference_decision']==row['dataset_label']).astype(int)
        records.append(row)
        sessions.append(dict(session=i,true_family=fam,action_coverage=o['action']['coverage'],
                             action_selective_risk=o['action']['selective_risk'],
                             action_ordered_fraction=o['action']['ordered_fraction'],
                             action_feasible=bool(o['action_policy'].get('feasible',False))))
    rec=pd.concat(records,ignore_index=True)
    rec.to_csv(OUT/'v26_vqc_record_results.csv',index=False)
    pd.DataFrame(sessions).to_csv(OUT/'v26_vqc_session_action_results.csv',index=False)

    # One experimental unit is one dataset-depth-training-seed configuration.
    unit=[]
    for config,g in rec.groupby('config'):
        base=dict(config=config,dataset=g.dataset.iloc[0],depth=int(g.depth.iloc[0]),train_seed=int(g.train_seed.iloc[0]))
        for scope,mask in [('low_margin',g.reference_margin<=.10),('all_margin',np.ones(len(g),dtype=bool))]:
            q=g[mask & g.true_family.isin(DYNAMIC)]
            if len(q)==0: continue
            r=dict(base,scope=scope,n_records=len(q),n_margin_bins=q.margin_bin.nunique(),
                   v26_ordered_error=q.v26_o_error.mean(),v26_unordered_error=q.v26_u_error.mean(),
                   v26_gain=q.v26_order_gain_record.mean(),bma5_ordered_error=q.bma5_o_error.mean(),
                   bma5_gain=q.bma5_order_gain_record.mean(),oracle_gain=q.oracle_order_gain_record.mean(),
                   label_agreement_change=(q.v26_label_correct-q.reference_label_correct).mean())
            unit.append(r)
    unit=pd.DataFrame(unit); unit.to_csv(OUT/'v26_vqc_unit_results.csv',index=False)

    proc=[]
    for fam,g in rec.groupby('true_family'):
        for scope,mask in [('low_margin',g.reference_margin<=.10),('all_margin',np.ones(len(g),dtype=bool))]:
            q=g[mask]
            proc.append(dict(true_family=fam,scope=scope,n_records=len(q),
                             v26_ordered_error=q.v26_o_error.mean(),v26_unordered_error=q.v26_u_error.mean(),
                             v26_gain=q.v26_order_gain_record.mean(),bma5_gain=q.bma5_order_gain_record.mean(),
                             oracle_gain=q.oracle_order_gain_record.mean()))
    proc=pd.DataFrame(proc); proc.to_csv(OUT/'v26_vqc_process_results.csv',index=False)

    summary={'status':'official LockQ v2.6.2 evaluated on saved 80 trained VQC configurations with new measurement streams',
             'n_configs':int(META.config.nunique()),'selected_records_per_session':int(len(META)),
             'calibration_records_per_session':int(L.N_CAL),'test_sessions':len(sessions)}
    for scope in ['low_margin','all_margin']:
        u=unit[unit.scope==scope]
        summary[scope]={
          'eligible_units':int(len(u)),
          'v26_gain':bootstrap(u.v26_gain,seed0=seed(scope,'v26gain')),
          'bma5_gain':bootstrap(u.bma5_gain,seed0=seed(scope,'bmagain')),
          'oracle_gain':bootstrap(u.oracle_gain,seed0=seed(scope,'oracle')),
          'v26_ordered_error':bootstrap(u.v26_ordered_error,seed0=seed(scope,'err')),
          'bma5_ordered_error':bootstrap(u.bma5_ordered_error,seed0=seed(scope,'bmaerr')),
          'label_agreement_change':bootstrap(u.label_agreement_change,seed0=seed(scope,'label')),
        }
    ex=proc[(proc.scope=='low_margin') & proc.true_family.isin(EXCHANGEABLE)]
    dy=proc[(proc.scope=='low_margin') & proc.true_family.isin(DYNAMIC)]
    summary['low_margin_process']=proc[proc.scope=='low_margin'].to_dict(orient='records')
    summary['exchangeable_mean_gain']=float(ex.v26_gain.mean())
    den=float(dy.oracle_gain.sum()); summary['dynamic_pooled_recovery']=float(dy.v26_gain.sum()/den) if den>1e-12 else math.nan
    ses=pd.DataFrame(sessions)
    summary['action']={'feasible_sessions':int(ses.action_feasible.sum()),'mean_coverage':float(ses.action_coverage.mean()),
                       'mean_selective_risk':float(ses.action_selective_risk.mean())}
    (OUT/'LOCKQ_V26_2_VQC_SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=True))
    print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__':
    if args.worker_index is not None: worker(args.worker_index)
    elif args.summarize_only: summarize()
    else:
        for i in range(len(L.true_specs_infamily())): worker(i)
        summarize()