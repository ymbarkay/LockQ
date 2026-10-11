#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, json, subprocess, sys, time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['smoke','full'],default='full')
p.add_argument('--out',default='/content/LockQ_v29_Full_Run')
p.add_argument('--fit-records',type=int,default=80)
p.add_argument('--risk-records',type=int,default=80)
p.add_argument('--permutations',type=int,default=499)
p.add_argument('--max-workers',type=int,default=4)
p.add_argument('--worker-timeout',type=int,default=900)
p.add_argument('--retries',type=int,default=2)
p.add_argument('--skip-external',action='store_true')
p.add_argument('--skip-targeted',action='store_true')
a=p.parse_args()
HERE=Path(__file__).resolve().parent
METHOD=HERE/'safeguard.py'
OUT=Path(a.out); (OUT/'logs').mkdir(parents=True,exist_ok=True); (OUT/'cache').mkdir(parents=True,exist_ok=True)

def command(kind,index):
    return [sys.executable,str(METHOD),'--mode',a.mode,'--out',str(OUT),'--fit-records',str(a.fit_records),'--risk-records',str(a.risk_records),'--permutations',str(a.permutations),'--worker-kind',kind,'--worker-index',str(index)]

def cache_path(kind,index):
    if kind=='infamily': return OUT/'cache'/f'infamily_{index}.pkl'
    if kind=='three': return OUT/'cache'/f'external_three_{index}.pkl'
    if kind=='semi': return OUT/'cache'/f'external_semi_{index}.pkl'
    return None

def run_one(task):
    kind,index=task
    cp=cache_path(kind,index)
    if cp is not None and cp.exists(): return {'kind':kind,'index':index,'status':'cached'}
    log=OUT/'logs'/f'{kind}_{index}.log'
    for attempt in range(1,a.retries+2):
        with log.open('a') as f:
            f.write(f'\n=== attempt {attempt} ===\n')
            try:
                r=subprocess.run(command(kind,index),stdout=f,stderr=subprocess.STDOUT,timeout=a.worker_timeout)
                if r.returncode==0: return {'kind':kind,'index':index,'status':'done','attempt':attempt}
            except subprocess.TimeoutExpired:
                f.write(f'\nTIMEOUT after {a.worker_timeout}s\n')
    return {'kind':kind,'index':index,'status':'failed'}

def run_group(tasks):
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.max_workers) as ex:
        for r in ex.map(run_one,tasks):
            results.append(r); print(json.dumps(r),flush=True)
    return results

n_inf=5 if a.mode=='smoke' else 20
results=run_group([('infamily',i) for i in range(n_inf)])
if not a.skip_external:
    n=1 if a.mode=='smoke' else 4
    results+=run_group([('three',i) for i in range(n)]+[('semi',i) for i in range(n)])
if not a.skip_targeted:
    n=3 if a.mode=='smoke' else 12
    results+=run_group([('targeted',i) for i in range(n)])
(OUT/'WORKER_STATUS.json').write_text(json.dumps(results,indent=2))
failed=[r for r in results if r['status']=='failed']
if failed:
    print(f'{len(failed)} workers failed; rerun this launcher to resume.',file=sys.stderr)
    sys.exit(2)
cmd=[sys.executable,str(METHOD),'--mode',a.mode,'--out',str(OUT),'--fit-records',str(a.fit_records),'--risk-records',str(a.risk_records),'--permutations',str(a.permutations),'--summarize-only']
subprocess.run(cmd,check=True)