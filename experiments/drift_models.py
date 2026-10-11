#!/usr/bin/env python3
from __future__ import annotations
import os, sys, math, json, pickle, hashlib, importlib.util, time, argparse
from pathlib import Path
from dataclasses import dataclass, asdict, replace
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import qmc

parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['smoke','full'],default=os.environ.get('LOCKQ_V25_MODE','full'))
parser.add_argument('--out',default=os.environ.get('LOCKQ_V25_OUT','runs/drift_models'))
args=parser.parse_args()
SMOKE=args.mode=='smoke'
OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True); (OUT/'cache').mkdir(exist_ok=True)
HERE=Path(__file__).resolve().parent
if not (HERE / 'simulation.py').exists():
    import gzip
    (HERE / 'simulation.py').write_bytes(
        gzip.decompress((HERE / 'simulation.py.gz').read_bytes())
    )
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); DTYPE=torch.float64
torch.set_num_threads(min(12,os.cpu_count() or 4))
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
spec=importlib.util.spec_from_file_location('base',str(HERE/'simulation.py'))
base=importlib.util.module_from_spec(spec); sys.modules['base']=base; spec.loader.exec_module(base)
SessionSpec=base.SessionSpec

SEED=20260725
FAMILIES=['latent_static','iid_mode_mixture','persistent_markov','abrupt_single_change','smooth_drift','persistent_markov_3state']
BINARY_FAMILIES=FAMILIES[:5]
ACTIVE=['persistent_markov','abrupt_single_change','smooth_drift']
MARGINS=[(0.005,.02),(.02,.05),(.05,.10),(.10,.20),(.20,.38)]
THETA_GRID=np.linspace(.12,.88,21 if SMOKE else 41)
CLASS_MASK=torch.tensor(THETA_GRID>=.5,device=DEVICE)
THETA_LOGPRIOR=torch.full((len(THETA_GRID),),-math.log(len(THETA_GRID)),dtype=DTYPE,device=DEVICE)
N_CAND=3 if SMOKE else 8
N_SESS_PER_FAMILY=1 if SMOKE else 4
N_CAL=24 if SMOKE else 72
PER_STRAT_TEST=8 if SMOKE else 40
N_FOLDS=3 if SMOKE else 4

RANGES={'persistence':[0.55,0.995], 'occupancy':[0.10,0.90], 'depolarizing_gap':[0.08,0.28], 'readout_gap':[0.04,0.22], 'readout_asymmetry':[-0.14,0.14], 'calibration_log_gap':[-0.75,0.30], 'drift_amplitude':[0.08,0.55], 'change_center':[0.20,0.80], 'change_spread':[0.05,0.25]}

def seed(*parts): return int.from_bytes(hashlib.sha256('::'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**63-1)

def logsumexp_np(a,axis=None):
    m=np.max(a,axis=axis,keepdims=True); out=m+np.log(np.sum(np.exp(a-m),axis=axis,keepdims=True)); return np.squeeze(out,axis=axis)

def unit_to_binary(u,f):
    def lin(name,j): a,b=RANGES[name]; return float(a+u[j]*(b-a))
    return SessionSpec(f,2048,16,128,lin('persistence',0),lin('occupancy',1),lin('depolarizing_gap',2),lin('readout_gap',3),lin('readout_asymmetry',4),lin('calibration_log_gap',5),lin('drift_amplitude',6),lin('change_center',7),lin('change_spread',8))

def sobol_binary(n,seed0,f):
    U=qmc.Sobol(9,scramble=True,seed=seed0).random_base2(math.ceil(math.log2(n)))[:n]
    return [unit_to_binary(u,f) for u in U]

@dataclass(frozen=True)
class ThreeStateSpec:
    total_shots:int=2048; n_batches:int=16; shots_per_batch:int=128
    persistence:float=.92; occupancy0:float=.50; occupancy1:float=.30
    dep1:float=.10; dep2:float=.24; read1:float=.06; read2:float=.17
    asym1:float=.02; asym2:float=-.08; cal1:float=-.20; cal2:float=-.60

def pi3(s): return np.array([s.occupancy0,s.occupancy1,1-s.occupancy0-s.occupancy1],float)
def A3(s):
    pi=pi3(s); return s.persistence*np.eye(3)+(1-s.persistence)*np.tile(pi,(3,1))
def obs3(theta,h,s):
    dep=[.01,.01+s.dep1,.01+s.dep2][h]; p=(1-dep)*theta+dep*.5
    lo=math.log(p/(1-p)); p=1/(1+math.exp(-math.exp([0,s.cal1,s.cal2][h])*lo))
    rg=[0,s.read1,s.read2][h]; a=[0,s.asym1,s.asym2][h]
    e01=np.clip(.01+rg+a/2,0,.49); e10=np.clip(.01+rg-a/2,0,.49)
    return float(np.clip((1-e10)*p+e01*(1-p),1e-12,1-1e-12))
def emissions3(theta,s): return np.array([obs3(theta,h,s) for h in range(3)])

def unit_to_three(u):
    o0=.32+.30*u[0]; o1=.16+.24*u[1]
    if o0+o1>.88: o1=.88-o0
    return ThreeStateSpec(persistence=.78+.20*u[2],occupancy0=o0,occupancy1=o1,
        dep1=.04+.12*u[3],dep2=.15+.18*u[4],read1=.025+.08*u[5],read2=.10+.14*u[6],
        asym1=-.04+.08*u[7],asym2=-.14+.28*u[8],cal1=-.10-.25*u[5],cal2=-.40-.40*u[6])
def sobol_three(n,seed0):
    U=qmc.Sobol(9,scramble=True,seed=seed0).random_base2(math.ceil(math.log2(n)))[:n]
    return [unit_to_three(u) for u in U]

def sim3(s,thetas,seed0):
    rng=np.random.default_rng(seed0); B=len(thetas); out=np.empty((B,s.n_batches),np.int64); pi=pi3(s); A=A3(s); h=rng.choice(3,size=B,p=pi)
    for t in range(s.n_batches):
        ps=np.array([obs3(float(th),int(hh),s) for th,hh in zip(thetas,h)])
        out[:,t]=rng.binomial(s.shots_per_batch,ps)
        if t<s.n_batches-1:
            cs=np.cumsum(A[h],axis=1); h=(rng.random(B)[:,None]>cs).sum(1)
    return torch.tensor(out,dtype=torch.long,device=DEVICE)

def binary_ordered_grid(seqs,s):
    B,T=seqs.shape; Q=len(THETA_GRID)
    E=torch.tensor(np.stack([base.emissions_theta(float(th),s) for th in THETA_GRID]),dtype=DTYPE,device=DEVICE)
    ll=base.log_binom_torch(seqs[:,:,None,None],s.shots_per_batch,E[None,None,:,:]) # B,T,Q,2
    if s.family=='abrupt_single_change':
        vals=[]
        if T==1:
            for h,w in [(0,1-s.occupancy),(1,s.occupancy)]: vals.append(ll[:,0,:,h]+math.log(max(w,1e-300)))
        else:
            taus,tw=base.changepoint_weights(s)
            for tau,w in zip(taus,tw):
                for direction,dw in [(0,1-s.occupancy),(1,s.occupancy)]:
                    path=np.r_[np.full(int(tau),direction),np.full(T-int(tau),1-direction)].astype(int)
                    idx=torch.tensor(path,dtype=torch.long,device=DEVICE)
                    vals.append(ll.gather(3,idx[None,:,None,None].expand(B,T,Q,1)).squeeze(-1).sum(1)+math.log(max(float(w*dw),1e-300)))
        return torch.logsumexp(torch.stack(vals,-1),-1)
    As,pi0=base.transition_schedule(s)
    alpha=torch.log(torch.tensor(pi0,dtype=DTYPE,device=DEVICE))[None,None,:]+ll[:,0,:,:]
    for t in range(1,T):
        A=torch.tensor(As[t-1],dtype=DTYPE,device=DEVICE)
        alpha=torch.logsumexp(alpha.unsqueeze(-1)+torch.log(torch.clamp(A,1e-300))[None,None,:,:],-2)+ll[:,t,:,:]
    return torch.logsumexp(alpha,-1)

def occupancy_lw_binary(s):
    T=s.n_batches
    if s.family=='abrupt_single_change':
        out=torch.full((T+1,),-torch.inf,dtype=DTYPE,device=DEVICE)
        if T==1:
            out[0]=math.log(max(1-s.occupancy,1e-300)); out[1]=math.log(max(s.occupancy,1e-300)); return out
        taus,tw=base.changepoint_weights(s)
        for tau,w in zip(taus,tw):
            for direction,dw in [(0,1-s.occupancy),(1,s.occupancy)]:
                q=T-int(tau) if direction==0 else int(tau)
                out[q]=torch.logaddexp(out[q],torch.tensor(math.log(max(float(w*dw),1e-300)),dtype=DTYPE,device=DEVICE))
        return out
    As,pi0=base.transition_schedule(s)
    cur=torch.full((2,T+1),-torch.inf,dtype=DTYPE,device=DEVICE); cur[0,0]=math.log(max(pi0[0],1e-300)); cur[1,1]=math.log(max(pi0[1],1e-300))
    for t in range(1,T):
        A=torch.tensor(As[t-1],dtype=DTYPE,device=DEVICE); new=torch.full_like(cur,-torch.inf)
        for h in range(2):
            prev=cur[:,:t+1]+torch.log(torch.clamp(A[:,h],1e-300))[:,None]
            new[h,h:h+t+1]=torch.logsumexp(prev,0)
        cur=new
    return torch.logsumexp(cur,0)

def binary_unordered_grid(seqs,s):
    B,T=seqs.shape; Q=len(THETA_GRID)
    E=torch.tensor(np.stack([base.emissions_theta(float(th),s) for th in THETA_GRID]),dtype=DTYPE,device=DEVICE)
    ll=base.log_binom_torch(seqs[:,:,None,None],s.shots_per_batch,E[None,None,:,:])
    g=torch.full((B,Q,T+1),-torch.inf,dtype=DTYPE,device=DEVICE); g[:,:,0]=0
    for t in range(T):
        old=g; new=torch.full_like(g,-torch.inf)
        new[:,:,:t+1]=old[:,:,:t+1]+ll[:,t,:,0,None]
        new[:,:,1:t+2]=torch.logaddexp(new[:,:,1:t+2],old[:,:,:t+1]+ll[:,t,:,1,None]); g=new
    q=torch.arange(T+1,dtype=DTYPE,device=DEVICE); lw=occupancy_lw_binary(s)
    coeff=lw+torch.lgamma(q+1)+torch.lgamma(torch.tensor(float(T),dtype=DTYPE,device=DEVICE)-q+1)-torch.lgamma(torch.tensor(float(T+1),dtype=DTYPE,device=DEVICE))
    return torch.logsumexp(g+coeff[None,None,:],-1)

def three_ordered_grid(seqs,s):
    B,T=seqs.shape; E=torch.tensor(np.stack([emissions3(float(th),s) for th in THETA_GRID]),dtype=DTYPE,device=DEVICE)
    ll=base.log_binom_torch(seqs[:,:,None,None],s.shots_per_batch,E[None,None,:,:])
    A=torch.tensor(A3(s),dtype=DTYPE,device=DEVICE); pi=torch.tensor(pi3(s),dtype=DTYPE,device=DEVICE)
    alpha=torch.log(pi)[None,None,:]+ll[:,0,:,:]
    for t in range(1,T): alpha=torch.logsumexp(alpha.unsqueeze(-1)+torch.log(A)[None,None,:,:],-2)+ll[:,t,:,:]
    return torch.logsumexp(alpha,-1)

def occ3_lw(s):
    T=s.n_batches; A=A3(s); pi=pi3(s); neg=-1e300
    cur=np.full((3,T+1,T+1),neg)
    for h in range(3): cur[h,1 if h==0 else 0,1 if h==1 else 0]=math.log(pi[h])
    for t in range(1,T):
        new=np.full_like(cur,neg)
        for hp in range(3):
            for n0 in range(t+1):
                for n1 in range(t+1-n0):
                    v=cur[hp,n0,n1]
                    if v<neg/2: continue
                    for h in range(3):
                        i=n0+(h==0); j=n1+(h==1); new[h,i,j]=np.logaddexp(new[h,i,j],v+math.log(A[hp,h]))
        cur=new
    return np.logaddexp.reduce(cur,axis=0)

def three_unordered_grid(seqs,s):
    B,T=seqs.shape; Q=len(THETA_GRID)
    E=torch.tensor(np.stack([emissions3(float(th),s) for th in THETA_GRID]),dtype=DTYPE,device=DEVICE)
    ll=base.log_binom_torch(seqs[:,:,None,None],s.shots_per_batch,E[None,None,:,:])
    g=torch.full((B,Q,T+1,T+1),-torch.inf,dtype=DTYPE,device=DEVICE); g[:,:,0,0]=0
    for t in range(T):
        old=g; new=torch.full_like(g,-torch.inf)
        new[:,:,:t+1,:t+1]=torch.logaddexp(new[:,:,:t+1,:t+1],old[:,:,:t+1,:t+1]+ll[:,t,:,2,None,None])
        new[:,:,1:t+2,:t+1]=torch.logaddexp(new[:,:,1:t+2,:t+1],old[:,:,:t+1,:t+1]+ll[:,t,:,0,None,None])
        new[:,:,:t+1,1:t+2]=torch.logaddexp(new[:,:,:t+1,1:t+2],old[:,:,:t+1,:t+1]+ll[:,t,:,1,None,None]); g=new
    lw=occ3_lw(s); terms=[]
    for n0 in range(T+1):
        for n1 in range(T+1-n0):
            n2=T-n0-n1
            if lw[n0,n1]<-1e200: continue
            coef=lw[n0,n1]+math.lgamma(n0+1)+math.lgamma(n1+1)+math.lgamma(n2+1)-math.lgamma(T+1)
            terms.append(g[:,:,n0,n1]+coef)
    return torch.logsumexp(torch.stack(terms,-1),-1)

def ll_grid(seqs,s,rep):
    if isinstance(s,SessionSpec): return binary_ordered_grid(seqs,s) if rep=='ordered' else binary_unordered_grid(seqs,s)
    return three_ordered_grid(seqs,s) if rep=='ordered' else three_unordered_grid(seqs,s)

def known_ll(seqs,thetas,s):
    out=torch.empty(len(thetas),dtype=DTYPE,device=DEVICE)
    for th in np.unique(thetas):
        idx=np.where(np.isclose(thetas,th))[0]; ix=torch.tensor(idx,dtype=torch.long,device=DEVICE); sub=seqs[ix]
        if isinstance(s,SessionSpec): val=base.ordered_loglik_state_model(sub,float(th),s)
        else:
            # direct 3-state likelihood
            E=torch.tensor(emissions3(float(th),s),dtype=DTYPE,device=DEVICE); A=torch.tensor(A3(s),dtype=DTYPE,device=DEVICE); pi=torch.tensor(pi3(s),dtype=DTYPE,device=DEVICE)
            alpha=torch.log(pi)[None,:]+base.log_binom_torch(sub[:,0,None],s.shots_per_batch,E[None,:])
            for t in range(1,s.n_batches): alpha=torch.logsumexp(alpha.unsqueeze(-1)+torch.log(A)[None,:,:],-2)+base.log_binom_torch(sub[:,t,None],s.shots_per_batch,E[None,:])
            val=torch.logsumexp(alpha,-1)
        out[ix]=val
    return out

def class_and_total(logtheta):
    z=logtheta+THETA_LOGPRIOR[None,:]
    c0=torch.logsumexp(z[:,~CLASS_MASK],-1); c1=torch.logsumexp(z[:,CLASS_MASK],-1); total=torch.logsumexp(z,-1)
    return torch.stack([c0,c1],-1),total

def candidate_predictive(seqs,cands,rep):
    classes=[]; totals=[]
    for s in cands:
        c,t=class_and_total(ll_grid(seqs,s,rep)); classes.append(c); totals.append(t)
    return torch.stack(classes,1),torch.stack(totals,1) # B,K,2 ; B,K

def mix_family(cand_class,cand_total,param_logw):
    # candidate joint class evidence combined with posterior weights from calibration
    cj=torch.logsumexp(cand_class+param_logw[None,:,None],1)
    tot=torch.logsumexp(cand_total+param_logw[None,:],1)
    probs=torch.softmax(cj,-1)
    return probs,tot

def fit_stacking(P,y):
    # P: N,F,2 probabilities
    eps=1e-12; target=P[np.arange(len(y)),:,y]
    def softmax(a):
        a=a-np.max(a); e=np.exp(a); return e/e.sum()
    def obj(a):
        w=softmax(a); return -np.log(np.clip(target@w,eps,1)).sum()+1e-4*np.sum(a*a)
    r=minimize(obj,np.zeros(P.shape[1]),method='BFGS',options={'maxiter':500,'gtol':1e-8})
    return softmax(r.x)

def make_cal_thetas(n):
    vals=np.array([.20,.30,.40,.60,.70,.80]); return np.resize(vals,n)
def make_test_thetas(per,seed0):
    rng=np.random.default_rng(seed0); vals=[]
    for lo,hi in MARGINS:
        m=rng.uniform(lo,hi,per); side=rng.integers(0,2,per); vals.extend(.5+np.where(side,m,-m))
    return np.asarray(vals)
def folds_stratified(theta,k):
    # distribute each theta value across folds
    folds=np.empty(len(theta),int)
    for th in np.unique(theta):
        idx=np.where(np.isclose(theta,th))[0]; folds[idx]=np.arange(len(idx))%k
    return folds

def candidate_library():
    lib={f:sobol_binary(N_CAND,seed('cand',f),f) for f in BINARY_FAMILIES}; lib['persistent_markov_3state']=sobol_three(N_CAND,seed('cand','3state')); return lib
LIB=candidate_library()

def infer_session(seq_cal,theta_cal,seq_test):
    ycal=(theta_cal>=.5).astype(int); folds=folds_stratified(theta_cal,N_FOLDS)
    # Session-specific calibration-only refinement of the abrupt-change prior over all change locations.
    session_lib={f:list(c) for f,c in LIB.items()}
    initial=session_lib['abrupt_single_change']
    K0=torch.stack([known_ll(seq_cal,theta_cal,s) for s in initial],1)
    parent=initial[int(torch.argmax(K0.sum(0)).item())]
    refined=[]
    for centre in np.linspace(.10,.90,9):
        for spread in [.04,.08,.12,.18,.26,.36]:
            refined.append(replace(parent,change_center=float(centre),change_spread=float(spread)))
    Kr=torch.stack([known_ll(seq_cal,theta_cal,s) for s in refined],1)
    keep=torch.argsort(Kr.sum(0),descending=True)[:8].cpu().numpy()
    session_lib['abrupt_single_change']=[refined[int(i)] for i in keep]
    # Fit parameter support on calibration records before any test likelihood is evaluated.
    known_all={}
    chosen={}
    for f,cands in session_lib.items():
        K=torch.stack([known_ll(seq_cal,theta_cal,s) for s in cands],1)
        known_all[f]=K
        # Two retained settings per binary family, one for the more expensive 3-state family.
        nkeep=1 if f=='persistent_markov_3state' else 2
        ix=torch.argsort(K.sum(0),descending=True)[:nkeep].cpu().numpy()
        chosen[f]=[cands[int(i)] for i in ix]
    known={}; cal_pred={}; test_pred_o={}; test_pred_u={}
    for f,cands in chosen.items():
        # Recompute only the retained support; selection used calibration data only.
        known[f]=torch.stack([known_ll(seq_cal,theta_cal,s) for s in cands],1)
        cal_pred[f]=candidate_predictive(seq_cal,cands,'ordered')
        test_pred_o[f]=candidate_predictive(seq_test,cands,'ordered')
        if f in ('latent_static','iid_mode_mixture'):
            test_pred_u[f]=test_pred_o[f]
        else:
            test_pred_u[f]=candidate_predictive(seq_test,cands,'unordered')
    cvP=np.zeros((len(theta_cal),len(FAMILIES),2)); cvLogTot=np.zeros((len(theta_cal),len(FAMILIES)))
    for fold in range(N_FOLDS):
        tr=folds!=fold; va=folds==fold
        for fi,f in enumerate(FAMILIES):
            lw=torch.log_softmax(known[f][tr].sum(0),0)
            pc,pt=cal_pred[f]; probs,tot=mix_family(pc[va],pt[va],lw)
            cvP[va,fi]=probs.detach().cpu().numpy(); cvLogTot[va,fi]=tot.detach().cpu().numpy()
    sw=fit_stacking(cvP,ycal)
    cv_mix_prob=(cvP*sw[None,:,None]).sum(1)
    cv_mix_logtot=logsumexp_np(cvLogTot+np.log(np.clip(sw,1e-300,1))[None,:],axis=1)
    fam_o=[]; fam_u=[]; fam_tot=[]; family_cal_evidence=[]
    for f in FAMILIES:
        lw=torch.log_softmax(known[f].sum(0),0)
        po,to=mix_family(*test_pred_o[f],lw); pu,tu=mix_family(*test_pred_u[f],lw)
        fam_o.append(po); fam_u.append(pu); fam_tot.append(to)
        family_cal_evidence.append(torch.logsumexp(known[f].sum(0)-math.log(len(chosen[f])),0))
    fam_o=torch.stack(fam_o,1); fam_u=torch.stack(fam_u,1); fam_tot=torch.stack(fam_tot,1)
    swt=torch.tensor(sw,dtype=DTYPE,device=DEVICE)
    stack_o=(fam_o*swt[None,:,None]).sum(1); stack_u=(fam_u*swt[None,:,None]).sum(1)
    fw=torch.softmax(torch.stack(family_cal_evidence),0)
    bma_o=(fam_o*fw[None,:,None]).sum(1); bma_u=(fam_u*fw[None,:,None]).sum(1)
    test_score=-torch.logsumexp(fam_tot+torch.log(torch.clamp(swt,1e-300))[None,:],1).detach().cpu().numpy()/seq_test.shape[1]
    cv_score=-cv_mix_logtot/seq_cal.shape[1]
    return {'stack_o':stack_o,'stack_u':stack_u,'bma_o':bma_o,'bma_u':bma_u,'test_score':test_score,'cv_score':cv_score,'stack_weights':sw,'bma_weights':fw.detach().cpu().numpy(),'cv_accuracy':float((cv_mix_prob.argmax(1)==ycal).mean())}

def oracle_binary(seq,s,rep): return torch.softmax(class_and_total(ll_grid(seq,s,rep))[0],-1)
def oracle_three(seq,s,rep): return torch.softmax(class_and_total(ll_grid(seq,s,rep))[0],-1)

def true_specs_infamily():
    out=[]
    for f in BINARY_FAMILIES:
        specs=sobol_binary(N_SESS_PER_FAMILY,seed('true',f),f)
        out.extend((f,i,s) for i,s in enumerate(specs))
    return out

def metrics(p,truth):
    oh=torch.nn.functional.one_hot(truth,2).to(DTYPE)
    return {'error':float((p.argmax(-1)!=truth).double().mean()),'brier':float(((p-oh)**2).sum(-1).mean()),'confidence':float(p.max(-1).values.mean())}

def run_infamily():
    rows=[]; record_paths=[]
    for idx,(fam,j,s) in enumerate(true_specs_infamily()):
        cache=OUT/'cache'/f'infamily_{idx}.pkl'
        if cache.exists(): payload=pickle.loads(cache.read_bytes()); rows.append(payload['row']); record_paths.append(cache); print('in-family',idx+1,'resumed',flush=True); continue
        tc=make_cal_thetas(N_CAL); sc=base.simulate_sequences(s,tc,f'v25_cal_{idx}',shared_seed=seed('simcal',idx))
        tt=make_test_thetas(PER_STRAT_TEST,seed('theta',idx)); st=base.simulate_sequences(s,tt,f'v25_test_{idx}',shared_seed=seed('simtest',idx)); truth=torch.tensor(tt>=.5,dtype=torch.long,device=DEVICE)
        inf=infer_session(sc,tc,st); oo=oracle_binary(st,s,'ordered'); ou=oracle_binary(st,s,'unordered')
        m={k:metrics(inf[k],truth) for k in ['bma_o','bma_u','stack_o','stack_u']}; mo=metrics(oo,truth); mu=metrics(ou,truth)
        row={'session':idx,'true_family':fam,'within_family_index':j,'oracle_ordered_error':mo['error'],'oracle_unordered_error':mu['error'],
             'bma_ordered_error':m['bma_o']['error'],'bma_unordered_error':m['bma_u']['error'],'stack_ordered_error':m['stack_o']['error'],'stack_unordered_error':m['stack_u']['error'],
             'oracle_gain':mu['error']-mo['error'],'bma_gain':m['bma_u']['error']-m['bma_o']['error'],'stack_gain':m['stack_u']['error']-m['stack_o']['error'],
             'bma_brier':m['bma_o']['brier'],'stack_brier':m['stack_o']['brier'],'stack_confidence':m['stack_o']['confidence'],'cv_accuracy':inf['cv_accuracy'],
             **{f'stack_weight_{f}':float(w) for f,w in zip(FAMILIES,inf['stack_weights'])}, **{f'bma_weight_{f}':float(w) for f,w in zip(FAMILIES,inf['bma_weights'])}}
        payload={'row':row,'theta_test':tt,'truth':truth.cpu().numpy(),'stack_o':inf['stack_o'].cpu().numpy(),'stack_u':inf['stack_u'].cpu().numpy(),'bma_o':inf['bma_o'].cpu().numpy(),'test_score':inf['test_score'],'cv_score':inf['cv_score']}
        cache.write_bytes(pickle.dumps(payload)); rows.append(row); record_paths.append(cache); print('in-family',idx+1,'/',len(true_specs_infamily()),fam,'done',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'independent_infamily_sessions.csv',index=False)
    return df,record_paths

# semi-Markov expanded-state generator/oracle
@dataclass(frozen=True)
class SemiMarkovSpec:
    emission:ThreeStateSpec
    duration_means:tuple=(3.0,5.0,7.0)
    switch_matrix:tuple=((0,.7,.3),(.55,0,.45),(.35,.65,0))

def duration_pmf(mean,T=16):
    # peaked discrete distribution with support 1..T, non-geometric
    d=np.arange(1,T+1); sigma=max(1.0,mean*.28); p=np.exp(-.5*((d-mean)/sigma)**2); return p/p.sum()
def hazards(mean,T=16):
    p=duration_pmf(mean,T); surv=np.flip(np.cumsum(np.flip(p))); return np.clip(p/surv,0,1)
def sim_semimarkov(s,thetas,seed0):
    rng=np.random.default_rng(seed0); B=len(thetas); T=s.emission.n_batches; out=np.empty((B,T),int); pi=pi3(s.emission); M=np.array(s.switch_matrix,float); h=rng.choice(3,B,p=pi); age=np.ones(B,int); hz=np.stack([hazards(m,T) for m in s.duration_means])
    for t in range(T):
        ps=np.array([obs3(float(th),int(hh),s.emission) for th,hh in zip(thetas,h)]); out[:,t]=rng.binomial(s.emission.shots_per_batch,ps)
        if t<T-1:
            leave=rng.random(B)<hz[h,np.minimum(age,T)-1]
            for b in np.where(leave)[0]: h[b]=rng.choice(3,p=M[h[b]])
            age=np.where(leave,1,age+1)
    return torch.tensor(out,dtype=torch.long,device=DEVICE)
def semimarkov_ll_grid(seqs,s):
    B,T=seqs.shape; Q=len(THETA_GRID); E=torch.tensor(np.stack([emissions3(float(th),s.emission) for th in THETA_GRID]),dtype=DTYPE,device=DEVICE)
    ll=base.log_binom_torch(seqs[:,:,None,None],s.emission.shots_per_batch,E[None,None,:,:])
    hz=torch.tensor(np.stack([hazards(m,T) for m in s.duration_means]),dtype=DTYPE,device=DEVICE); M=torch.tensor(np.array(s.switch_matrix),dtype=DTYPE,device=DEVICE); pi=torch.tensor(pi3(s.emission),dtype=DTYPE,device=DEVICE)
    alpha=torch.full((B,Q,3,T),-torch.inf,dtype=DTYPE,device=DEVICE); alpha[:,:,:,0]=torch.log(pi)[None,None,:]+ll[:,0,:,:]
    for t in range(1,T):
        new=torch.full_like(alpha,-torch.inf)
        # stay and age
        for h in range(3):
            for a in range(1,t+1):
                lp=alpha[:,:,h,a-1]+torch.log(torch.clamp(1-hz[h,a-1],1e-300))
                new[:,:,h,a]=torch.logaddexp(new[:,:,h,a],lp)
        # leave to other state at age 1
        leave_by_h=[]
        for hp in range(3):
            vals=[]
            for a in range(1,t+1): vals.append(alpha[:,:,hp,a-1]+torch.log(torch.clamp(hz[hp,a-1],1e-300)))
            leave_by_h.append(torch.logsumexp(torch.stack(vals,-1),-1))
        leave_by_h=torch.stack(leave_by_h,-1)
        for h in range(3): new[:,:,h,0]=torch.logsumexp(leave_by_h+torch.log(torch.clamp(M[:,h],1e-300))[None,None,:],-1)
        new=new+ll[:,t,:, :,None]
        alpha=new
    return torch.logsumexp(alpha.flatten(2),-1)

def run_external(global_threshold):
    rows=[]
    # independent persistent 3-state sessions, now represented in candidate set
    specs3=sobol_three(1 if SMOKE else 4,seed('true3'))
    for kind,specs in [('three_state',specs3)]:
        for i,s in enumerate(specs):
            idx=i; cache=OUT/'cache'/f'external_{kind}_{i}.pkl'
            if cache.exists(): row=pickle.loads(cache.read_bytes()); rows.append(row); print(kind,i+1,'resumed'); continue
            tc=make_cal_thetas(N_CAL); sc=sim3(s,tc,seed('cal3',i)); tt=make_test_thetas(PER_STRAT_TEST,seed('theta3',i)); st=sim3(s,tt,seed('test3',i)); truth=torch.tensor(tt>=.5,dtype=torch.long,device=DEVICE)
            inf=infer_session(sc,tc,st); oo=oracle_three(st,s,'ordered'); ou=oracle_three(st,s,'unordered')
            bad=inf['test_score']>global_threshold; gated=inf['stack_o'].clone(); gated[torch.tensor(bad,device=DEVICE)]=inf['stack_u'][torch.tensor(bad,device=DEVICE)]
            row={'kind':kind,'session':i,'oracle_ordered_error':metrics(oo,truth)['error'],'oracle_unordered_error':metrics(ou,truth)['error'],'bma_error':metrics(inf['bma_o'],truth)['error'],'stack_error':metrics(inf['stack_o'],truth)['error'],'gated_error':metrics(gated,truth)['error'],'stack_brier':metrics(inf['stack_o'],truth)['brier'],'stack_confidence':metrics(inf['stack_o'],truth)['confidence'],'adequacy_flag_rate':float(bad.mean()),'stack_gain':metrics(inf['stack_u'],truth)['error']-metrics(inf['stack_o'],truth)['error'],'oracle_gain':metrics(ou,truth)['error']-metrics(oo,truth)['error']}
            cache.write_bytes(pickle.dumps(row)); rows.append(row); print(kind,i+1,'done',flush=True)
    # semi-Markov out-of-family sessions
    semis=[]
    U=qmc.Sobol(3,scramble=True,seed=seed('semi')).random_base2(0 if SMOKE else 2)[:(1 if SMOKE else 4)]
    bases=sobol_three(1 if SMOKE else 4,seed('semibase'))
    for i,(u,e) in enumerate(zip(U,bases)): semis.append(SemiMarkovSpec(e,(2.5+1.5*u[0],4.0+2.5*u[1],6.0+3.0*u[2])))
    for i,s in enumerate(semis):
        cache=OUT/'cache'/f'external_semimarkov_{i}.pkl'
        if cache.exists(): row=pickle.loads(cache.read_bytes()); rows.append(row); print('semimarkov',i+1,'resumed'); continue
        tc=make_cal_thetas(N_CAL); sc=sim_semimarkov(s,tc,seed('calsemi',i)); tt=make_test_thetas(PER_STRAT_TEST,seed('thetasemi',i)); st=sim_semimarkov(s,tt,seed('testsemi',i)); truth=torch.tensor(tt>=.5,dtype=torch.long,device=DEVICE)
        inf=infer_session(sc,tc,st); logt=semimarkov_ll_grid(st,s); oo=torch.softmax(class_and_total(logt)[0],-1)
        bad=inf['test_score']>global_threshold; mask=torch.tensor(bad,device=DEVICE); gated=inf['stack_o'].clone(); gated[mask]=inf['stack_u'][mask]
        abst=~mask
        row={'kind':'semi_markov','session':i,'oracle_ordered_error':metrics(oo,truth)['error'],'bma_error':metrics(inf['bma_o'],truth)['error'],'stack_error':metrics(inf['stack_o'],truth)['error'],'gated_error':metrics(gated,truth)['error'],'stack_brier':metrics(inf['stack_o'],truth)['brier'],'stack_confidence':metrics(inf['stack_o'],truth)['confidence'],'adequacy_flag_rate':float(bad.mean()),'adequacy_coverage':float(abst.float().mean()),'adequacy_selective_error':float((inf['stack_o'].argmax(-1)[abst]!=truth[abst]).double().mean()) if int(abst.sum()) else np.nan}
        cache.write_bytes(pickle.dumps(row)); rows.append(row); print('semimarkov',i+1,'done',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/'external_robustness_sessions.csv',index=False); return df

def bootstrap(v,B=5000,seed0=1):
    v=np.asarray(v,float); rng=np.random.default_rng(seed0); m=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(B)]); return [float(v.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def summarize(inf,ext,record_paths,global_threshold):
    dyn=inf[inf.true_family.isin(ACTIVE)].copy()
    def pooled(num,den): return float(num.sum()/den.sum()) if den.sum()>1e-12 else None
    proc=[]
    for f,g in dyn.groupby('true_family'):
        proc.append({'family':f,'sessions':len(g),'oracle_gain':float(g.oracle_gain.mean()),'bma_gain':float(g.bma_gain.mean()),'stack_gain':float(g.stack_gain.mean()),'bma_recovery':pooled(g.bma_gain,g.oracle_gain),'stack_recovery':pooled(g.stack_gain,g.oracle_gain)})
    summary={'status':'post-hoc robust extension evaluated on new independent simulation streams','device':str(DEVICE),'mode':args.mode,'in_family_sessions':len(inf),'global_adequacy_threshold_per_batch_nll':float(global_threshold),'process':proc,
      'pooled_dynamic_bma_recovery':pooled(dyn.bma_gain,dyn.oracle_gain),'pooled_dynamic_stack_recovery':pooled(dyn.stack_gain,dyn.oracle_gain),
      'bma_ordered_error':bootstrap(inf.bma_ordered_error),'stack_ordered_error':bootstrap(inf.stack_ordered_error),'bma_brier':bootstrap(inf.bma_brier),'stack_brier':bootstrap(inf.stack_brier),
      'external':ext.groupby('kind').mean(numeric_only=True).reset_index().to_dict(orient='records')}
    (OUT/'LOCKQ_V25_FINAL_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame(proc).to_csv(OUT/'v25_process_recovery.csv',index=False)
    return summary

def main():
    t=time.time(); inf,paths=run_infamily()
    # global adequacy threshold from calibration-only out-of-fold scores
    scores=[]
    for p in paths: scores.extend(pickle.loads(p.read_bytes())['cv_score'])
    threshold=float(np.quantile(scores,.95)); ext=run_external(threshold); summary=summarize(inf,ext,paths,threshold)
    print(json.dumps(summary,indent=2)); print('elapsed minutes',(time.time()-t)/60)

if __name__=='__main__': main()