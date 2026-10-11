#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import gc
import importlib.util
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import beta, qmc, t as student_t

# -----------------------------------------------------------------------------
# Run control. The protocol below is written before any v2.6 test stream exists.
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["smoke", "full"], default="full")
parser.add_argument("--out", default="/mnt/data/LockQ_v26_2_FinalStack")
parser.add_argument("--skip-targeted", action="store_true")
parser.add_argument("--worker-kind", choices=["infamily", "three", "semi", "targeted"], default=None)
parser.add_argument("--worker-index", type=int, default=None)
parser.add_argument("--summarize-only", action="store_true")
args = parser.parse_args()
SMOKE = args.mode == "smoke"
OUT = Path(args.out)
(OUT / "cache").mkdir(parents=True, exist_ok=True)

# Import the validated v2.5 likelihood implementation as a library only.
_saved_argv = sys.argv[:]
sys.argv = ["drift_models.py", "--mode", "full", "--out", str(OUT / "_v25_import")]
spec = importlib.util.spec_from_file_location("v25lib", str(Path(__file__).resolve().parent / "drift_models.py"))
v25 = importlib.util.module_from_spec(spec)
sys.modules["v25lib"] = v25
assert spec.loader is not None
spec.loader.exec_module(v25)
sys.argv = _saved_argv

DEVICE = v25.DEVICE
DTYPE = v25.DTYPE
# CPU stability: the inherited multi-thread pool deadlocks after repeated session calls.
torch.set_grad_enabled(False)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
base = v25.base
SessionSpec = base.SessionSpec
import __main__
__main__.SessionSpec = SessionSpec

SEED = 20260728
FAMILIES = list(v25.FAMILIES)
EXCHANGEABLE = ["latent_static", "iid_mode_mixture"]
DYNAMIC = ["persistent_markov", "abrupt_single_change", "smooth_drift", "persistent_markov_3state"]
ORIGINAL_DYNAMIC = ["persistent_markov", "abrupt_single_change", "smooth_drift"]
MARGINS = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.38)]
BOUNDARY_CUTOFF = 0.10

# Freeze numerical sizes. Smoke keeps every branch but uses smaller samples.
THETA_GRID = np.linspace(0.12, 0.88, 21 if SMOKE else 41)
v25.THETA_GRID = THETA_GRID
v25.CLASS_MASK = torch.tensor(THETA_GRID >= 0.5, device=DEVICE)
v25.THETA_LOGPRIOR = torch.full(
    (len(THETA_GRID),), -math.log(len(THETA_GRID)), dtype=DTYPE, device=DEVICE
)

N_BASE_BINARY = 4 if SMOKE else 12
N_BASE_THREE = 3 if SMOKE else 8
N_SESS_PER_FAMILY = 1 if SMOKE else 4
N_CAL = 40 if SMOKE else 320
PER_STRAT_TEST = 8 if SMOKE else 50
N_FOLDS = 2 if SMOKE else 4
PARAMETER_MASS = 0.999
MIN_SUPPORT = {
    "latent_static": 2,
    "iid_mode_mixture": 2,
    "persistent_markov": 3,
    "abrupt_single_change": 4 if SMOKE else 5,
    "smooth_drift": 3,
    "persistent_markov_3state": 2,
}
LAMBDA_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
ACTION_G_GRID = [0.25, 0.50, 0.75]
ACTION_CONF_GRID = [0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
RISK_TARGET = 0.10
RISK_ALPHA = 0.05

PROTOCOL = {
    "name": "LockQ v2.6.2 frozen six-family predictive stack",
    "status": "frozen before v2.6 full test-stream generation; smoke-only runtime amendments recorded",
    "runtime": "single PyTorch CPU thread to avoid a reproduced repeated-call thread-pool deadlock; scientific calculations unchanged",
    "seed": SEED,
    "primary_objective": "ordered reference-decision log loss and error",
    "secondary_diagnostic": "ordered-versus-exact-unordered gain and oracle-gap recovery",
    "calibration_design": "equal symmetric allocation at two Gauss--Legendre nodes in each of the five pre-existing reference-margin strata",
    "candidate_families": FAMILIES,
            "family_support": "all structural families retained; only within-family parameter support is reduced by posterior mass",
    "combination": "six-family predictive stacking optimized by cross-fitted reference-decision log score",
        "lambda_selection": "nested calibration-only folds with one-standard-error preference for stronger regularization",
    "margin_adaptation": "two preregistered regimes at estimated reference margin 0.10",
    "parameter_support": {"posterior_mass": PARAMETER_MASS, "minimum": MIN_SUPPORT},
    "abrupt_change": "exact sum over every possible change location within every retained candidate; uncertainty over centre and spread retained by posterior mass",
        "risk_controller": "simultaneously calibrated ordered/unordered/abstain policy",
    "risk_target": RISK_TARGET,
    "risk_alpha": RISK_ALPHA,
    "lambda_grid": LAMBDA_GRID,
    "action_dynamic_probability_grid": ACTION_G_GRID,
    "action_confidence_grid": ACTION_CONF_GRID,
    "boundary_cutoff": BOUNDARY_CUTOFF,
    "sizes": {
        "mode": args.mode,
        "n_base_binary": N_BASE_BINARY,
        "n_base_three": N_BASE_THREE,
        "n_sessions_per_family": N_SESS_PER_FAMILY,
        "n_calibration_records": N_CAL,
        "test_records_per_margin_stratum": PER_STRAT_TEST,
    },
    "adoption_criteria": {
        "targeted_pooled_recovery_minimum": 0.40,
        "targeted_abrupt_recovery_minimum": 0.40,
        "exchangeable_absolute_gain_maximum": 0.002,
        "ordered_error_must_not_exceed_best_baseline": True,
        "external_selective_risk_target": RISK_TARGET,
    },
}
(OUT / "LOCKQ_V26_FROZEN_PROTOCOL.json").write_text(json.dumps(PROTOCOL, indent=2))


def seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("::".join(map(str, parts)).encode()).digest()[:8], "little"
    ) % (2**63 - 1)


def softmax_np(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    a = a - np.max(a)
    e = np.exp(a)
    return e / e.sum()


def logsumexp_np(a: np.ndarray, axis=None) -> np.ndarray:
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    z = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(z, axis=axis)


def make_cal_thetas(n: int, seed0: int) -> np.ndarray:
    """Balanced symmetric calibration at two Gauss--Legendre nodes per margin stratum."""
    del seed0  # deterministic quadrature support; randomness is only in measurement streams.
    n_support = 4 * len(MARGINS)  # two interior nodes, mirrored across the decision boundary.
    if n % n_support != 0:
        raise ValueError(f"N_CAL must be divisible by {n_support}")
    repeats = n // n_support
    node_lo = 0.5 - 1.0 / (2.0 * math.sqrt(3.0))
    node_hi = 0.5 + 1.0 / (2.0 * math.sqrt(3.0))
    support: List[float] = []
    for lo, hi in MARGINS:
        for u in (node_lo, node_hi):
            margin = lo + u * (hi - lo)
            support.extend((0.5 - margin, 0.5 + margin))
    return np.repeat(np.asarray(support, float), repeats)


def make_test_thetas(per: int, seed0: int) -> np.ndarray:
    rng = np.random.default_rng(seed0)
    vals: List[float] = []
    for lo, hi in MARGINS:
        m = rng.uniform(lo, hi, per)
        side = rng.integers(0, 2, per)
        vals.extend(0.5 + np.where(side, m, -m))
    return np.asarray(vals, float)


def stratified_folds(theta: np.ndarray, k: int, seed0: int) -> np.ndarray:
    """Stratify jointly by decision side and registered margin stratum."""
    margin = np.abs(theta - 0.5)
    bins = np.array([next(i for i, (lo, hi) in enumerate(MARGINS) if lo - 1e-12 <= m <= hi + 1e-12) for m in margin])
    side = (theta >= 0.5).astype(int)
    folds = np.empty(len(theta), int)
    rng = np.random.default_rng(seed0)
    for b in range(len(MARGINS)):
        for s in (0, 1):
            idx = np.where((bins == b) & (side == s))[0]
            rng.shuffle(idx)
            folds[idx] = np.arange(len(idx)) % k
    return folds


def candidate_library() -> Dict[str, list]:
    lib = {
        f: v25.sobol_binary(N_BASE_BINARY, seed("v26_candidates", f), f)
        for f in v25.BINARY_FAMILIES
    }
    lib["persistent_markov_3state"] = v25.sobol_three(
        N_BASE_THREE, seed("v26_candidates", "three")
    )
    return lib


BASE_LIBRARY = candidate_library()


def refine_abrupt_library(seq_cal: torch.Tensor, theta_cal: np.ndarray, initial: Sequence[SessionSpec]) -> list:
    """Calibration-only refinement; every candidate still sums exactly over all 15 locations."""
    K0 = torch.stack([v25.known_ll(seq_cal, theta_cal, s) for s in initial], 1)
    parent = initial[int(torch.argmax(K0.sum(0)).item())]
    centres = np.linspace(1 / 16, 15 / 16, 15)
    spreads = [0.06, 0.15] if SMOKE else [0.05, 0.12, 0.25, 0.50]
    return [
        replace(parent, change_center=float(c), change_spread=float(sp))
        for c in centres
        for sp in spreads
    ]


def retain_support(log_evidence: torch.Tensor, minimum: int) -> np.ndarray:
    """Smallest support reaching 99.9% posterior mass, with a frozen minimum."""
    p = torch.softmax(log_evidence, 0)
    order = torch.argsort(p, descending=True)
    cumulative = torch.cumsum(p[order], 0)
    crossing = torch.where(cumulative >= PARAMETER_MASS)[0]
    n_mass = int(crossing[0].item()) + 1 if len(crossing) else len(order)
    n = min(len(order), max(minimum, n_mass))
    return order[:n].detach().cpu().numpy()


def mix_family(cand_class: torch.Tensor, cand_total: torch.Tensor, logw: torch.Tensor):
    return v25.mix_family(cand_class, cand_total, logw)


def fit_regularized_stack(P: np.ndarray, y: np.ndarray, anchor: np.ndarray, lam: float) -> np.ndarray:
    """Proper log-score stacking with KL anchoring to calibration evidence."""
    P = np.asarray(P, float)
    y = np.asarray(y, int)
    anchor = np.clip(np.asarray(anchor, float), 1e-10, 1)
    anchor = anchor / anchor.sum()
    target = np.clip(P[np.arange(len(y)), :, y], 1e-12, 1)

    def objective(a: np.ndarray) -> float:
        w = softmax_np(a)
        nll = -np.log(np.clip(target @ w, 1e-12, 1)).mean()
        kl = float(np.sum(w * (np.log(np.clip(w, 1e-12, 1)) - np.log(anchor))))
        return nll + lam * kl

    start = np.log(anchor)
    res = minimize(objective, start, method="BFGS", options={"maxiter": 500, "gtol": 1e-9})
    if not res.success or not np.all(np.isfinite(res.x)):
        return anchor
    return softmax_np(res.x)


def select_lambda(P: np.ndarray, y: np.ndarray, anchor: np.ndarray, folds: np.ndarray) -> Tuple[float, np.ndarray, dict]:
    """Nested calibration selection with a one-standard-error regularization rule."""
    unique = np.unique(folds)
    losses = {lam: [] for lam in LAMBDA_GRID}
    for lam in LAMBDA_GRID:
        for fold in unique:
            tr = folds != fold
            va = folds == fold
            if tr.sum() < P.shape[1] + 2 or va.sum() == 0:
                continue
            w = fit_regularized_stack(P[tr], y[tr], anchor, lam)
            chosen = np.clip((P[va] * w[None, :, None]).sum(1)[np.arange(va.sum()), y[va]], 1e-12, 1)
            losses[lam].append(float(-np.log(chosen).mean()))
    means = {lam: float(np.mean(v)) if v else math.inf for lam, v in losses.items()}
    ses = {lam: float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0 for lam, v in losses.items()}
    best = min(LAMBDA_GRID, key=lambda z: means[z])
    tolerance = means[best] + ses[best]
    eligible = [lam for lam in LAMBDA_GRID if means[lam] <= tolerance]
    selected = max(eligible) if eligible else best
    final = fit_regularized_stack(P, y, anchor, selected)
    return selected, final, {"mean_loss": means, "se_loss": ses, "minimum_lambda": best}


def apply_weights(family_probs: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    w = torch.tensor(weights, dtype=DTYPE, device=DEVICE)
    return (family_probs * w[None, :, None]).sum(1)


def cp_upper(k: int, n: int, alpha: float) -> float:
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def choose_action_policy(
    p_ordered: np.ndarray,
    p_unordered: np.ndarray,
    y: np.ndarray,
    dynamic_probability: float,
) -> dict:
    policies = [(g, c) for g in ACTION_G_GRID for c in ACTION_CONF_GRID]
    alpha_each = RISK_ALPHA / len(policies)
    evaluated = []
    for g, conf in policies:
        co = p_ordered.max(1)
        cu = p_unordered.max(1)
        use_o = (dynamic_probability >= g) & (co >= conf)
        use_u = (~use_o) & (cu >= conf)
        issued = use_o | use_u
        pred = np.where(use_o, p_ordered.argmax(1), p_unordered.argmax(1))
        n = int(issued.sum())
        k = int(((pred != y) & issued).sum())
        upper = cp_upper(k, n, alpha_each)
        evaluated.append(
            {
                "dynamic_threshold": g,
                "confidence_threshold": conf,
                "coverage": n / len(y),
                "errors": k,
                "issued": n,
                "selective_risk": k / n if n else math.nan,
                "upper": upper,
                "ordered_fraction": float(use_o.sum() / n) if n else 0.0,
                "feasible": upper <= RISK_TARGET,
            }
        )
    feasible = [r for r in evaluated if r["feasible"]]
    if not feasible:
        return {"feasible": False, "coverage": 0.0, "evaluated": evaluated}
    selected = sorted(
        feasible,
        key=lambda r: (-r["coverage"], r["selective_risk"], r["ordered_fraction"], -r["confidence_threshold"]),
    )[0]
    selected = dict(selected)
    selected["evaluated"] = evaluated
    return selected


def apply_action_policy(p_o: torch.Tensor, p_u: torch.Tensor, dynamic_probability: float, policy: dict, truth: torch.Tensor) -> dict:
    if not policy.get("feasible", False):
        return {
            "coverage": 0.0,
            "selective_risk": math.nan,
            "ordered_fraction": 0.0,
            "issued": 0,
            "errors": 0,
        }
    co = p_o.max(-1).values
    cu = p_u.max(-1).values
    use_o = (dynamic_probability >= policy["dynamic_threshold"]) & (co >= policy["confidence_threshold"])
    use_u = (~use_o) & (cu >= policy["confidence_threshold"])
    issued = use_o | use_u
    pred = torch.where(use_o, p_o.argmax(-1), p_u.argmax(-1))
    n = int(issued.sum().item())
    k = int(((pred != truth) & issued).sum().item())
    return {
        "coverage": n / len(truth),
        "selective_risk": k / n if n else math.nan,
        "ordered_fraction": float(use_o.sum().item() / n) if n else 0.0,
        "issued": n,
        "errors": k,
    }


def infer_session(seq_cal: torch.Tensor, theta_cal: np.ndarray, seq_test: torch.Tensor) -> dict:
    ycal = (theta_cal >= 0.5).astype(int)
    folds = stratified_folds(theta_cal, N_FOLDS, seed("outer_folds", len(theta_cal)))
    session_lib = {f: list(c) for f,c in BASE_LIBRARY.items()}
    session_lib["abrupt_single_change"] = refine_abrupt_library(seq_cal, theta_cal, session_lib["abrupt_single_change"])
    family_oof_o = np.zeros((len(theta_cal),len(FAMILIES),2),float)
    family_oof_u = np.zeros_like(family_oof_o)
    family_test_o=[]; family_test_u=[]; scores=[]; retained_counts={}
    for fi,family in enumerate(FAMILIES):
        cands=session_lib[family]
        known_all=torch.stack([v25.known_ll(seq_cal,theta_cal,s) for s in cands],1)
        keep=retain_support(known_all.sum(0),MIN_SUPPORT[family]); retained=[cands[int(i)] for i in keep]
        retained_counts[family]=len(retained); known=known_all[:,keep]
        cvscore=0.0
        co,ct=v25.candidate_predictive(seq_cal,retained,'ordered')
        if family in EXCHANGEABLE: cu,cut=co,ct
        else: cu,cut=v25.candidate_predictive(seq_cal,retained,'unordered')
        for fold in range(N_FOLDS):
            tr=folds!=fold; va=folds==fold
            lw_all=torch.log_softmax(known_all[tr].sum(0),0)
            cvscore += float(torch.logsumexp(known_all[va]+lw_all[None,:],1).sum().item())
            lw=torch.log_softmax(known[tr].sum(0),0)
            po,_=mix_family(co[va],ct[va],lw); pu,_=mix_family(cu[va],cut[va],lw)
            family_oof_o[va,fi]=po.cpu().numpy(); family_oof_u[va,fi]=pu.cpu().numpy()
        scores.append(cvscore)
        lw=torch.log_softmax(known.sum(0),0)
        to,tot=v25.candidate_predictive(seq_test,retained,'ordered')
        if family in EXCHANGEABLE: tu,tut=to,tot
        else: tu,tut=v25.candidate_predictive(seq_test,retained,'unordered')
        po,_=mix_family(to,tot,lw); pu,_=mix_family(tu,tut,lw)
        family_test_o.append(po); family_test_u.append(pu)
    family_test_o=torch.stack(family_test_o,1); family_test_u=torch.stack(family_test_u,1)
    scores=np.asarray(scores)
    # Original-family evidence comparator.
    bmaw=softmax_np(scores[:5]); bma5_o=apply_weights(family_test_o[:,:5],bmaw); bma5_u=apply_weights(family_test_u[:,:5],bmaw)
    # Frozen primary estimator: unregularized proper-score stacking across all six families.
    final_w=fit_regularized_stack(family_oof_o,ycal,np.full(len(FAMILIES),1/len(FAMILIES)),0.0)
    v26_o=apply_weights(family_test_o,final_w); v26_u=apply_weights(family_test_u,final_w)
    stack_o=v26_o.clone(); stack_u=v26_u.clone()
    # Fully cross-fitted meta predictions for finite-sample action calibration.
    oof_o=np.zeros((len(ycal),2)); oof_u=np.zeros_like(oof_o)
    for fold in range(N_FOLDS):
        tr=folds!=fold; va=folds==fold
        w=fit_regularized_stack(family_oof_o[tr],ycal[tr],np.full(len(FAMILIES),1/len(FAMILIES)),0.0)
        oof_o[va]=(family_oof_o[va]*w[None,:,None]).sum(1)
        oof_u[va]=(family_oof_u[va]*w[None,:,None]).sum(1)
    policy=choose_action_policy(oof_o,oof_u,ycal,1.0)
    return {
        'bma5_o':bma5_o,'bma5_u':bma5_u,'stack_o':stack_o,'stack_u':stack_u,
        'v26_o':v26_o,'v26_u':v26_u,'oof_v26_o':oof_o,'oof_v26_u':oof_u,
        'group_weights':np.array([0.0,1.0]),'stack_weights':final_w,'regime_weights':{},
        'lambda_records':{},'retained_counts':retained_counts,'active_families':FAMILIES,
        'cv_known_score':{f:float(v) for f,v in zip(FAMILIES,scores)},'action_policy':policy,
        'predicted_boundary_rate':float((torch.abs(v26_o[:,1]-.5)<=BOUNDARY_CUTOFF).double().mean().item()),
        'order_enabled':True,'order_gate_mean':math.nan,'order_gate_lower':math.nan,
        'effective_dynamic_probability':1.0,
    }

def metrics(p: torch.Tensor, truth: torch.Tensor) -> dict:
    return v25.metrics(p, truth)


def oracle_binary(seq: torch.Tensor, s: SessionSpec, rep: str) -> torch.Tensor:
    return v25.oracle_binary(seq, s, rep)


def oracle_three(seq: torch.Tensor, s: v25.ThreeStateSpec, rep: str) -> torch.Tensor:
    return v25.oracle_three(seq, s, rep)


def result_row(prefix: dict, inf: dict, truth: torch.Tensor, oo: torch.Tensor, ou: torch.Tensor) -> dict:
    result = dict(prefix)
    mo, mu = metrics(oo, truth), metrics(ou, truth)
    result.update(
        oracle_ordered_error=mo["error"],
        oracle_unordered_error=mu["error"],
        oracle_gain=mu["error"] - mo["error"],
    )
    for name in ("bma5", "stack", "v26"):
        ao = metrics(inf[f"{name}_o"], truth)
        au = metrics(inf[f"{name}_u"], truth)
        result.update(
            {
                f"{name}_ordered_error": ao["error"],
                f"{name}_unordered_error": au["error"],
                f"{name}_gain": au["error"] - ao["error"],
                f"{name}_brier": ao["brier"],
                f"{name}_confidence": ao["confidence"],
            }
        )
    action = apply_action_policy(
        inf["v26_o"], inf["v26_u"], float(inf["effective_dynamic_probability"]), inf["action_policy"], truth
    )
    result.update({f"action_{k}": v for k, v in action.items()})
    result.update(
        dynamic_group_probability=float(inf["group_weights"][1]),
        predicted_boundary_rate=inf["predicted_boundary_rate"],
        action_policy_feasible=bool(inf["action_policy"].get("feasible", False)),
        action_calibration_coverage=float(inf["action_policy"].get("coverage", 0.0)),
        action_calibration_upper=float(inf["action_policy"].get("upper", 1.0)),
        order_enabled=bool(inf["order_enabled"]),
        order_gate_mean=float(inf["order_gate_mean"]),
        order_gate_lower=float(inf["order_gate_lower"]),
    )
    for f, n in inf["retained_counts"].items():
        result[f"retained_{f}"] = int(n)
    return result


def true_specs_infamily():
    out = []
    for f in v25.BINARY_FAMILIES:
        specs = v25.sobol_binary(N_SESS_PER_FAMILY, seed("v26_true", f), f)
        out.extend((f, i, s) for i, s in enumerate(specs))
    return out


def run_infamily() -> pd.DataFrame:
    rows = []
    specs = true_specs_infamily()
    for idx, (family, j, s) in enumerate(specs):
        path = OUT / "cache" / f"infamily_{idx}.pkl"
        if path.exists():
            rows.append(pickle.loads(path.read_bytes()))
            print(f"in-family {idx + 1}/{len(specs)} resumed", flush=True)
            continue
        tc = make_cal_thetas(N_CAL, seed("v26_cal_theta", idx))
        sc = base.simulate_sequences(s, tc, f"v26_cal_{idx}", shared_seed=seed("v26_cal", idx))
        tt = make_test_thetas(PER_STRAT_TEST, seed("v26_test_theta", idx))
        st = base.simulate_sequences(s, tt, f"v26_test_{idx}", shared_seed=seed("v26_test", idx))
        truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
        inf = infer_session(sc, tc, st)
        row = result_row(
            {"session": idx, "true_family": family, "within_family_index": j, "n_test": len(tt)},
            inf,
            truth,
            oracle_binary(st, s, "ordered"),
            oracle_binary(st, s, "unordered"),
        )
        path.write_bytes(pickle.dumps(row))
        rows.append(row)
        del inf, sc, st, truth
        gc.collect()
        print(f"in-family {idx + 1}/{len(specs)} {family} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v26_independent_infamily_sessions.csv", index=False)
    return df


def run_external() -> pd.DataFrame:
    rows = []
    n = 1 if SMOKE else 4
    three_specs = v25.sobol_three(n, seed("v26_external_three"))
    for i, s in enumerate(three_specs):
        path = OUT / "cache" / f"external_three_{i}.pkl"
        if path.exists():
            rows.append(pickle.loads(path.read_bytes()))
            print(f"three-state {i + 1}/{n} resumed", flush=True)
            continue
        tc = make_cal_thetas(N_CAL, seed("v26_three_cal_theta", i))
        sc = v25.sim3(s, tc, seed("v26_three_cal", i))
        tt = make_test_thetas(PER_STRAT_TEST, seed("v26_three_test_theta", i))
        st = v25.sim3(s, tt, seed("v26_three_test", i))
        truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
        inf = infer_session(sc, tc, st)
        row = result_row(
            {"kind": "three_state", "session": i, "n_test": len(tt)},
            inf,
            truth,
            oracle_three(st, s, "ordered"),
            oracle_three(st, s, "unordered"),
        )
        path.write_bytes(pickle.dumps(row))
        rows.append(row)
        del inf, sc, st, truth
        gc.collect()
        print(f"three-state {i + 1}/{n} done", flush=True)

    U = qmc.Sobol(3, scramble=True, seed=seed("v26_semi_duration")).random_base2(0 if SMOKE else 2)[:n]
    bases = v25.sobol_three(n, seed("v26_semi_emission"))
    semis = [
        v25.SemiMarkovSpec(e, (2.5 + 1.5 * u[0], 4.0 + 2.5 * u[1], 6.0 + 3.0 * u[2]))
        for u, e in zip(U, bases)
    ]
    for i, s in enumerate(semis):
        path = OUT / "cache" / f"external_semi_{i}.pkl"
        if path.exists():
            rows.append(pickle.loads(path.read_bytes()))
            print(f"semi-Markov {i + 1}/{n} resumed", flush=True)
            continue
        tc = make_cal_thetas(N_CAL, seed("v26_semi_cal_theta", i))
        sc = v25.sim_semimarkov(s, tc, seed("v26_semi_cal", i))
        tt = make_test_thetas(PER_STRAT_TEST, seed("v26_semi_test_theta", i))
        st = v25.sim_semimarkov(s, tt, seed("v26_semi_test", i))
        truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
        inf = infer_session(sc, tc, st)
        oo = torch.softmax(v25.class_and_total(v25.semimarkov_ll_grid(st, s))[0], -1)
        # No tractable complete-unordered semi-Markov oracle is claimed. Use the ordered oracle in both
        # fields only to keep the result table rectangular; oracle_gain is explicitly set to NaN.
        row = result_row(
            {"kind": "semi_markov", "session": i, "n_test": len(tt)}, inf, truth, oo, oo
        )
        row["oracle_unordered_error"] = math.nan
        row["oracle_gain"] = math.nan
        path.write_bytes(pickle.dumps(row))
        rows.append(row)
        del inf, sc, st, truth, oo
        gc.collect()
        print(f"semi-Markov {i + 1}/{n} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v26_external_robustness_sessions.csv", index=False)
    return df


def load_registered_dynamic_specs():
    files = list((v25.ROOT / "cache").glob("session_calibration_*_a4429d907cbb.pkl"))
    by = {}
    for p in files:
        m = re.match(r"session_calibration_(\d+)_.*_a4429d907cbb\.pkl", p.name)
        if m:
            by[int(m.group(1))] = p
    out = []
    for i in range(8, 20):
        obj = pickle.loads(by[i].read_bytes())
        out.append((i, obj["true_spec"]))
    return out


def run_targeted() -> pd.DataFrame:
    rows = []
    specs = load_registered_dynamic_specs()
    per = 12 if SMOKE else 80  # 60 smoke / 400 full records per session.
    for k, (old_i, s) in enumerate(specs):
        if SMOKE and k >= 3:
            break
        path = OUT / "cache" / f"targeted_{old_i}.pkl"
        if path.exists():
            rows.append(pickle.loads(path.read_bytes()))
            print(f"targeted {old_i} resumed", flush=True)
            continue
        tc = make_cal_thetas(N_CAL, seed("v26_target_cal_theta", old_i))
        sc = base.simulate_sequences(s, tc, f"v26_target_cal_{old_i}", shared_seed=seed("v26_target_cal", old_i))
        tt = make_test_thetas(per, seed("v26_target_test_theta", old_i))
        st = base.simulate_sequences(s, tt, f"v26_target_test_{old_i}", shared_seed=seed("v26_target_test", old_i))
        truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
        inf = infer_session(sc, tc, st)
        row = result_row(
            {"old_session": old_i, "true_family": s.family, "n_test": len(tt)},
            inf,
            truth,
            oracle_binary(st, s, "ordered"),
            oracle_binary(st, s, "unordered"),
        )
        path.write_bytes(pickle.dumps(row))
        rows.append(row)
        del inf, sc, st, truth
        gc.collect()
        print(f"targeted {old_i} {s.family} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v26_targeted_registered_dynamic_sessions.csv", index=False)
    return df


def bootstrap(values: Sequence[float], B: int = 20000, seed0: int = SEED) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": math.nan, "low": math.nan, "high": math.nan, "n": 0}
    if SMOKE:
        B = min(B, 2000)
    rng = np.random.default_rng(seed0)
    # Vectorized session bootstrap; identical estimator without Python-level resampling loops.
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boot = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "low": float(np.quantile(boot, 0.025)),
        "high": float(np.quantile(boot, 0.975)),
        "n": int(len(x)),
    }


def pooled_recovery(df: pd.DataFrame, method: str) -> float:
    den = float(df["oracle_gain"].sum())
    return float(df[f"{method}_gain"].sum() / den) if den > 1e-12 else math.nan


def process_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, g in df.groupby("true_family"):
        row = {
            "family": family,
            "sessions": len(g),
            "oracle_gain": g.oracle_gain.mean(),
        }
        for m in ("bma5", "stack", "v26"):
            row[f"{m}_error"] = g[f"{m}_ordered_error"].mean()
            row[f"{m}_gain"] = g[f"{m}_gain"].mean()
            row[f"{m}_recovery"] = pooled_recovery(g, m)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(inf: pd.DataFrame, ext: pd.DataFrame, targeted: pd.DataFrame) -> dict:
    dynamic_inf = inf[inf.true_family.isin(ORIGINAL_DYNAMIC)]
    exchangeable_inf = inf[inf.true_family.isin(EXCHANGEABLE)]
    target_process = process_summary(targeted) if len(targeted) else pd.DataFrame()
    if len(target_process):
        target_process.to_csv(OUT / "v26_targeted_process_summary.csv", index=False)
    process_summary(inf).to_csv(OUT / "v26_independent_process_summary.csv", index=False)

    abrupt = targeted[targeted.true_family == "abrupt_single_change"] if len(targeted) else targeted
    summary = {
        "status": "independent v2.6 evaluation completed",
        "device": str(DEVICE),
        "mode": args.mode,
        "in_family_sessions": int(len(inf)),
        "targeted_sessions": int(len(targeted)),
        "external_sessions": int(len(ext)),
        "ordered_error": {
            m: bootstrap(inf[f"{m}_ordered_error"], seed0=seed("boot", m))
            for m in ("bma5", "stack", "v26")
        },
        "brier": {
            m: bootstrap(inf[f"{m}_brier"], seed0=seed("boot_brier", m))
            for m in ("bma5", "stack", "v26")
        },
        "paired_error_improvement": {
            "bma5_minus_v26": bootstrap(inf.bma5_ordered_error - inf.v26_ordered_error, seed0=seed("bma5_v26")),
            "stack_minus_v26": bootstrap(inf.stack_ordered_error - inf.v26_ordered_error, seed0=seed("stack_v26")),
        },
        "independent_dynamic_recovery": {
            m: pooled_recovery(dynamic_inf, m) for m in ("bma5", "stack", "v26")
        },
        "targeted_dynamic_recovery": {
            m: pooled_recovery(targeted, m) if len(targeted) else math.nan
            for m in ("bma5", "stack", "v26")
        },
        "targeted_abrupt_recovery": {
            m: pooled_recovery(abrupt, m) if len(abrupt) else math.nan
            for m in ("bma5", "stack", "v26")
        },
        "exchangeable_gain": {
            m: float(exchangeable_inf[f"{m}_gain"].mean())
            for m in ("bma5", "stack", "v26")
        },
        "action": {
            "in_family_feasible_sessions": int(inf.action_policy_feasible.sum()),
            "mean_coverage": float(inf.action_coverage.mean()),
            "mean_selective_risk": float(inf.action_selective_risk.mean(skipna=True)),
            "external_by_kind": ext.groupby("kind")[["action_coverage", "action_selective_risk", "action_policy_feasible"]].mean(numeric_only=True).reset_index().to_dict(orient="records"),
        },
        "external_by_kind": ext.groupby("kind")[[
            "oracle_ordered_error", "bma5_ordered_error", "stack_ordered_error", "v26_ordered_error",
            "bma5_brier", "stack_brier", "v26_brier", "v26_confidence"
        ]].mean(numeric_only=True).reset_index().to_dict(orient="records"),
    }

    best_baseline_error = min(summary["ordered_error"]["bma5"]["mean"], summary["ordered_error"]["stack"]["mean"])
    criteria = {
        "targeted_pooled_recovery": summary["targeted_dynamic_recovery"]["v26"] >= 0.40,
        "targeted_abrupt_recovery": summary["targeted_abrupt_recovery"]["v26"] >= 0.40,
        "exchangeable_null": abs(summary["exchangeable_gain"]["v26"]) <= 0.002,
        "ordered_error": summary["ordered_error"]["v26"]["mean"] <= best_baseline_error,
    }
    criteria["all_core"] = all(criteria.values())
    summary["adoption_criteria"] = criteria
    (OUT / "LOCKQ_V26_FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=True))

    lines = [
        "# LockQ v2.6 Principled Hybrid: Adjudication",
        "",
        f"Core adoption verdict: **{'PASS' if criteria['all_core'] else 'DO NOT REPLACE v2.4'}**.",
        "",
        "## Frozen criteria",
    ]
    for k, v in criteria.items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Headline metrics",
        f"- Mean ordered error: BMA5 {summary['ordered_error']['bma5']['mean']:.4f}, pure stack {summary['ordered_error']['stack']['mean']:.4f}, v2.6 {summary['ordered_error']['v26']['mean']:.4f}.",
        f"- Targeted pooled recovery: BMA5 {summary['targeted_dynamic_recovery']['bma5']:.3f}, stack {summary['targeted_dynamic_recovery']['stack']:.3f}, v2.6 {summary['targeted_dynamic_recovery']['v26']:.3f}.",
        f"- Targeted abrupt recovery: BMA5 {summary['targeted_abrupt_recovery']['bma5']:.3f}, stack {summary['targeted_abrupt_recovery']['stack']:.3f}, v2.6 {summary['targeted_abrupt_recovery']['v26']:.3f}.",
        f"- Exchangeable v2.6 gain: {summary['exchangeable_gain']['v26']:.4f}.",
        "",
        "The paper should be updated only if the frozen core criteria pass. Otherwise v2.6 remains a method-development result.",
    ]
    (OUT / "LOCKQ_V26_ADJUDICATION.md").write_text("\n".join(lines) + "\n")
    return summary


def _worker_infamily(idx: int) -> None:
    specs = true_specs_infamily()
    family, j, s = specs[idx]
    path = OUT / "cache" / f"infamily_{idx}.pkl"
    if path.exists():
        return
    tc = make_cal_thetas(N_CAL, seed("v26_cal_theta", idx))
    sc = base.simulate_sequences(s, tc, f"v26_cal_{idx}", shared_seed=seed("v26_cal", idx))
    tt = make_test_thetas(PER_STRAT_TEST, seed("v26_test_theta", idx))
    st = base.simulate_sequences(s, tt, f"v26_test_{idx}", shared_seed=seed("v26_test", idx))
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sc, tc, st)
    row = result_row(
        {"session": idx, "true_family": family, "within_family_index": j, "n_test": len(tt)},
        inf, truth, oracle_binary(st, s, "ordered"), oracle_binary(st, s, "unordered")
    )
    path.write_bytes(pickle.dumps(row))


def _external_specs():
    n = 1 if SMOKE else 4
    three_specs = v25.sobol_three(n, seed("v26_external_three"))
    U = qmc.Sobol(3, scramble=True, seed=seed("v26_semi_duration")).random_base2(0 if SMOKE else 2)[:n]
    bases = v25.sobol_three(n, seed("v26_semi_emission"))
    semis = [v25.SemiMarkovSpec(e, (2.5 + 1.5*u[0], 4.0 + 2.5*u[1], 6.0 + 3.0*u[2])) for u,e in zip(U,bases)]
    return three_specs, semis


def _worker_three(i: int) -> None:
    s = _external_specs()[0][i]
    path = OUT / "cache" / f"external_three_{i}.pkl"
    if path.exists(): return
    tc = make_cal_thetas(N_CAL, seed("v26_three_cal_theta", i))
    sc = v25.sim3(s, tc, seed("v26_three_cal", i))
    tt = make_test_thetas(PER_STRAT_TEST, seed("v26_three_test_theta", i))
    st = v25.sim3(s, tt, seed("v26_three_test", i))
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sc, tc, st)
    row = result_row({"kind":"three_state","session":i,"n_test":len(tt)}, inf, truth,
                     oracle_three(st,s,"ordered"), oracle_three(st,s,"unordered"))
    path.write_bytes(pickle.dumps(row))


def _worker_semi(i: int) -> None:
    s = _external_specs()[1][i]
    path = OUT / "cache" / f"external_semi_{i}.pkl"
    if path.exists(): return
    tc = make_cal_thetas(N_CAL, seed("v26_semi_cal_theta", i))
    sc = v25.sim_semimarkov(s, tc, seed("v26_semi_cal", i))
    tt = make_test_thetas(PER_STRAT_TEST, seed("v26_semi_test_theta", i))
    st = v25.sim_semimarkov(s, tt, seed("v26_semi_test", i))
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sc, tc, st)
    oo = torch.softmax(v25.class_and_total(v25.semimarkov_ll_grid(st, s))[0], -1)
    row = result_row({"kind":"semi_markov","session":i,"n_test":len(tt)}, inf, truth, oo, oo)
    row["oracle_unordered_error"] = math.nan
    row["oracle_gain"] = math.nan
    path.write_bytes(pickle.dumps(row))


def _worker_targeted(k: int) -> None:
    old_i, s = load_registered_dynamic_specs()[k]
    path = OUT / "cache" / f"targeted_{old_i}.pkl"
    if path.exists(): return
    per = 12 if SMOKE else 80
    tc = make_cal_thetas(N_CAL, seed("v26_target_cal_theta", old_i))
    sc = base.simulate_sequences(s, tc, f"v26_target_cal_{old_i}", shared_seed=seed("v26_target_cal", old_i))
    tt = make_test_thetas(per, seed("v26_target_test_theta", old_i))
    st = base.simulate_sequences(s, tt, f"v26_target_test_{old_i}", shared_seed=seed("v26_target_test", old_i))
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sc, tc, st)
    row = result_row({"old_session":old_i,"true_family":s.family,"n_test":len(tt)}, inf, truth,
                     oracle_binary(st,s,"ordered"), oracle_binary(st,s,"unordered"))
    path.write_bytes(pickle.dumps(row))


def _run_worker(kind: str, index: int) -> None:
    if kind == "infamily": _worker_infamily(index)
    elif kind == "three": _worker_three(index)
    elif kind == "semi": _worker_semi(index)
    elif kind == "targeted": _worker_targeted(index)
    else: raise ValueError(kind)


def _spawn(kind: str, index: int) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", args.mode, "--out", str(OUT),
           "--worker-kind", kind, "--worker-index", str(index)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _collect_rows(pattern: str) -> list:
    return [pickle.loads(p.read_bytes()) for p in sorted((OUT/"cache").glob(pattern))]


def orchestrate() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    n_inf = len(true_specs_infamily())
    for i in range(n_inf):
        _spawn("infamily", i)
        print(f"in-family {i+1}/{n_inf} done", flush=True)
    inf = pd.DataFrame(_collect_rows("infamily_*.pkl")).sort_values("session")
    inf.to_csv(OUT/"v26_independent_infamily_sessions.csv", index=False)

    n_ext = 1 if SMOKE else 4
    for i in range(n_ext):
        _spawn("three", i); print(f"three-state {i+1}/{n_ext} done", flush=True)
    for i in range(n_ext):
        _spawn("semi", i); print(f"semi-Markov {i+1}/{n_ext} done", flush=True)
    ext = pd.DataFrame(_collect_rows("external_three_*.pkl") + _collect_rows("external_semi_*.pkl"))
    ext.to_csv(OUT/"v26_external_robustness_sessions.csv", index=False)

    targeted = pd.DataFrame()
    if not args.skip_targeted:
        n_target = 3 if SMOKE else 12
        for i in range(n_target):
            _spawn("targeted", i); print(f"targeted {i+1}/{n_target} done", flush=True)
        targeted = pd.DataFrame(_collect_rows("targeted_*.pkl")).sort_values("old_session")
        targeted.to_csv(OUT/"v26_targeted_registered_dynamic_sessions.csv", index=False)
    return inf, ext, targeted


def main():
    if args.worker_kind is not None:
        if args.worker_index is None: raise ValueError("--worker-index is required")
        _run_worker(args.worker_kind, args.worker_index)
        return
    if args.summarize_only:
        inf = pd.DataFrame(_collect_rows("infamily_*.pkl")).sort_values("session")
        ext = pd.DataFrame(_collect_rows("external_three_*.pkl") + _collect_rows("external_semi_*.pkl"))
        targeted_rows = _collect_rows("targeted_*.pkl")
        targeted = pd.DataFrame(targeted_rows).sort_values("old_session") if targeted_rows else pd.DataFrame()
        inf.to_csv(OUT/"v26_independent_infamily_sessions.csv", index=False)
        ext.to_csv(OUT/"v26_external_robustness_sessions.csv", index=False)
        if len(targeted): targeted.to_csv(OUT/"v26_targeted_registered_dynamic_sessions.csv", index=False)
        summary = summarize(inf, ext, targeted)
        print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
        return
    start = time.time()
    print(json.dumps(PROTOCOL, indent=2), flush=True)
    inf, ext, targeted = orchestrate()
    summary = summarize(inf, ext, targeted)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"elapsed minutes: {(time.time()-start)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()