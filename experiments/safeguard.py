#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import beta, binom, qmc

# -----------------------------------------------------------------------------
# Run control
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["smoke", "full"], default="full")
parser.add_argument("--out", default="/mnt/data/LockQ_v29_SessionCertified")
parser.add_argument("--skip-targeted", action="store_true")
parser.add_argument("--worker-kind", choices=["infamily", "three", "semi", "targeted"], default=None)
parser.add_argument("--worker-index", type=int, default=None)
parser.add_argument("--summarize-only", action="store_true")
parser.add_argument("--fit-records", type=int, default=None)
parser.add_argument("--risk-records", type=int, default=None)
parser.add_argument("--permutations", type=int, default=None)
parser.add_argument("--perm-record-chunk", type=int, default=4)
parser.add_argument("--perm-block", type=int, default=64)
args = parser.parse_args()

SMOKE = args.mode == "smoke"
OUT = Path(args.out)
(OUT / "cache").mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

# Import the validated exact-likelihood library without running its main.
_saved_argv = sys.argv[:]
sys.argv = ["drift_models.py", "--mode", args.mode, "--out", str(OUT / "_v25_import")]
spec = importlib.util.spec_from_file_location("v25lib", str(HERE / "drift_models.py"))
v25 = importlib.util.module_from_spec(spec)
sys.modules["v25lib"] = v25
assert spec.loader is not None
spec.loader.exec_module(v25)
sys.argv = _saved_argv

DEVICE = v25.DEVICE
DTYPE = v25.DTYPE
base = v25.base
SessionSpec = base.SessionSpec
import __main__
__main__.SessionSpec = SessionSpec

torch.set_grad_enabled(False)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

SEED = 20260730
FAMILIES = list(v25.FAMILIES)
EXCHANGEABLE = ["latent_static", "iid_mode_mixture"]
DYNAMIC = ["persistent_markov", "abrupt_single_change", "smooth_drift", "persistent_markov_3state"]
ORIGINAL_DYNAMIC = ["persistent_markov", "abrupt_single_change", "smooth_drift"]
MARGINS = [(0.005, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.38)]
CONF_GRID = [0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
RISK_TARGET = 0.10
RISK_ALPHA = 0.05
NULL_TOLERANCE = 0.002

N_FIT = args.fit_records if args.fit_records is not None else (40 if SMOKE else 80)
N_RISK = args.risk_records if args.risk_records is not None else (40 if SMOKE else 80)
N_PERM = args.permutations if args.permutations is not None else (19 if SMOKE else 499)
PER_STRAT_TEST = 4 if SMOKE else 50
N_FOLDS = 2 if SMOKE else 4
N_BASE_BINARY = 4 if SMOKE else 12
N_BASE_THREE = 3 if SMOKE else 8
N_SESS_PER_FAMILY = 1 if SMOKE else 4
PARAMETER_MASS = 0.999
MIN_SUPPORT = {
    "latent_static": 2,
    "iid_mode_mixture": 2,
    "persistent_markov": 3,
    "abrupt_single_change": 4 if SMOKE else 5,
    "smooth_drift": 3,
    "persistent_markov_3state": 2,
}

if N_FIT % (4 * len(MARGINS)) != 0 or N_RISK % (4 * len(MARGINS)) != 0:
    raise ValueError("fit and risk record counts must be divisible by 20")
if N_PERM < 1:
    raise ValueError("--permutations must be positive")

# The rank guarantee is 1/(K+1). Full mode freezes K=499, exactly 0.002.
NULL_RANK_BOUND = 1.0 / (N_PERM + 1)

PROTOCOL = {
    "name": "LockQ v2.9 certified null-safe stack",
    "status": "frozen implementation; partial diagnostics completed, full confirmatory streams required before manuscript replacement",
    "seed": SEED,
    "observations": {"batches": 16, "shots_per_batch": 128, "shots_per_record": 2048},
    "fit_calibration_records": N_FIT,
    "independent_risk_calibration_records": N_RISK,
    "total_calibration_records": N_FIT + N_RISK,
    "total_calibration_shots": (N_FIT + N_RISK) * 2048,
    "candidate_families": FAMILIES,
    "combination": "hierarchical four-fold predictive stacking: all six families when exchangeability is not rejected; the four time-dependent families after an exact session certificate",
    "session_exchangeability_certificate": {
        "permutations": N_PERM,
        "test": "multiset-preserving omnibus randomization test over adjacent variation, lag-one dependence, CUSUM change, and linear trend",
        "rank_false_activation_bound": NULL_RANK_BOUND,
        "rule": "if temporal order is not certified, ordered probabilities are retained only where the deterministic posterior-discrepancy bound proves the class decision cannot change; exact unordered inference is mandatory otherwise",
        "frozen_tolerance": NULL_TOLERANCE,
    },
    "posterior_discrepancy_certificate": {
        "bound": "sum_r w_r ||q_r^o-q_r^u||_1",
        "decision_condition": "ordered and unordered class decisions are guaranteed equal when the unordered top-two gap exceeds the bound",
        "fallback": "in sessions without the exchangeability certificate, exact unordered inference is used whenever the bound reaches the unordered top-two gap; ordered probabilities are retained only in certified decision-agreement cases",
    },
    "selective_controller": {
        "threshold_grid": CONF_GRID,
        "selection": "Holm step-down testing of the eight unique confidence policies on an independent risk split",
        "certification": "exact one-sided binomial tests control family-wise error without the 16 duplicated dynamic-threshold tests",
        "risk_target": RISK_TARGET,
        "alpha": RISK_ALPHA,
    },
    "adoption_checks": {
        "exchangeable_absolute_gain_maximum": NULL_TOLERANCE,
        "abrupt_recovery_must_match_or_exceed_previous_80_record_result": 0.625,
        "feasible_sessions_must_exceed_previous_80_record_result": 10,
        "mean_coverage_must_exceed_previous_80_record_result": 0.4036,
        "empirical_selective_risk_maximum": RISK_TARGET,
    },
}
(OUT / "LOCKQ_V29_FROZEN_PROTOCOL.json").write_text(json.dumps(PROTOCOL, indent=2))


def seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("::".join(map(str, parts)).encode()).digest()[:8], "little"
    ) % (2**63 - 1)


def softmax_np(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    a = a - np.max(a)
    e = np.exp(a)
    return e / e.sum()


def make_cal_thetas(n: int) -> np.ndarray:
    """Balanced symmetric quadrature support at two nodes in each margin stratum."""
    node_lo = 0.5 - 1.0 / (2.0 * math.sqrt(3.0))
    node_hi = 0.5 + 1.0 / (2.0 * math.sqrt(3.0))
    support: List[float] = []
    for lo, hi in MARGINS:
        for u in (node_lo, node_hi):
            margin = lo + u * (hi - lo)
            support.extend((0.5 - margin, 0.5 + margin))
    return np.repeat(np.asarray(support, float), n // len(support))


def make_test_thetas(per: int, seed0: int) -> np.ndarray:
    rng = np.random.default_rng(seed0)
    vals: List[float] = []
    for lo, hi in MARGINS:
        margin = rng.uniform(lo, hi, per)
        side = rng.integers(0, 2, per)
        vals.extend(0.5 + np.where(side, margin, -margin))
    return np.asarray(vals, float)


def stratified_folds(theta: np.ndarray, k: int, seed0: int) -> np.ndarray:
    margin = np.abs(theta - 0.5)
    bins = np.array([
        next(i for i, (lo, hi) in enumerate(MARGINS) if lo - 1e-12 <= m <= hi + 1e-12)
        for m in margin
    ])
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
        f: v25.sobol_binary(N_BASE_BINARY, seed("v28_candidates", f), f)
        for f in v25.BINARY_FAMILIES
    }
    lib["persistent_markov_3state"] = v25.sobol_three(
        N_BASE_THREE, seed("v28_candidates", "three")
    )
    return lib


BASE_LIBRARY = candidate_library()


def refine_abrupt_library(
    seq_fit: torch.Tensor, theta_fit: np.ndarray, initial: Sequence[SessionSpec]
) -> list:
    K0 = torch.stack([v25.known_ll(seq_fit, theta_fit, s) for s in initial], 1)
    parent = initial[int(torch.argmax(K0.sum(0)).item())]
    centres = np.linspace(1 / 16, 15 / 16, 15)
    spreads = [0.06, 0.15] if SMOKE else [0.05, 0.12, 0.25, 0.50]
    return [
        replace(parent, change_center=float(c), change_spread=float(sp))
        for c in centres
        for sp in spreads
    ]


def retain_support(log_evidence: torch.Tensor, minimum: int) -> np.ndarray:
    p = torch.softmax(log_evidence, 0)
    order = torch.argsort(p, descending=True)
    cumulative = torch.cumsum(p[order], 0)
    crossing = torch.where(cumulative >= PARAMETER_MASS)[0]
    n_mass = int(crossing[0].item()) + 1 if len(crossing) else len(order)
    n = min(len(order), max(minimum, n_mass))
    return order[:n].detach().cpu().numpy()


def fit_stack(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    P = np.asarray(P, float)
    y = np.asarray(y, int)
    target = np.clip(P[np.arange(len(y)), :, y], 1e-12, 1)

    def objective(a: np.ndarray) -> float:
        w = softmax_np(a)
        return float(-np.log(np.clip(target @ w, 1e-12, 1)).mean())

    res = minimize(objective, np.zeros(P.shape[1]), method="BFGS", options={"maxiter": 500, "gtol": 1e-9})
    if not res.success or not np.all(np.isfinite(res.x)):
        return np.full(P.shape[1], 1.0 / P.shape[1])
    return softmax_np(res.x)


@dataclass
class FittedSession:
    retained: Dict[str, list]
    parameter_logweights: Dict[str, torch.Tensor]
    stack_weights: np.ndarray
    bma5_weights: np.ndarray
    retained_counts: Dict[str, int]


def fit_session_model(seq_fit: torch.Tensor, theta_fit: np.ndarray, namespace: str, certified_dynamic: bool) -> FittedSession:
    yfit = (theta_fit >= 0.5).astype(int)
    folds = stratified_folds(theta_fit, min(N_FOLDS, max(2, len(theta_fit) // 20)), seed("inner", namespace))
    session_lib = {f: list(c) for f, c in BASE_LIBRARY.items()}
    session_lib["abrupt_single_change"] = refine_abrupt_library(
        seq_fit, theta_fit, session_lib["abrupt_single_change"]
    )

    retained: Dict[str, list] = {}
    known: Dict[str, torch.Tensor] = {}
    candidate_fit_o: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    family_scores = []
    for family in FAMILIES:
        cands = session_lib[family]
        known_all = torch.stack([v25.known_ll(seq_fit, theta_fit, s) for s in cands], 1)
        keep = retain_support(known_all.sum(0), MIN_SUPPORT[family])
        retained[family] = [cands[int(i)] for i in keep]
        known[family] = known_all[:, keep]
        candidate_fit_o[family] = v25.candidate_predictive(seq_fit, retained[family], "ordered")
        cvscore = 0.0
        for fold in np.unique(folds):
            tr, va = folds != fold, folds == fold
            lw_all = torch.log_softmax(known_all[tr].sum(0), 0)
            cvscore += float(torch.logsumexp(known_all[va] + lw_all[None, :], 1).sum().item())
        family_scores.append(cvscore)

    family_oof = np.zeros((len(theta_fit), len(FAMILIES), 2), float)
    for fold in np.unique(folds):
        tr, va = folds != fold, folds == fold
        for fi, family in enumerate(FAMILIES):
            lw = torch.log_softmax(known[family][tr].sum(0), 0)
            pc, pt = candidate_fit_o[family]
            po, _ = v25.mix_family(pc[va], pt[va], lw)
            family_oof[va, fi] = po.cpu().numpy()

    if certified_dynamic:
        dynamic_idx = np.asarray([FAMILIES.index(f) for f in DYNAMIC], dtype=int)
        dynamic_weights = fit_stack(family_oof[:, dynamic_idx, :], yfit)
        stack_weights = np.zeros(len(FAMILIES), dtype=float)
        stack_weights[dynamic_idx] = dynamic_weights
    else:
        stack_weights = fit_stack(family_oof, yfit)
    parameter_logweights = {
        family: torch.log_softmax(known[family].sum(0), 0) for family in FAMILIES
    }
    return FittedSession(
        retained=retained,
        parameter_logweights=parameter_logweights,
        stack_weights=stack_weights,
        bma5_weights=softmax_np(np.asarray(family_scores[:5])),
        retained_counts={f: len(retained[f]) for f in FAMILIES},
    )


def predict_family(model: FittedSession, seqs: torch.Tensor, representation: str) -> torch.Tensor:
    family_probs = []
    for family in FAMILIES:
        if representation == "unordered" and family in EXCHANGEABLE:
            rep = "ordered"
        else:
            rep = representation
        cand_class, cand_total = v25.candidate_predictive(seqs, model.retained[family], rep)
        p, _ = v25.mix_family(
            cand_class, cand_total, model.parameter_logweights[family]
        )
        family_probs.append(p)
    return torch.stack(family_probs, 1)


def predict_raw(model: FittedSession, seqs: torch.Tensor) -> dict:
    fam_o = predict_family(model, seqs, "ordered")
    fam_u = predict_family(model, seqs, "unordered")
    sw = torch.tensor(model.stack_weights, dtype=DTYPE, device=DEVICE)
    bw = torch.tensor(model.bma5_weights, dtype=DTYPE, device=DEVICE)
    ordered = (fam_o * sw[None, :, None]).sum(1)
    unordered = (fam_u * sw[None, :, None]).sum(1)
    # Convexity gives an exact deterministic bound on the stacked posterior
    # discrepancy.  It is evaluated without labels and without Monte Carlo.
    family_l1 = torch.sum(torch.abs(fam_o - fam_u), dim=2)
    discrepancy_bound = (family_l1 * sw[None, :]).sum(1)
    return {
        "ordered": ordered,
        "unordered": unordered,
        "bma5_ordered": (fam_o[:, :5] * bw[None, :, None]).sum(1),
        "bma5_unordered": (fam_u[:, :5] * bw[None, :, None]).sum(1),
        "family_ordered": fam_o,
        "family_unordered": fam_u,
        "discrepancy_bound": discrepancy_bound,
    }


def _temporal_components(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixed dimensionless omnibus statistic for temporal organization.

    x has shape (R,N,T).  Four complementary components target persistent
    adjacency, abrupt changes, and smooth trends.  Each amplitude-dependent
    component is normalized by the within-record standard deviation, so equal
    weights do not privilege a measurement scale or reference-margin stratum.
    The scalar sum is frozen before outcome generation.
    """
    T = x.shape[-1]
    centered = x - x.mean(axis=-1, keepdims=True)
    scale = np.sqrt((centered ** 2).mean(axis=-1)) + 1e-6
    smoothness = -np.abs(np.diff(x, axis=-1)).mean(axis=-1) / scale
    numerator = (centered[..., :-1] * centered[..., 1:]).sum(axis=-1)
    denominator = np.sqrt(
        (centered[..., :-1] ** 2).sum(axis=-1)
        * (centered[..., 1:] ** 2).sum(axis=-1)
    ) + 1e-12
    lag_one = numerator / denominator
    cumulative = np.cumsum(x, axis=-1)
    tau = np.arange(1, T)
    left = cumulative[..., :-1] / tau
    right = (cumulative[..., -1:] - cumulative[..., :-1]) / (T - tau)
    cusum = np.abs(left - right).max(axis=-1) / scale
    time_axis = np.arange(T, dtype=float)
    time_axis = (time_axis - time_axis.mean()) / np.sqrt((time_axis ** 2).sum())
    trend = np.abs(
        (centered * time_axis).sum(axis=-1)
        / (np.sqrt((centered ** 2).sum(axis=-1)) + 1e-12)
    )
    per_record = np.stack([smoothness, lag_one, cusum, trend], axis=-1)
    aggregate_components = per_record.mean(axis=-2)
    omnibus = aggregate_components.sum(axis=-1)
    return omnibus, aggregate_components


def session_exchangeability_test(seqs: torch.Tensor, namespace: str) -> dict:
    """Multiset-preserving Monte Carlo randomization certificate.

    The test statistic is fixed and label-free. Under exchangeability, the
    observed order and every within-record random permutation have the same
    conditional distribution given the complete batch multisets. The standard
    plus-one rank p-value is therefore conservative, and a rejection at
    1/(K+1) bounds false activation by that amount.
    """
    observed = seqs.detach().cpu().numpy().astype(float) / 128.0
    n, T = observed.shape
    rng = np.random.default_rng(seed("session_perm", namespace))
    permutations = np.argsort(rng.random((N_PERM, n, T)), axis=-1)
    permuted = np.take_along_axis(observed[None, :, :], permutations, axis=2)
    orbit = np.concatenate([observed[None, :, :], permuted], axis=0)
    omnibus, components = _temporal_components(orbit)
    exceedances = int((omnibus[1:] >= omnibus[0] - 1e-15).sum())
    pvalue = float((1 + exceedances) / float(N_PERM + 1))
    certified = bool(pvalue <= NULL_RANK_BOUND + 1e-15)
    return {
        "certified_dynamic": certified,
        "pvalue": pvalue,
        "rank_bound": float(NULL_RANK_BOUND),
        "exceedances": exceedances,
        "omnibus_score": float(omnibus[0]),
        "component_names": ["standardized_negative_total_variation", "lag_one_dependence", "standardized_cusum_change", "absolute_linear_trend"],
        "component_scores": components[0].tolist(),
        "permutation_median": float(np.median(omnibus[1:])),
        "permutation_998_quantile": float(np.quantile(omnibus[1:], 0.998)),
    }


def null_safe_predict(model: FittedSession, seqs: torch.Tensor, session_gate: dict) -> dict:
    """Apply the calibrated session certificate and deterministic decision bound.

    If the session randomization test rejects exchangeability at rank 1/(K+1),
    ordered inference is enabled. Otherwise the convex posterior-discrepancy
    bound is compared with the exact-unordered top-two gap record by record.
    Ordered probabilities are retained only when the bound proves that they
    cannot change the exact-unordered class decision. Whenever the bound can
    change the decision, exact unordered inference is mandatory.

    Under an exchangeable session, the final class decision can therefore
    differ from exact unordered inference only if the session randomization
    certificate falsely activates, whose probability is at most 1/(K+1)=0.002.
    """
    raw = predict_raw(model, seqs)
    qo, qu = raw["ordered"], raw["unordered"]
    unordered_gap = torch.abs(qu[:, 1] - qu[:, 0])
    observed_discrepancy = torch.sum(torch.abs(qo - qu), dim=1)
    discrepancy_can_change = raw["discrepancy_bound"] >= unordered_gap - 1e-15
    certified = bool(session_gate["certified_dynamic"])
    if certified:
        use_order = torch.ones(len(seqs), dtype=torch.bool, device=DEVICE)
    else:
        use_order = ~discrepancy_can_change
    qns = torch.where(use_order[:, None], qo, qu)
    return {
        **raw,
        "null_safe": qns,
        "use_order": use_order,
        "unordered_gap": unordered_gap,
        "observed_discrepancy": observed_discrepancy,
        "decision_changed": qo.argmax(1) != qu.argmax(1),
        "discrepancy_can_change": discrepancy_can_change,
        "bound_safe_order_rate": float((~discrepancy_can_change).double().mean().item()),
        "session_gate": session_gate,
    }


def cp_upper(k: int, n: int, alpha: float = RISK_ALPHA) -> float:
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def evaluate_threshold(prob: np.ndarray, y: np.ndarray, gamma: float) -> dict:
    conf = prob.max(1)
    issued = conf >= gamma
    pred = prob.argmax(1)
    n = int(issued.sum())
    k = int(((pred != y) & issued).sum())
    pvalue = float(binom.cdf(k, n, RISK_TARGET)) if n > 0 else 1.0
    return {
        "confidence_threshold": float(gamma),
        "coverage": n / len(y),
        "issued": n,
        "errors": k,
        "selective_risk": k / n if n else math.nan,
        "upper": cp_upper(k, n),
        "pvalue": pvalue,
    }


def holm_certify_thresholds(prob: np.ndarray, y: np.ndarray) -> dict:
    """Certify the eight distinct nested policies with Holm FWER control."""
    evaluated = [evaluate_threshold(prob, y, g) for g in CONF_GRID]
    order = sorted(range(len(evaluated)), key=lambda i: evaluated[i]["pvalue"])
    rejected: list[int] = []
    running_adjusted = 0.0
    m = len(evaluated)
    for rank, idx in enumerate(order):
        raw = evaluated[idx]["pvalue"]
        critical = RISK_ALPHA / (m - rank)
        running_adjusted = max(running_adjusted, min(1.0, (m - rank) * raw))
        evaluated[idx]["holm_rank"] = rank + 1
        evaluated[idx]["holm_critical"] = critical
        evaluated[idx]["holm_adjusted_p"] = running_adjusted
        if raw <= critical:
            evaluated[idx]["holm_rejected_unsafe_null"] = True
            rejected.append(idx)
        else:
            evaluated[idx]["holm_rejected_unsafe_null"] = False
            break
    rejected_set = set(rejected)
    for idx, row in enumerate(evaluated):
        row.setdefault("holm_rank", None)
        row.setdefault("holm_critical", None)
        row.setdefault("holm_adjusted_p", 1.0)
        row.setdefault("holm_rejected_unsafe_null", idx in rejected_set)
    safe = [evaluated[i] for i in rejected]
    if not safe:
        return {"feasible": False, "coverage": 0.0, "evaluated": evaluated, "reason": "no_holm_rejection"}
    selected = sorted(
        safe,
        key=lambda r: (-r["coverage"], r["selective_risk"], -r["confidence_threshold"]),
    )[0]
    return {"feasible": True, **selected, "evaluated": evaluated, "reason": "holm_certified"}


def apply_policy(ns: dict, policy: dict, truth: torch.Tensor) -> dict:
    if not policy.get("feasible", False):
        return {
            "coverage": 0.0,
            "selective_risk": math.nan,
            "ordered_fraction": 0.0,
            "issued": 0,
            "errors": 0,
        }
    p = ns["null_safe"]
    conf = p.max(1).values
    issued = conf >= float(policy["confidence_threshold"])
    pred = p.argmax(1)
    n = int(issued.sum().item())
    k = int(((pred != truth) & issued).sum().item())
    ordered_issued = int((ns["use_order"] & issued).sum().item())
    return {
        "coverage": n / len(truth),
        "selective_risk": k / n if n else math.nan,
        "ordered_fraction": ordered_issued / n if n else 0.0,
        "issued": n,
        "errors": k,
    }



def infer_session(
    seq_fit: torch.Tensor,
    theta_fit: np.ndarray,
    seq_risk: torch.Tensor,
    theta_risk: np.ndarray,
    seq_test: torch.Tensor,
    namespace: str,
) -> dict:
    session_gate = session_exchangeability_test(seq_fit, f"{namespace}_fit")
    model = fit_session_model(
        seq_fit, theta_fit, f"{namespace}_fit",
        certified_dynamic=bool(session_gate["certified_dynamic"]),
    )
    risk_ns = null_safe_predict(model, seq_risk, session_gate)
    risk_policy = holm_certify_thresholds(
        risk_ns["null_safe"].cpu().numpy(),
        (theta_risk >= 0.5).astype(int),
    )
    test_ns = null_safe_predict(model, seq_test, session_gate)
    return {
        "model": model,
        "session_gate": session_gate,
        "risk_policy": risk_policy,
        "risk_ns": risk_ns,
        "test": test_ns,
    }


def metrics(p: torch.Tensor, truth: torch.Tensor) -> dict:
    return v25.metrics(p, truth)


def oracle_binary(seq: torch.Tensor, s: SessionSpec, rep: str) -> torch.Tensor:
    return v25.oracle_binary(seq, s, rep)


def oracle_three(seq: torch.Tensor, s: v25.ThreeStateSpec, rep: str) -> torch.Tensor:
    return v25.oracle_three(seq, s, rep)


def result_row(prefix: dict, inf: dict, truth: torch.Tensor, oo: torch.Tensor, ou: torch.Tensor) -> dict:
    pred = inf["test"]
    result = dict(prefix)
    mo, mu = metrics(oo, truth), metrics(ou, truth)
    result.update(
        oracle_ordered_error=mo["error"],
        oracle_unordered_error=mu["error"],
        oracle_gain=mu["error"] - mo["error"],
    )
    mapping = {
        "bma5": (pred["bma5_ordered"], pred["bma5_unordered"]),
        "raw_stack": (pred["ordered"], pred["unordered"]),
        "v29": (pred["null_safe"], pred["unordered"]),
    }
    for name, (po, pu) in mapping.items():
        a, u = metrics(po, truth), metrics(pu, truth)
        result.update({
            f"{name}_ordered_error": a["error"],
            f"{name}_unordered_error": u["error"],
            f"{name}_gain": u["error"] - a["error"],
            f"{name}_brier": a["brier"],
            f"{name}_confidence": a["confidence"],
        })
    action = apply_policy(pred, inf["risk_policy"], truth)
    result.update({f"action_{k}": v for k, v in action.items()})
    result.update(
        action_policy_feasible=bool(inf["risk_policy"].get("feasible", False)),
        selected_confidence_threshold=float(inf["risk_policy"].get("confidence_threshold", math.nan)),
        risk_calibration_coverage=float(inf["risk_policy"].get("coverage", 0.0)),
        risk_calibration_upper=float(inf["risk_policy"].get("upper", 1.0)),
        risk_holm_adjusted_p=float(inf["risk_policy"].get("holm_adjusted_p", 1.0)),
        order_use_rate=float(pred["use_order"].double().mean().item()),
        decision_change_rate=float(pred["decision_changed"].double().mean().item()),
        discrepancy_can_change_rate=float(pred["discrepancy_can_change"].double().mean().item()),
        mean_discrepancy_bound=float(pred["discrepancy_bound"].mean().item()),
        max_discrepancy_bound=float(pred["discrepancy_bound"].max().item()),
        mean_observed_discrepancy=float(pred["observed_discrepancy"].mean().item()),
        bound_safe_order_rate=float(pred["bound_safe_order_rate"]),
        session_dynamic_certified=bool(inf["session_gate"]["certified_dynamic"]),
        session_exchangeability_pvalue=float(inf["session_gate"]["pvalue"]),
        session_omnibus_score=float(inf["session_gate"]["omnibus_score"]),
        session_gate_exceedances=int(inf["session_gate"]["exceedances"]),
        fitted_dynamic_weight=float(sum(inf["model"].stack_weights[FAMILIES.index(f)] for f in DYNAMIC)),
        null_rank_bound=float(NULL_RANK_BOUND),
    )
    for f, n in inf["model"].retained_counts.items():
        result[f"retained_{f}"] = int(n)
    return result


def true_specs_infamily() -> list:
    out = []
    for f in v25.BINARY_FAMILIES:
        specs = v25.sobol_binary(N_SESS_PER_FAMILY, seed("v28_true", f), f)
        out.extend((f, i, s) for i, s in enumerate(specs))
    return out


REGISTERED_RANGES = {
    "persistence": [0.55, 0.995],
    "occupancy": [0.10, 0.90],
    "depolarizing_gap": [0.08, 0.28],
    "readout_gap": [0.04, 0.22],
    "readout_asymmetry": [-0.14, 0.14],
    "calibration_log_gap": [-0.75, 0.30],
    "drift_amplitude": [0.08, 0.55],
    "change_center": [0.20, 0.80],
    "change_spread": [0.05, 0.25],
}


def registered_unit_to_spec(u: np.ndarray, family: str) -> SessionSpec:
    def lin(name: str, j: int) -> float:
        a, b = REGISTERED_RANGES[name]
        return float(a + u[j] * (b - a))
    return SessionSpec(
        family=family,
        total_shots=2048,
        n_batches=16,
        shots_per_batch=128,
        persistence=lin("persistence", 0),
        occupancy=lin("occupancy", 1),
        depolarizing_gap=lin("depolarizing_gap", 2),
        readout_gap=lin("readout_gap", 3),
        readout_asymmetry=lin("readout_asymmetry", 4),
        calibration_log_gap=lin("calibration_log_gap", 5),
        drift_amplitude=lin("drift_amplitude", 6),
        change_center=lin("change_center", 7),
        change_spread=lin("change_spread", 8),
    )


def registered_dynamic_specs() -> list:
    out = []
    for fi, family in enumerate(["persistent_markov", "abrupt_single_change", "smooth_drift"], start=2):
        U = qmc.Sobol(9, scramble=True, seed=2026072210 + fi).random_base2(2)[:4]
        for j, u in enumerate(U):
            old_index = 8 + (fi - 2) * 4 + j
            out.append((old_index, registered_unit_to_spec(u, family)))
    return out


def generate_three_state_splits(s, namespace: str, test_per: int) -> tuple:
    tf, tr = make_cal_thetas(N_FIT), make_cal_thetas(N_RISK)
    sf = v25.sim3(s, tf, seed(namespace, "fit"))
    sr = v25.sim3(s, tr, seed(namespace, "risk"))
    tt = make_test_thetas(test_per, seed(namespace, "theta_test"))
    st = v25.sim3(s, tt, seed(namespace, "test"))
    return tf, sf, tr, sr, tt, st


def generate_binary_splits(s, namespace: str, test_per: int) -> tuple:
    tf, tr = make_cal_thetas(N_FIT), make_cal_thetas(N_RISK)
    sf = base.simulate_sequences(s, tf, namespace + "_fit", shared_seed=seed(namespace, "fit"))
    sr = base.simulate_sequences(s, tr, namespace + "_risk", shared_seed=seed(namespace, "risk"))
    tt = make_test_thetas(test_per, seed(namespace, "theta_test"))
    st = base.simulate_sequences(s, tt, namespace + "_test", shared_seed=seed(namespace, "test"))
    return tf, sf, tr, sr, tt, st


def _worker_infamily(idx: int) -> None:
    family, j, s = true_specs_infamily()[idx]
    path = OUT / "cache" / f"infamily_{idx}.pkl"
    if path.exists():
        return
    tf, sf, tr, sr, tt, st = generate_binary_splits(s, f"v28_infamily_{idx}", PER_STRAT_TEST)
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sf, tf, sr, tr, st, f"infamily_{idx}")
    row = result_row(
        {"session": idx, "true_family": family, "within_family_index": j, "n_test": len(tt)},
        inf,
        truth,
        oracle_binary(st, s, "ordered"),
        oracle_binary(st, s, "unordered"),
    )
    path.write_bytes(pickle.dumps(row))


def _external_specs() -> tuple[list, list]:
    n = 1 if SMOKE else 4
    three_specs = v25.sobol_three(n, seed("v28_external_three"))
    U = qmc.Sobol(3, scramble=True, seed=seed("v28_semi_duration")).random_base2(0 if SMOKE else 2)[:n]
    bases = v25.sobol_three(n, seed("v28_semi_emission"))
    semis = [
        v25.SemiMarkovSpec(e, (2.5 + 1.5 * u[0], 4.0 + 2.5 * u[1], 6.0 + 3.0 * u[2]))
        for u, e in zip(U, bases)
    ]
    return three_specs, semis


def _worker_three(i: int) -> None:
    s = _external_specs()[0][i]
    path = OUT / "cache" / f"external_three_{i}.pkl"
    if path.exists():
        return
    tf, sf, tr, sr, tt, st = generate_three_state_splits(s, f"v28_three_{i}", PER_STRAT_TEST)
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sf, tf, sr, tr, st, f"three_{i}")
    row = result_row(
        {"kind": "three_state", "session": i, "n_test": len(tt)},
        inf,
        truth,
        oracle_three(st, s, "ordered"),
        oracle_three(st, s, "unordered"),
    )
    path.write_bytes(pickle.dumps(row))


def _worker_semi(i: int) -> None:
    s = _external_specs()[1][i]
    path = OUT / "cache" / f"external_semi_{i}.pkl"
    if path.exists():
        return
    tf, tr = make_cal_thetas(N_FIT), make_cal_thetas(N_RISK)
    sf = v25.sim_semimarkov(s, tf, seed("v28_semi", i, "fit"))
    sr = v25.sim_semimarkov(s, tr, seed("v28_semi", i, "risk"))
    tt = make_test_thetas(PER_STRAT_TEST, seed("v28_semi", i, "theta_test"))
    st = v25.sim_semimarkov(s, tt, seed("v28_semi", i, "test"))
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sf, tf, sr, tr, st, f"semi_{i}")
    oo = torch.softmax(v25.class_and_total(v25.semimarkov_ll_grid(st, s))[0], -1)
    row = result_row({"kind": "semi_markov", "session": i, "n_test": len(tt)}, inf, truth, oo, oo)
    row["oracle_unordered_error"] = math.nan
    row["oracle_gain"] = math.nan
    path.write_bytes(pickle.dumps(row))


def _worker_targeted(k: int) -> None:
    old_i, s = registered_dynamic_specs()[k]
    path = OUT / "cache" / f"targeted_{old_i}.pkl"
    if path.exists():
        return
    per = 8 if SMOKE else 80
    tf, sf, tr, sr, tt, st = generate_binary_splits(s, f"v28_targeted_{old_i}", per)
    truth = torch.tensor(tt >= 0.5, dtype=torch.long, device=DEVICE)
    inf = infer_session(sf, tf, sr, tr, st, f"targeted_{old_i}")
    row = result_row(
        {"old_session": old_i, "true_family": s.family, "n_test": len(tt)},
        inf,
        truth,
        oracle_binary(st, s, "ordered"),
        oracle_binary(st, s, "unordered"),
    )
    path.write_bytes(pickle.dumps(row))


def bootstrap(values: Sequence[float], B: int = 20000, seed0: int = SEED) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": math.nan, "low": math.nan, "high": math.nan, "n": 0}
    if SMOKE:
        B = min(B, 2000)
    rng = np.random.default_rng(seed0)
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


def summarize(inf: pd.DataFrame, ext: pd.DataFrame, targeted: pd.DataFrame) -> dict:
    dynamic_inf = inf[inf.true_family.isin(ORIGINAL_DYNAMIC)]
    exchangeable_inf = inf[inf.true_family.isin(EXCHANGEABLE)]
    abrupt = targeted[targeted.true_family == "abrupt_single_change"] if len(targeted) else targeted
    action_risks = inf.action_selective_risk.dropna()
    summary = {
        "status": "LockQ v2.9 evaluation completed" if not SMOKE else "LockQ v2.9 smoke validation completed; not scientific evidence",
        "mode": args.mode,
        "device": str(DEVICE),
        "fit_records": N_FIT,
        "risk_records": N_RISK,
        "calibration_shots": (N_FIT + N_RISK) * 2048,
        "permutations": N_PERM,
        "null_rank_bound": NULL_RANK_BOUND,
        "in_family_sessions": int(len(inf)),
        "targeted_sessions": int(len(targeted)),
        "external_sessions": int(len(ext)),
        "ordered_error": {
            m: bootstrap(inf[f"{m}_ordered_error"], seed0=seed("error", m))
            for m in ("bma5", "raw_stack", "v29")
        },
        "brier": {
            m: bootstrap(inf[f"{m}_brier"], seed0=seed("brier", m))
            for m in ("bma5", "raw_stack", "v29")
        },
        "exchangeable_gain": {
            m: float(exchangeable_inf[f"{m}_gain"].mean())
            for m in ("bma5", "raw_stack", "v29")
        },
        "independent_dynamic_recovery": {
            m: pooled_recovery(dynamic_inf, m) for m in ("bma5", "raw_stack", "v29")
        },
        "targeted_dynamic_recovery": {
            m: pooled_recovery(targeted, m) if len(targeted) else math.nan
            for m in ("bma5", "raw_stack", "v29")
        },
        "targeted_abrupt_recovery": {
            m: pooled_recovery(abrupt, m) if len(abrupt) else math.nan
            for m in ("bma5", "raw_stack", "v29")
        },
        "action": {
            "feasible_sessions": int(inf.action_policy_feasible.sum()),
            "mean_coverage": float(inf.action_coverage.mean()),
            "mean_selective_risk": float(action_risks.mean()) if len(action_risks) else math.nan,
            "max_selective_risk": float(action_risks.max()) if len(action_risks) else math.nan,
        },
        "session_certificate": {
            "certified_sessions": int(inf.session_dynamic_certified.sum()),
            "certified_exchangeable_sessions": int(exchangeable_inf.session_dynamic_certified.sum()),
            "certified_dynamic_sessions": int(dynamic_inf.session_dynamic_certified.sum()),
            "mean_exchangeability_pvalue": float(inf.session_exchangeability_pvalue.mean()),
            "mean_order_use_rate": float(inf.order_use_rate.mean()),
        },
        "posterior_discrepancy": {
            "mean_decision_change_rate": float(inf.decision_change_rate.mean()),
            "mean_can_change_rate": float(inf.discrepancy_can_change_rate.mean()),
            "mean_bound": float(inf.mean_discrepancy_bound.mean()),
            "mean_observed_discrepancy": float(inf.mean_observed_discrepancy.mean()),
        },
        "external_by_kind": ext.groupby("kind")[[
            "oracle_ordered_error", "bma5_ordered_error", "raw_stack_ordered_error",
            "v29_ordered_error", "action_coverage", "action_selective_risk",
            "action_policy_feasible", "order_use_rate", "session_dynamic_certified", "session_exchangeability_pvalue"
        ]].mean(numeric_only=True).reset_index().to_dict(orient="records") if len(ext) else [],
    }
    criteria = {
        "exchangeable_null": abs(summary["exchangeable_gain"]["v29"]) <= NULL_TOLERANCE,
        "no_exchangeable_session_activation": summary["session_certificate"]["certified_exchangeable_sessions"] == 0,
        "abrupt_recovery_vs_old_80": (
            summary["targeted_abrupt_recovery"]["v29"] >= 0.625
            if len(targeted) else False
        ),
        "feasible_sessions_vs_old_80": summary["action"]["feasible_sessions"] > 10 if not SMOKE else None,
        "coverage_vs_old_80": summary["action"]["mean_coverage"] > 0.4036 if not SMOKE else None,
        "empirical_selective_risk": (
            summary["action"]["max_selective_risk"] <= RISK_TARGET
            if np.isfinite(summary["action"]["max_selective_risk"]) else False
        ),
    }
    criteria["all_core"] = all(v for v in criteria.values() if v is not None) if not SMOKE else None
    summary["adoption_criteria"] = criteria
    (OUT / "LOCKQ_V29_FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def _run_worker(kind: str, index: int) -> None:
    if kind == "infamily":
        _worker_infamily(index)
    elif kind == "three":
        _worker_three(index)
    elif kind == "semi":
        _worker_semi(index)
    elif kind == "targeted":
        _worker_targeted(index)
    else:
        raise ValueError(kind)


def _spawn(kind: str, index: int) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", args.mode,
        "--out", str(OUT),
        "--fit-records", str(N_FIT),
        "--risk-records", str(N_RISK),
        "--permutations", str(N_PERM),
        "--perm-record-chunk", str(args.perm_record_chunk),
        "--perm-block", str(args.perm_block),
        "--worker-kind", kind,
        "--worker-index", str(index),
    ]
    subprocess.run(cmd, check=True)


def _collect(pattern: str) -> list:
    return [pickle.loads(p.read_bytes()) for p in sorted((OUT / "cache").glob(pattern))]


def orchestrate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_inf = len(true_specs_infamily())
    for i in range(n_inf):
        _spawn("infamily", i)
        print(f"in-family {i + 1}/{n_inf} done", flush=True)
    inf = pd.DataFrame(_collect("infamily_*.pkl")).sort_values("session")
    inf.to_csv(OUT / "v29_independent_infamily_sessions.csv", index=False)

    n_ext = 1 if SMOKE else 4
    for i in range(n_ext):
        _spawn("three", i)
        print(f"three-state {i + 1}/{n_ext} done", flush=True)
    for i in range(n_ext):
        _spawn("semi", i)
        print(f"semi-Markov {i + 1}/{n_ext} done", flush=True)
    ext = pd.DataFrame(_collect("external_three_*.pkl") + _collect("external_semi_*.pkl"))
    ext.to_csv(OUT / "v29_external_robustness_sessions.csv", index=False)

    targeted = pd.DataFrame()
    if not args.skip_targeted:
        n_target = 3 if SMOKE else 12
        for i in range(n_target):
            _spawn("targeted", i)
            print(f"targeted {i + 1}/{n_target} done", flush=True)
        targeted = pd.DataFrame(_collect("targeted_*.pkl")).sort_values("old_session")
        targeted.to_csv(OUT / "v29_targeted_registered_dynamic_sessions.csv", index=False)
    return inf, ext, targeted


def main() -> None:
    if args.worker_kind is not None:
        if args.worker_index is None:
            raise ValueError("--worker-index is required")
        _run_worker(args.worker_kind, args.worker_index)
        return
    if args.summarize_only:
        inf = pd.DataFrame(_collect("infamily_*.pkl")).sort_values("session")
        ext = pd.DataFrame(_collect("external_three_*.pkl") + _collect("external_semi_*.pkl"))
        targeted_rows = _collect("targeted_*.pkl")
        targeted = pd.DataFrame(targeted_rows).sort_values("old_session") if targeted_rows else pd.DataFrame()
        summary = summarize(inf, ext, targeted)
        print(json.dumps(summary, indent=2, allow_nan=True))
        return
    start = time.time()
    print(json.dumps(PROTOCOL, indent=2), flush=True)
    inf, ext, targeted = orchestrate()
    summary = summarize(inf, ext, targeted)
    print(json.dumps(summary, indent=2, allow_nan=True), flush=True)
    print(f"elapsed minutes: {(time.time() - start) / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()