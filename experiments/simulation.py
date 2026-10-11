from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def deterministic_seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("::".join(map(str, parts)).encode()).digest()[:8], "little"
    ) % (2**63 - 1)


def stable_hash(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class SessionSpec:
    family: str
    total_shots: int
    n_batches: int
    shots_per_batch: int
    persistence: float
    occupancy: float
    depolarizing_gap: float
    readout_gap: float
    readout_asymmetry: float
    calibration_log_gap: float
    drift_amplitude: float = 0.0
    change_center: float = 0.5
    change_spread: float = 0.15


def spec_id(s: SessionSpec) -> str:
    return stable_hash(asdict(s))[:16]


def transition_matrix(spec: SessionSpec) -> np.ndarray:
    rho, pi1 = float(spec.persistence), float(spec.occupancy)
    a01 = (1 - rho) * pi1
    a10 = (1 - rho) * (1 - pi1)
    return np.array([[1 - a01, a01], [a10, 1 - a10]], float)


def hidden_prior(spec: SessionSpec) -> np.ndarray:
    return np.array([1 - spec.occupancy, spec.occupancy], float)


def observed_probability_one(theta: float, h: int, spec: SessionSpec) -> float:
    pr = float(np.clip(theta, 1e-12, 1 - 1e-12))
    dep = np.clip(0.01 + (spec.depolarizing_gap if h == 1 else 0.0), 0, 0.49)
    pr = (1 - dep) * pr + dep * 0.5
    lo = math.log(pr / (1 - pr))
    scale = math.exp(spec.calibration_log_gap if h == 1 else 0.0)
    pr = 1 / (1 + math.exp(-scale * lo))
    gap = spec.readout_gap if h == 1 else 0.0
    asym = spec.readout_asymmetry if h == 1 else 0.0
    e01 = np.clip(0.01 + gap + asym / 2, 0, 0.49)
    e10 = np.clip(0.01 + gap - asym / 2, 0, 0.49)
    return float(np.clip((1 - e10) * pr + e01 * (1 - pr), 1e-12, 1 - 1e-12))


def emissions_theta(theta: float, spec: SessionSpec) -> np.ndarray:
    return np.array([observed_probability_one(theta, h, spec) for h in (0, 1)], float)


def log_binom_torch(y: torch.Tensor, n: int, p: torch.Tensor) -> torch.Tensor:
    y = y.to(dtype=p.dtype)
    nn = torch.as_tensor(float(n), dtype=p.dtype, device=p.device)
    p = torch.clamp(p, 1e-300, 1 - 1e-15)
    return (
        torch.lgamma(nn + 1)
        - torch.lgamma(y + 1)
        - torch.lgamma(nn - y + 1)
        + y * torch.log(p)
        + (nn - y) * torch.log1p(-p)
    )


def transition_schedule(spec: SessionSpec) -> tuple[np.ndarray, np.ndarray]:
    T = spec.n_batches
    if spec.family == "iid_mode_mixture":
        A = np.array(
            [[1 - spec.occupancy, spec.occupancy], [1 - spec.occupancy, spec.occupancy]],
            float,
        )
        return np.repeat(A[None], T - 1, axis=0), hidden_prior(spec)
    if spec.family == "latent_static":
        return np.repeat(np.eye(2)[None], T - 1, axis=0), hidden_prior(spec)
    if spec.family == "smooth_drift":
        occ = np.clip(
            spec.occupancy + spec.drift_amplitude * np.linspace(-0.5, 0.5, T),
            0.02,
            0.98,
        )
        if T <= 1:
            return np.zeros((0, 2, 2), float), np.array([1 - occ[0], occ[0]], float)
        mats = []
        for t in range(T - 1):
            p1 = float(occ[t + 1])
            mats.append(
                np.array(
                    [
                        [1 - (1 - spec.persistence) * p1, (1 - spec.persistence) * p1],
                        [
                            (1 - spec.persistence) * (1 - p1),
                            1 - (1 - spec.persistence) * (1 - p1),
                        ],
                    ],
                    float,
                )
            )
        return np.stack(mats), np.array([1 - occ[0], occ[0]], float)
    A = transition_matrix(spec)
    return np.repeat(A[None], T - 1, axis=0), hidden_prior(spec)


def changepoint_weights(spec: SessionSpec) -> tuple[np.ndarray, np.ndarray]:
    T = spec.n_batches
    tau = np.arange(1, T)
    centre = spec.change_center * T
    spread = max(spec.change_spread * T, 0.5)
    w = np.exp(-0.5 * ((tau - centre) / spread) ** 2)
    return tau, w / w.sum()


def ordered_loglik_state_model(
    seqs: torch.Tensor, theta: float, spec: SessionSpec
) -> torch.Tensor:
    B, T = seqs.shape
    E = torch.tensor(emissions_theta(theta, spec), dtype=DTYPE, device=seqs.device)
    if spec.family == "abrupt_single_change":
        if T == 1:
            vals = []
            for h, wt in [(0, 1 - spec.occupancy), (1, spec.occupancy)]:
                ll = log_binom_torch(seqs[:, 0], spec.shots_per_batch, E[h])
                vals.append(ll + math.log(max(float(wt), 1e-300)))
            return torch.logsumexp(torch.stack(vals, -1), -1)
        tau_values, tau_weights = changepoint_weights(spec)
        vals = []
        for tau, wt in zip(tau_values, tau_weights):
            for direction, dw in [(0, 1 - spec.occupancy), (1, spec.occupancy)]:
                path = np.r_[
                    np.full(tau, direction, int), np.full(T - tau, 1 - direction, int)
                ]
                h = torch.tensor(path, dtype=torch.long, device=seqs.device)
                ll = log_binom_torch(
                    seqs, spec.shots_per_batch, E[h][None, :]
                ).sum(-1)
                vals.append(ll + math.log(max(float(wt * dw), 1e-300)))
        return torch.logsumexp(torch.stack(vals, -1), -1)

    As, pi0 = transition_schedule(spec)
    loga = torch.log(torch.tensor(pi0, dtype=DTYPE, device=seqs.device))[None, :] + log_binom_torch(
        seqs[:, 0, None], spec.shots_per_batch, E[None, :]
    )
    for t in range(1, T):
        A = torch.tensor(As[t - 1], dtype=DTYPE, device=seqs.device)
        loga = torch.logsumexp(
            loga.unsqueeze(-1) + torch.log(torch.clamp(A, 1e-300))[None, :, :],
            dim=-2,
        ) + log_binom_torch(seqs[:, t, None], spec.shots_per_batch, E[None, :])
    return torch.logsumexp(loga, -1)


def simulate_sequences(
    spec: SessionSpec,
    thetas: np.ndarray,
    namespace: str,
    shared_seed: int | None = None,
) -> torch.Tensor:
    thetas = np.asarray(thetas, float)
    B, T, s = len(thetas), spec.n_batches, spec.shots_per_batch
    seed = deterministic_seed(namespace, spec_id(spec), B) if shared_seed is None else int(shared_seed)
    g = torch.Generator(device=DEVICE)
    g.manual_seed(seed)
    out = torch.empty((B, T), dtype=torch.long, device=DEVICE)

    if spec.family == "abrupt_single_change":
        tau_values, tau_weights = changepoint_weights(spec)
        tau = torch.tensor(
            np.random.default_rng(seed).choice(tau_values, size=B, p=tau_weights),
            dtype=torch.long,
            device=DEVICE,
        )
        direction = (torch.rand(B, generator=g, device=DEVICE) < spec.occupancy).long()
        Eall = torch.tensor(
            np.stack([emissions_theta(t, spec) for t in thetas]),
            dtype=DTYPE,
            device=DEVICE,
        )
        for t in range(T):
            h = torch.where(t < tau, direction, 1 - direction)
            p = Eall.gather(1, h[:, None]).squeeze(1)
            out[:, t] = torch.binomial(
                torch.full((B,), float(s), dtype=DTYPE, device=DEVICE),
                p,
                generator=g,
            ).long()
        return out

    As, pi0 = transition_schedule(spec)
    h = (torch.rand(B, generator=g, device=DEVICE) < pi0[1]).long()
    Eall = torch.tensor(
        np.stack([emissions_theta(t, spec) for t in thetas]),
        dtype=DTYPE,
        device=DEVICE,
    )
    for t in range(T):
        p = Eall.gather(1, h[:, None]).squeeze(1)
        out[:, t] = torch.binomial(
            torch.full((B,), float(s), dtype=DTYPE, device=DEVICE),
            p,
            generator=g,
        ).long()
        if t < T - 1:
            A = torch.tensor(As[t], dtype=DTYPE, device=DEVICE)
            h = (torch.rand(B, generator=g, device=DEVICE) < A[:, 1].gather(0, h)).long()
    return out