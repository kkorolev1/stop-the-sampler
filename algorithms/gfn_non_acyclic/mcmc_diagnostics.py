"""MCMC diagnostics utilities (R-hat and Geweke)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


EPS = 1e-12


@dataclass
class RhatSummary:
    rank_split_rhat: np.ndarray
    split_rhat: np.ndarray
    classical_rhat: np.ndarray

    @property
    def max_rank_split_rhat(self) -> float:
        return float(np.nanmax(self.rank_split_rhat))

    @property
    def max_split_rhat(self) -> float:
        return float(np.nanmax(self.split_rhat))

    @property
    def max_classical_rhat(self) -> float:
        return float(np.nanmax(self.classical_rhat))


def _flatten_chain_dims(chains: np.ndarray) -> np.ndarray:
    arr = np.asarray(chains)
    if arr.ndim < 3:
        raise ValueError("Expected chains shape [n_chains, n_steps, ...params].")
    return arr.reshape(arr.shape[0], arr.shape[1], -1)


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Rank-normalize each scalar parameter across all chains/steps."""
    x = _flatten_chain_dims(chains)
    c, n, p = x.shape
    out = np.empty_like(x, dtype=np.float64)
    for k in range(p):
        v = x[:, :, k].reshape(-1)
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
        u = ranks / (v.size + 1.0)
        out[:, :, k] = u.reshape(c, n)
    return out


def _split_chains(chains: np.ndarray) -> np.ndarray:
    x = _flatten_chain_dims(chains)
    c, n, p = x.shape
    half = n // 2
    if half < 2:
        raise ValueError("Need at least 4 draws for split-Rhat.")
    first = x[:, :half, :]
    second = x[:, n - half :, :]
    return np.concatenate([first, second], axis=0).reshape(2 * c, half, p)


def _classical_psrf(chains: np.ndarray) -> np.ndarray:
    """Classical PSRF over shape [n_chains, n_steps, n_params]."""
    x = _flatten_chain_dims(chains).astype(np.float64)
    m, n, _ = x.shape
    if m < 2 or n < 2:
        raise ValueError("Need at least 2 chains and 2 draws per chain for R-hat.")

    chain_means = np.mean(x, axis=1)
    overall_mean = np.mean(chain_means, axis=0)
    sq = (chain_means - overall_mean[None, :]) ** 2
    b = n * np.sum(sq, axis=0) / max(m - 1, 1)
    w = np.mean(np.var(x, axis=1, ddof=1), axis=0)
    vhat = ((n - 1) / n) * w + b / n
    rhat = np.sqrt(np.maximum(vhat, EPS) / np.maximum(w, EPS))
    return rhat


def compute_rhat_summary(chains: np.ndarray) -> RhatSummary:
    """Compute classical, split, and rank-normalized split R-hat."""
    x = _flatten_chain_dims(chains)
    classical = _classical_psrf(x)
    split = _classical_psrf(_split_chains(x))
    rank_split = _classical_psrf(_split_chains(_rank_normalize(x)))
    return RhatSummary(
        rank_split_rhat=rank_split,
        split_rhat=split,
        classical_rhat=classical,
    )


def rhat_status(max_rhat: float) -> str:
    if max_rhat < 1.01:
        return "strict_pass"
    if max_rhat < 1.05:
        return "warning"
    if max_rhat < 1.10:
        return "exploratory_only"
    return "fail"


def _spectrum0_iid(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1))


def _spectrum0_batchmeans(x: np.ndarray) -> float:
    n = x.size
    b = max(int(np.sqrt(n)), 2)
    a = n // b
    if a < 2:
        return _spectrum0_iid(x)
    y = x[: a * b].reshape(a, b).mean(axis=1)
    return float(b * np.var(y, ddof=1))


def _spectrum0_ar1(x: np.ndarray) -> float:
    """AR(1)-based spectral estimate at frequency zero."""
    n = x.size
    if n < 4:
        return _spectrum0_iid(x)
    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    if gamma0 <= EPS:
        return EPS
    gamma1 = np.dot(xc[1:], xc[:-1]) / (n - 1)
    phi = float(np.clip(gamma1 / gamma0, -0.99, 0.99))
    sigma2 = max(gamma0 * (1.0 - phi * phi), EPS)
    return float(sigma2 / max((1.0 - phi) ** 2, EPS))


def spectrum0(
    x: np.ndarray, estimator: Literal["ar", "batchmeans", "iid"] = "ar"
) -> float:
    if estimator == "ar":
        return _spectrum0_ar1(x)
    if estimator == "batchmeans":
        return _spectrum0_batchmeans(x)
    if estimator == "iid":
        return _spectrum0_iid(x)
    raise ValueError(f"Unknown spectrum estimator: {estimator}")


def compute_geweke_z(
    chains: np.ndarray,
    frac1: float = 0.1,
    frac2: float = 0.5,
    estimator: Literal["ar", "batchmeans", "iid"] = "ar",
) -> np.ndarray:
    """Per chain/param Geweke Z scores with shape [n_chains, n_params]."""
    if frac1 <= 0 or frac2 <= 0 or (frac1 + frac2) >= 1:
        raise ValueError("Need frac1 > 0, frac2 > 0, and frac1 + frac2 < 1.")

    x = _flatten_chain_dims(chains).astype(np.float64)
    c, n, p = x.shape
    n_a = max(int(frac1 * n), 2)
    n_b = max(int(frac2 * n), 2)
    if n_a + n_b >= n:
        raise ValueError("Geweke windows leave no gap; increase chain length.")

    z = np.zeros((c, p), dtype=np.float64)
    for i in range(c):
        for k in range(p):
            series = x[i, :, k]
            a = series[:n_a]
            b = series[-n_b:]
            sa0 = spectrum0(a, estimator=estimator)
            sb0 = spectrum0(b, estimator=estimator)
            denom = np.sqrt(max(sa0 / n_a + sb0 / n_b, EPS))
            z[i, k] = (a.mean() - b.mean()) / denom
    return z


def summarize_geweke(z_scores: np.ndarray) -> dict:
    abs_z = np.abs(np.asarray(z_scores))
    frac_196 = float(np.mean(abs_z > 1.96))
    frac_258 = float(np.mean(abs_z > 2.58))
    max_abs = float(np.max(abs_z))

    if frac_258 == 0.0 and frac_196 <= 0.01:
        status = "pass"
    elif frac_258 <= 0.01 and frac_196 <= 0.10:
        status = "warning"
    else:
        status = "fail"

    return {
        "max_abs_z": max_abs,
        "frac_abs_z_gt_1p96": frac_196,
        "frac_abs_z_gt_2p58": frac_258,
        "status": status,
    }
