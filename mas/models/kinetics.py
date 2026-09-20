"""Серо-ящичная кинетика: ансамбль 27 членов, отклик на управление, шанс-ограничение.

Векторные вычисления по всем вариантам × членам ансамбля одновременно.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

LEVELS = ("weak", "medium", "strong")


def response_ensemble(calib: dict) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """27 = 3(βT) × 3(βF) × 3(b95); оценки слабая (данные), средняя и сильная (литература) равноправны: сведений о
    большем доверии к одной из них нет, поэтому веса одинаковы (1/27)."""
    weights = [1.0, 1.0, 1.0]
    bt, bf, b95 = calib["beta_t"], calib["beta_f"], calib["b95"]
    members, w = [], []
    for i, lt in enumerate(LEVELS):
        for j, lf in enumerate(LEVELS):
            for k, lb in enumerate(LEVELS):
                members.append((bt[lt], bf[lf], b95[lb]))
                w.append(weights[i] * weights[j] * weights[k])
    return members, np.array(w) / sum(w)


def ensemble_sulfur(ln_s_forecast: np.ndarray, dT5: np.ndarray, rF9: np.ndarray, dF32: np.ndarray,
                     gamma: float, members: list[tuple[float, float, float]], weights: np.ndarray) -> np.ndarray:
    """S_k (N вариантов × M членов), мг/кг."""
    m = np.asarray(members)
    ln_s = (ln_s_forecast[:, None] + np.outer(dT5, m[:, 0]) + np.outer(np.log1p(rF9), m[:, 1])
            + gamma * np.outer(dF32, m[:, 2]))
    return np.exp(ln_s)


def t95_ensemble(t95_mid: np.ndarray, dF32: np.ndarray, b95_values: np.ndarray) -> np.ndarray:
    """T95_k (N × M): T95_new = t95_est + b95·ΔF32."""
    return t95_mid[:, None] + np.outer(dF32, b95_values)


def chance_exceed(s_k: np.ndarray, sigma_rel: np.ndarray, weights: np.ndarray, threshold: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """P_exceed взвешенный по смеси и P_exceed_worst (максимум по членам), по строкам."""
    z = (threshold - s_k) / (s_k * sigma_rel[:, None])
    p = 1.0 - norm.cdf(z)
    return p @ weights, p.max(axis=1)


def upper_quantile_mix(s_k: np.ndarray, sigma_rel: np.ndarray, weights: np.ndarray,
                        lo: float = 5.0, hi: float = 16.0, step: float = 0.05, q: float = 0.95) -> np.ndarray:
    """Квантиль q смеси гауссиан, поиск по сетке."""
    grid = np.arange(lo, hi + step / 2, step)
    cdf = np.empty((s_k.shape[0], len(grid)))
    for j, x in enumerate(grid):
        z = (x - s_k) / (s_k * sigma_rel[:, None])
        cdf[:, j] = norm.cdf(z) @ weights
    reached = cdf >= q
    return np.where(reached.any(axis=1), grid[np.argmax(reached, axis=1)], hi)  # верхняя граница сетки, если квантиль выше неё
