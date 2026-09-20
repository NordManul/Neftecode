"""Погрешность оценки T95: σ одного анализа и скорость дрейфа истинного T95 (только обучающий период).

Модель: анализ = истинное T95 + шум σ; истинное T95 между отборами дрейфует как случайное блуждание с дисперсией
`drift` °C² в сутки. Оценка перед новым анализом (`t95_estimate`) - взвешенная по обратной дисперсии комбинация
предыдущих анализов, приведённых к текущему F32. Предсказательное распределение нового анализа - нормальное со
средним `оценка` и дисперсией `дисперсия оценки + σ²`; параметры находятся максимизацией правдоподобия остатков.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mas.models.t95 import recent_samples

MAX_SAMPLES = 3


def _residual_table(lims: pd.DataFrame, f32: pd.Series, b95_mid: float, train_end: pd.Timestamp, max_age_h: float):
    g = lims[(lims["point"] == "HT2") & (lims["parameter"] == "95%.T")].sort_values("available_at")
    y, adj, age = [], [], []
    for t, value in zip(g["time"], g["value"]):
        if t >= train_end:
            continue
        samples = recent_samples(g, f32, t - pd.Timedelta(seconds=1), max_age_h, n=MAX_SAMPLES)  # без самого анализа
        f32_now = f32.loc[t - pd.Timedelta(hours=5):t - pd.Timedelta(hours=4)].mean()
        if not samples or not np.isfinite(f32_now):
            continue
        row_adj, row_age = np.full(MAX_SAMPLES, np.nan), np.full(MAX_SAMPLES, np.nan)
        for i, s in enumerate(samples):
            row_adj[i] = s["value"] + b95_mid * (f32_now - s["F32_at_sample"])
            row_age[i] = s["age_h"]
        y.append(value), adj.append(row_adj), age.append(row_age)
    return np.array(y), np.array(adj), np.array(age)


def calibrate_t95_noise(lims: pd.DataFrame, f32: pd.Series, b95_mid: float, train_end: pd.Timestamp,
                         max_age_h: float, start: tuple[float, float] = (4.65, 1.2)) -> dict:
    y, adj, age = _residual_table(lims, f32, b95_mid, train_end, max_age_h)
    have = np.isfinite(adj)

    def nll(log_params: np.ndarray) -> float:
        sigma2, drift = np.exp(2 * log_params[0]), np.exp(log_params[1])
        w = np.where(have, 1.0 / (sigma2 + drift * np.where(have, age, 0.0) / 24.0), 0.0)
        est = (np.where(have, adj, 0.0) * w).sum(axis=1) / w.sum(axis=1)
        var = 1.0 / w.sum(axis=1) + sigma2
        return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - est) ** 2 / var))

    res = minimize(nll, x0=[np.log(start[0]), np.log(start[1])], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8})
    return {"sigma": float(np.exp(res.x[0])), "drift": float(np.exp(res.x[1])), "n": int(len(y))}
