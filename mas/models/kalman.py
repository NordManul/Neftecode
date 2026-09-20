"""Фильтр Калмана по сере: N анализаторов (ПАК, Q21) + ЛИМС.

Состояние x = [s_медл, (s_быстр), f1, b1, f2, b2, ...]: уровень серы S = μ + s_медл (+ s_быстр) центрирован для
линейности OU-модели; быстрая компонента настоящей вариации серы включается параметром `phi_fast` (медленная -
уровень на масштабе суток, быстрая - часов). Для каждого прибора: f - его быстрый шум, b - дрейфующее смещение.
ЛИМС никогда не отбрасывается. `params[прибор]["shift"]` - сдвиг показаний прибора по времени в шагах
(транспортное запаздывание между точкой измерения и точкой отбора). Синхронизация по времени (searchsorted),
не по номеру строки. Цикл последователен по времени - исключение из правила "без циклов по строкам",
т.к. алгоритм неотделим от порядка.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


INNOVATION_GATE = 5.0  # выброс: отклонение измерения от прогноза более чем на пять стандартных отклонений инновации


def _analyzer_matrix(dim: int, n_s: int, i: int) -> np.ndarray:
    h = np.zeros(dim)
    h[:n_s] = 1.0
    h[n_s + 2 * i] = 1.0
    h[n_s + 1 + 2 * i] = 1.0
    return h


def _delayed(z: np.ndarray, steps: int) -> np.ndarray:
    """Показание прибора, заведомо относящееся к моменту на `steps` шагов раньше (транспортное запаздывание)."""
    if steps <= 0:
        return z
    out = np.full_like(z, np.nan)
    out[steps:] = z[:-steps]
    return out


def forecast_sulfur(s_slow, s_fast, p_ss, p_ff, p_sf, params: dict, n_steps: int):
    """Прогноз серы на `n_steps` шагов: (среднее, СКО). Медленная и быстрая компоненты возвращаются к μ со своими φ."""
    mu, phi_s, phi_f = params["mu"], params["phi_s"], params.get("phi_fast", 0.0)
    a, b = phi_s ** n_steps, (phi_f ** n_steps if phi_f else 0.0)
    var_s = params["q_s"] / (1 - phi_s ** 2)
    var_f = params.get("q_fast", 0.0) / (1 - phi_f ** 2) if phi_f else 0.0
    mean = mu + a * (s_slow - mu) + b * s_fast
    var = a * a * p_ss + b * b * p_ff + 2 * a * b * p_sf + var_s * (1 - a * a) + var_f * (1 - b * b)
    return mean, np.sqrt(var)


def kalman_sulfur(analyzers: dict[str, pd.DataFrame], lims: pd.DataFrame, running: pd.Series,
                   params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(состояние на каждом шаге, априорные оценки перед каждым обновлением ЛИМС по HT2:Mg.Sulfur)."""
    idx = running.index
    n = len(idx)
    names = list(analyzers.keys())
    two = "phi_fast" in params
    n_s = 2 if two else 1
    dim = n_s + 2 * len(names)
    mu = params["mu"]

    phi = np.array([params["phi_s"]] + ([params["phi_fast"]] if two else [])
                   + [x for nm in names for x in (params[nm]["phi_f"], 1.0)])
    q_diag = np.array([params["q_s"]] + ([params["q_fast"]] if two else [])
                      + [x for nm in names for x in (params[nm]["q_f"], params["q_b"])])
    F = np.diag(phi)
    Q = np.diag(q_diag)
    h_level = np.zeros(dim)
    h_level[:n_s] = 1.0

    z = {nm: _delayed(analyzers[nm]["raw"].where(analyzers[nm]["reason"] == "").reindex(idx).to_numpy(),
                      int(params[nm].get("shift", 0))) for nm in names}
    run_arr = running.to_numpy()
    startup = (~run_arr[:-1]) & run_arr[1:]

    lims_h = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")].sort_values("available_at")
    lims_step = idx.searchsorted(lims_h["available_at"].to_numpy(), side="right") - 1  # backward
    lims_by_step: dict[int, float] = {int(s): float(v) for s, v in zip(lims_step, lims_h["value"]) if 0 <= s < n}
    sigma_lims2 = params["sigma_lims"] ** 2

    x = np.zeros(dim)
    P = np.eye(dim) * 9.0
    cols = {k: np.full(n, np.nan) for k in ("S_hat", "S_sigma", "S_slow", "S_fast", "P_ss", "P_ff", "P_sf")}
    out_bias = {nm: np.full(n, np.nan) for nm in names}
    priors: list[dict] = []

    for t in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        if t > 0 and startup[t - 1]:
            P[0, 0] = max(P[0, 0], 9.0)
            P[0, 1:] = 0.0
            P[1:, 0] = 0.0

        if t in lims_by_step:
            priors.append({"t": idx[t], "S_prior": mu + h_level @ x, "sigma_prior": float(np.sqrt(max(h_level @ P @ h_level, 0.0)))})

        if run_arr[t]:
            for i, nm in enumerate(names):
                zt = z[nm][t]
                if np.isnan(zt):
                    continue
                h = _analyzer_matrix(dim, n_s, i)
                y = zt - mu - h @ x
                sv = h @ P @ h + params[nm]["r"] + params[nm]["delay"] * params["q_s"]
                if sv <= 0 or abs(y) > INNOVATION_GATE * np.sqrt(sv):
                    continue
                k = (P @ h) / sv
                x = x + k * y
                P = P - np.outer(k, h @ P)

        if t in lims_by_step:
            y = lims_by_step[t] - mu - h_level @ x
            sv = h_level @ P @ h_level + sigma_lims2
            if abs(y) <= INNOVATION_GATE * np.sqrt(sv):  # результат, далёкий от прогноза фильтра, - выброс, а не измерение
                k = (P @ h_level) / sv
                x = x + k * y
                P = P - np.outer(k, h_level @ P)

        cols["S_hat"][t] = mu + h_level @ x
        cols["S_sigma"][t] = np.sqrt(max(h_level @ P @ h_level, 0.0))  # отрицательные значения - только ошибка округления
        cols["S_slow"][t], cols["P_ss"][t] = mu + x[0], P[0, 0]
        cols["S_fast"][t], cols["P_ff"][t], cols["P_sf"][t] = (x[1], P[1, 1], P[0, 1]) if two else (0.0, 0.0, 0.0)
        for i, nm in enumerate(names):
            out_bias[nm][t] = x[n_s + 1 + 2 * i]

    cols.update({f"bias_{nm}": out_bias[nm] for nm in names})
    return pd.DataFrame(cols, index=idx), pd.DataFrame(priors)
