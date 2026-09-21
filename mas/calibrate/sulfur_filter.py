"""Настройка фильтра серы по обучающему периоду: качество предсказания анализа ЛИМС.

Параметры динамики и шума (медленная и быстрая компоненты серы, φ_f, q_f и r приборов, запаздывание Q21, q_b, σ_ЛИМС) и
сдвиг показаний ПАК подбираются минимизацией суммы двух отрицательных логарифмов правдоподобия (NLL) анализов ЛИМС
обучающего периода: по оценке в момент отбора («сейчас») и по прогнозу, сделанному за `forecast_horizon_h` часов до
отбора (горизонт решения). Перед каждым анализом фильтр получает только данные, доступные до момента отбора (сам анализ
входит в фильтр после отбора), поэтому утечки из будущего нет. NLL - логарифм правдоподобия анализа при нормальном
прогнозе со средней и дисперсией `σ_оценки² + σ_ЛИМС²`: он одновременно настраивает точность и калибровку
неопределённости; весовых коэффициентов между критериями нет.

Критерий невыпуклый (локальный поиск из автоковариационной оценки застревает), поэтому стартовая точка хранится в
`config/sulfur_filter_start.json` и найдена глобальным поиском (`global_search`, команда
`python run.py calibrate --global-search`); шаг калибровки уточняет её локально (L-BFGS-B с параллельным градиентом).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

from mas.common import CACHE_DIR, CONFIG_DIR, load_cfg, v
from mas.models.kalman import forecast_sulfur, kalman_sulfur

NAMES = ("log_q_s", "phi_s", "pak_phi_f", "log_pak_q_f", "log_pak_r", "q21_phi_f", "log_q21_q_f", "log_q21_r",
         "q21_delay", "log_q_b", "log_sigma_lims", "phi_fast", "log_q_fast")
BOUNDS = ((-9, -3), (0.97, 0.9995), (0.3, 0.97), (-3, 1.5), (-6, 1.5), (0.3, 0.98), (-3, 1.5), (-6, 1.5), (0, 12),
          (-9, -4), (-0.5, 1.3), (0.5, 0.98), (-4, 1.0))
SHIFT_BOUNDS = (0, 15)
FIT_END_ENV = "NEFTEKOD_FIT_END"  # конец периода подбора вместо `periods.train_end` (внутренняя проверка); наследуется дочерними процессами
START_FILE = CONFIG_DIR / "sulfur_filter_start.json"
FAILED = 1e3


@dataclass
class Dataset:
    pak: pd.DataFrame
    q21: pd.DataFrame
    lims: pd.DataFrame        # анализы серы ГО т.2, доступные на обучающем периоде (для обновлений фильтра)
    running: pd.Series
    step: np.ndarray          # шаг фильтра, соответствующий моменту отбора каждой цели
    y: np.ndarray             # значения целей (анализы ЛИМС)
    keep: np.ndarray          # цели, взятые в рабочем режиме
    mu: float
    horizon_steps: int        # горизонт прогноза, шагов
    keep_forecast: np.ndarray  # цели, для которых установка работала и в момент отбора, и в момент прогноза


_DATASET: Dataset | None = None


def set_dataset(data: Dataset | None) -> None:
    """Подмена набора данных (тесты, однопроцессный режим)."""
    global _DATASET
    _DATASET = data


def make_dataset(pak: pd.DataFrame, q21: pd.DataFrame, lims: pd.DataFrame, running: pd.Series, train_end: pd.Timestamp,
                 mu: float, horizon_steps: int = 24) -> Dataset:
    ls = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")].sort_values("time")
    r = running[running.index < train_end]
    targets = ls[ls["time"] < train_end]
    step = np.clip(r.index.searchsorted(targets["time"].to_numpy(), side="right") - 1, 0, len(r) - 1)
    base = np.clip(step - horizon_steps, 0, None)
    keep = r.to_numpy()[step]
    return Dataset(pak, q21, ls[ls["available_at"] < train_end], r, step, targets["value"].to_numpy(), keep, mu,
                   horizon_steps, keep & r.to_numpy()[base] & (step >= horizon_steps))


def fit_end() -> pd.Timestamp:
    """Конец периода подбора: `periods.train_end`, либо значение переменной окружения `NEFTEKOD_FIT_END`."""
    return pd.Timestamp(os.environ.get(FIT_END_ENV) or load_cfg()["periods"]["train_end"])


def _dataset() -> Dataset:
    """Набор данных из кэша (в каждом процессе загружается один раз)."""
    global _DATASET
    if _DATASET is None:
        train_end = fit_end()
        pak, q21 = pd.read_parquet(CACHE_DIR / "pak_sulfur.parquet"), pd.read_parquet(CACHE_DIR / "q21_sulfur.parquet")
        lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
        running = pd.read_parquet(CACHE_DIR / "running.parquet")["ho_running"]
        clean = pak["raw"].where((pak["reason"] == "") & running.reindex(pak.index).fillna(False) & (pak.index < train_end))
        horizon = int(round(v(load_cfg()["sensor_fusion"]["forecast_horizon_h"]) * 6))
        _DATASET = make_dataset(pak, q21, lims, running, train_end, float(clean.mean()), horizon_steps=horizon)
    return _DATASET


def to_params(z, shift: int, mu: float) -> dict:
    """Параметры `kalman_sulfur` из вектора настройки."""
    z = list(map(float, z))
    return {"mu": mu, "phi_s": z[1], "q_s": float(np.exp(z[0])), "q_b": float(np.exp(z[9])),
            "sigma_lims": float(np.exp(z[10])),
            "pak": {"phi_f": z[2], "q_f": float(np.exp(z[3])), "r": float(np.exp(z[4])), "delay": 0, "shift": int(shift)},
            "q21": {"phi_f": z[5], "q_f": float(np.exp(z[6])), "r": float(np.exp(z[7])), "delay": int(round(z[8]))},
            "phi_fast": z[11], "q_fast": float(np.exp(z[12]))}


def _score(y: np.ndarray, mean: np.ndarray, sd: np.ndarray, sigma_lims: float) -> float | None:
    ok = np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
    if ok.sum() < 30:
        return None
    nll = float(-norm.logpdf(y[ok], mean[ok], np.sqrt(sd[ok] ** 2 + sigma_lims ** 2)).mean())
    return nll


def objective(z, shift: int) -> float:
    d = _dataset()
    params = to_params(z, shift, d.mu)
    state, _ = kalman_sulfur({"pak": d.pak, "q21": d.q21}, d.lims, d.running, params)
    at = state.iloc[np.clip(d.step[d.keep] - 1, 0, None)]  # состояние до внесения самого анализа
    now = _score(d.y[d.keep], at["S_hat"].to_numpy(), at["S_sigma"].to_numpy(), params["sigma_lims"])
    base = state.iloc[d.step[d.keep_forecast] - d.horizon_steps]
    mean, sd = forecast_sulfur(base["S_slow"].to_numpy(), base["S_fast"].to_numpy(), base["P_ss"].to_numpy(),
                               base["P_ff"].to_numpy(), base["P_sf"].to_numpy(), params, d.horizon_steps)
    ahead = _score(d.y[d.keep_forecast], mean, sd, params["sigma_lims"])
    return FAILED if now is None or ahead is None else now + ahead


def _task(args) -> float:
    return objective(*args)


def _de_task(x) -> float:
    return objective(x[:-1], int(round(x[-1])))


def load_start(path=None) -> tuple[np.ndarray, int]:
    with open(path or START_FILE, "r", encoding="utf-8") as f:
        start = json.load(f)
    z = np.array([start["params"][n] for n in NAMES], dtype=float)
    return np.clip(z, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS]), int(start["shift"])


def fit_sulfur_filter(data: Dataset | None = None, max_iter: int = 4, workers: int | None = None,
                      start: tuple[np.ndarray, int] | None = None, step: float = 2e-2) -> tuple[dict, dict]:
    """Локальная настройка из стартовой точки: (параметры фильтра, диагностика)."""
    if data is not None:
        set_dataset(data)
    z0, shift0 = start if start is not None else load_start()
    d = _dataset()
    workers = workers if workers is not None else max(1, min(12, (os.cpu_count() or 2) - 1))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    run = pool.map if pool else map
    try:
        f_start = objective(z0, shift0)

        def fun(z):
            points = [(z, shift0)] + [(z + step * np.eye(len(z))[i], shift0) for i in range(len(z))]
            vals = np.array(list(run(_task, points)))
            return float(vals[0]), (vals[1:] - vals[0]) / step

        res = minimize(fun, z0, jac=True, method="L-BFGS-B", bounds=list(BOUNDS), options={"maxiter": max_iter, "maxfun": 2 * max_iter})
        z_best = res.x if res.fun <= f_start else z0
        shifts = [s for s in range(shift0 - 2, shift0 + 3) if SHIFT_BOUNDS[0] <= s <= SHIFT_BOUNDS[1]]
        scores = list(run(_task, [(z_best, s) for s in shifts]))
        shift = shifts[int(np.argmin(scores))]
        f_best = float(min(scores))
    finally:
        if pool:
            pool.shutdown()
    diagnostics = {"objective_start": f_start, "objective": f_best, "shift_steps": shift, "iterations": int(res.nit),
                   "n_targets": int(d.keep.sum())}
    return to_params(z_best, shift, d.mu), diagnostics


def sulfur_diagnostics(pak: pd.DataFrame, q21: pd.DataFrame, lims: pd.DataFrame, running: pd.Series, params: dict,
                       train_end: pd.Timestamp) -> dict:
    """Диагностика настроенного фильтра на обучающем периоде: медианный NIS (около 0.45 при согласованных
    оценках и допущении о нормальности), корреляция приборов, постоянная времени медленной компоненты."""
    _, priors = kalman_sulfur({"pak": pak, "q21": q21}, lims, running, params)
    priors = priors[priors["t"] < train_end]
    ls = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")]
    merged = priors.merge(ls[["available_at", "value"]], left_on="t", right_on="available_at")
    innovation = merged["value"] - merged["S_prior"]
    nis = (innovation ** 2) / (merged["sigma_prior"] ** 2 + params["sigma_lims"] ** 2)
    mask = running.reindex(pak.index).fillna(False) & (pak.index < train_end)
    p, q = pak["raw"].where(mask & (pak["reason"] == "")), q21["raw"].where(mask & (q21["reason"] == ""))
    both = p.notna() & q.notna()
    return {"median_nis_train": float(nis.median()), "corr_pak_q21": float(p[both].corr(q[both])),
            "tau_slow_h": float(-(10.0 / 60.0) / np.log(params["phi_s"]))}


def _de_subset(x, free: tuple, base: np.ndarray, shift: int) -> float:
    z = base.copy()
    z[list(free)] = x
    return objective(z, shift)


def global_search(workers: int | None = None, popsize: int = 8, maxiter: int = 22, seed: int = 1,
                  free: tuple[str, ...] | None = None, start_file=None) -> dict:
    """Глобальный поиск (дифференциальная эволюция); результат записывается в `config/sulfur_filter_start.json`.

    `free` - имена искомых параметров (остальные остаются как в стартовой точке); по умолчанию - все и сдвиг ПАК.
    `start_file` - другой файл для стартовой точки и результата (внутренняя проверка не затрагивает рабочий)."""
    from functools import partial
    start_file = start_file or START_FILE
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 2)
    d = _dataset()
    z0, shift0 = load_start(start_file) if start_file.exists() else (np.array([np.mean(b) for b in BOUNDS]), 0)
    if free:
        idx = tuple(NAMES.index(n) for n in free)
        res = differential_evolution(partial(_de_subset, free=idx, base=z0, shift=shift0), [BOUNDS[i] for i in idx],
                                     x0=z0[list(idx)], popsize=popsize, maxiter=maxiter, tol=1e-4, seed=seed, workers=workers,
                                     updating="deferred", polish=False, init="sobol")
        z_best = z0.copy()
        z_best[list(idx)] = res.x
        res.x = np.append(z_best, shift0)
    else:
        res = differential_evolution(_de_task, list(BOUNDS) + [SHIFT_BOUNDS], x0=np.append(z0, shift0), popsize=popsize,
                                     maxiter=maxiter, tol=1e-4, seed=seed, workers=workers, updating="deferred",
                                     polish=False, init="sobol")
    out = {"описание": "Стартовая точка настройки фильтра серы, найденная глобальным поиском по обучающему периоду; "
                       "шаг калибровки уточняет её локально. Параметры: логарифмы дисперсий и σ_ЛИМС, коэффициенты затухания, "
                       "запаздывание Q21 (шагов), сдвиг показаний ПАК `shift` (шагов по 10 мин).",
           "params": {n: float(v) for n, v in zip(NAMES, res.x[:-1])}, "shift": int(round(res.x[-1])),
           "objective": float(res.fun), "mu": d.mu}
    with open(start_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out
