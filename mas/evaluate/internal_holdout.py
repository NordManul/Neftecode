"""Внутренняя проверка процедуры настройки фильтра серы на данных обучающего периода.

Данных после `periods.test_end` нет, поэтому независимую проверку структуры фильтра (две составляющие серы,
совместный критерий «сейчас» и «через 4 ч») выполняем внутри обучающего периода: параметры подбираются только по
началу периода (по умолчанию 2023 год, глобальный поиск из середины границ, без стартовой точки рабочей
калибровки), а качество оценивается на следующем отрезке (по умолчанию 2024 год), не участвовавшем в подборе. Отрезок
проверки не использовался и при выборе структуры (тот выполнялся по диагностикам отложенного периода).

Результат - `outputs/internal_holdout.json`; рабочие `config/calibrated.json` и `config/sulfur_filter_start.json`
не изменяются.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from mas.calibrate import sulfur_filter as sf
from mas.common import CACHE_DIR, OUTPUTS_DIR, load_calib, load_cfg, to_jsonable, v
from mas.evaluate.validation import _auc, _coverage_brier, _sulfur_estimators
from mas.models.kalman import forecast_sulfur, kalman_sulfur

DEFAULT_SPLIT = "2024-01-01"
START_FILE = OUTPUTS_DIR / "internal_holdout_start.json"
RESULT_FILE = OUTPUTS_DIR / "internal_holdout.json"


def _forecast_metrics(state: pd.DataFrame, params: dict, lims_s: pd.DataFrame, running: pd.Series, horizon_steps: int) -> dict:
    """Прогноз на горизонт решения по состоянию за `horizon_steps` шагов до отбора (без учёта будущих воздействий)."""
    idx = state.index.to_numpy()
    pos = np.clip(np.searchsorted(idx, lims_s["time"].to_numpy(), side="right") - 1, 0, len(idx) - 1)
    base = np.clip(pos - horizon_steps, 0, None)
    run = running.reindex(state.index).fillna(False).to_numpy()
    ok = run[pos] & run[base] & (pos >= horizon_steps)
    y = lims_s["value"].to_numpy()[ok]
    s = state.iloc[base[ok]]
    mean, sd = forecast_sulfur(s["S_slow"].to_numpy(), s["S_fast"].to_numpy(), s["P_ss"].to_numpy(), s["P_ff"].to_numpy(),
                               s["P_sf"].to_numpy(), params, horizon_steps)
    total = np.sqrt(sd ** 2 + params["sigma_lims"] ** 2)
    label = (y > 10).astype(int)
    p = 1 - norm.cdf((10 - mean) / total)
    return {"n": int(ok.sum()), "mae": float(np.abs(y - mean).mean()), "mae_persistence": float(np.abs(y - s["S_hat"].to_numpy()).mean()),
            "coverage_90": float(np.mean((y >= norm.ppf(0.05, mean, total)) & (y <= norm.ppf(0.95, mean, total)))),
            "auc": _auc(mean, label), "brier": float(np.mean((p - label) ** 2)),
            "brier_climatology": float(np.mean((label.mean() - label) ** 2))}


def evaluate_window(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Метрики оценки серы «сейчас» и прогноза на 4 ч для параметров `params` на анализах ЛИМС из [start, end)."""
    pak, q21 = pd.read_parquet(CACHE_DIR / "pak_sulfur.parquet"), pd.read_parquet(CACHE_DIR / "q21_sulfur.parquet")
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
    running = pd.read_parquet(CACHE_DIR / "running.parquet")["ho_running"]
    state, _ = kalman_sulfur({"pak": pak, "q21": q21}, lims, running, params)
    lims_s = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur") & (lims["time"] >= start) & (lims["time"] < end)]
    estimators = _sulfur_estimators(pak, q21, state, lims_s)
    clim = float((lims_s["value"] > 10).mean())
    coverage, brier, brier_clim, alert = _coverage_brier(state, lims_s, clim, params["sigma_lims"])
    horizon = int(round(v(load_cfg()["sensor_fusion"]["forecast_horizon_h"]) * 6))
    return {"n_analyses": int(len(lims_s)), "share_above_10": clim,
            "estimators": {k: estimators[k] for k in ("Фильтр по двум приборам", "Среднее двух приборов, очищенное",
                                                      "Q21, среднее за час, очищенный", "ПАК, среднее за час, очищенный",
                                                      "Последний анализ ЛИМС")},
            "coverage_90": coverage, "brier": brier, "brier_climatology": brier_clim, "alert": alert,
            "forecast": _forecast_metrics(state, params, lims_s, running, horizon)}


def run_internal_holdout(split: str = DEFAULT_SPLIT, workers: int | None = None, popsize: int = 8, maxiter: int = 22) -> dict:
    """Подбор по данным до `split`, проверка на отрезке от `split` до `periods.train_end`."""
    train_end = pd.Timestamp(load_cfg()["periods"]["train_end"])
    os.environ[sf.FIT_END_ENV] = split       # дочерние процессы глобального поиска наследуют окружение
    sf.set_dataset(None)
    try:
        if START_FILE.exists():
            START_FILE.unlink()
        sf.global_search(workers=workers, popsize=popsize, maxiter=maxiter, start_file=START_FILE)
        params, fit = sf.fit_sulfur_filter(workers=workers, start=sf.load_start(START_FILE))
    finally:
        os.environ.pop(sf.FIT_END_ENV, None)
        sf.set_dataset(None)
    start = pd.Timestamp(split)
    report = {"fit_period": {"end": split}, "check_period": {"start": split, "end": str(train_end)},
              "fit": fit, "params": params,
              "out_of_sample": evaluate_window(params, start, train_end),
              "working_parameters_same_window": evaluate_window(load_calib()["sulfur"], start, train_end)}
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)
    return report
