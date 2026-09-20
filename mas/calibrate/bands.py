"""Модельные коридоры (перцентили контролируемых тегов) и база индекса перепада давления.

Коридоры - обучающий период, только режим работы; ключи сортируются
(упорядоченные ключи, иначе calibrated.json недетерминирован).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.common import v

PERCENTILES = (1, 5, 95, 99)
T5_PERCENTILES = (1, 5, 50, 95, 99)  # для T5 нужна ещё медиана - точка отсчёта фактора близости к верхней границе


def _corridor(s: pd.Series, percentiles: tuple[int, ...] = PERCENTILES) -> dict[str, float]:
    clean = s.dropna()
    return {f"p{p}": float(np.percentile(clean, p)) for p in percentiles}


def reference_feed(ho_vals: pd.DataFrame, ho_running: pd.Series, train_end: pd.Timestamp) -> float:
    """Реперная загрузка ГО - медиана F9 в рабочем режиме обучающего периода, т/ч."""
    train = ho_vals.index < train_end
    return float(ho_vals["F9"].where(ho_running & train).median())


def calibrate_bands(avt_vals: pd.DataFrame, ho_vals: pd.DataFrame, avt_running: pd.Series,
                     ho_running: pd.Series, cfg: dict, feed_ref: float) -> dict:
    train_end = pd.Timestamp(cfg["periods"]["train_end"])
    density = v(cfg["process"]["feed_density_t_m3"])

    ho_train = ho_vals.loc[ho_vals.index < train_end]
    ho_run_train = ho_running.loc[ho_running.index < train_end]
    gor = (ho_train["F2"] / (ho_train["F9"] / density)).where(ho_run_train)
    avt_train = avt_vals.loc[avt_vals.index < train_end]
    avt_run_train = avt_running.loc[avt_running.index < train_end]

    ho_tags = sorted((set(cfg["reliability"]["monitored_tags"]["ho"]) | {"F9", "T5"}) - {"GOR"})
    avt_tags = sorted(set(cfg["reliability"]["monitored_tags"]["avt"]) | {"F32"})
    corridors = {tag: _corridor(ho_train[tag].where(ho_run_train), T5_PERCENTILES if tag == "T5" else PERCENTILES)
                 for tag in ho_tags}
    corridors["GOR"] = _corridor(gor)
    corridors.update({tag: _corridor(avt_train[tag].where(avt_run_train)) for tag in avt_tags})

    dp_norm = (ho_train["P8"] * (feed_ref / ho_train["F9"]) ** 2).where(ho_run_train)
    t5 = ho_train["T5"].where(ho_run_train)
    roc4h = (t5 - t5.shift(24)).abs().dropna()  # четырёхчасовое изменение T5 (шаг ряда 10 мин)
    return {"corridors": {k: corridors[k] for k in sorted(corridors)}, "dp_norm_base": float(np.nanmedian(dp_norm)),
            "t5_roc4h_p95": float(np.percentile(roc4h, 95)), "t5_roc4h_p99": float(np.percentile(roc4h, 99))}
