"""Пороги, получаемые по данным, а не задаваемые вручную.

* Признак работы: порог между режимом останова и рабочим режимом - метод Оцу (максимум межклассовой дисперсии
  гистограммы) по расходу сырья ГО (F9), температуре Т5 и расходу АВТ (F65).
* Заморозка телеметрии: длина серии одинаковых значений, которой не было в рабочем режиме обучающего периода.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.prepare.quality_flags import trailing_run_length

HISTOGRAM_BINS = 256  # стандартное число уровней метода Оцу


def otsu_threshold(x: pd.Series, bins: int = HISTOGRAM_BINS) -> float:
    counts, edges = np.histogram(x.dropna(), bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    below = np.cumsum(counts)
    above = below[-1] - below
    mean_below = np.cumsum(counts * centers) / np.maximum(below, 1)
    mean_above = (np.sum(counts * centers) - np.cumsum(counts * centers)) / np.maximum(above, 1)
    score = below * above * (mean_below - mean_above) ** 2
    best = centers[np.isclose(score, score.max(), rtol=1e-9)]  # при пустом промежутке между режимами - его середина
    return float((best[0] + best[-1]) / 2)


def derive_thresholds(avt_vals: pd.DataFrame, ho_vals: pd.DataFrame, train_end: pd.Timestamp,
                      control_tags: dict[str, list[str]]) -> dict:
    """{'ho_feed_min_tph', 'ho_temp_min_c', 'avt_feed_min', 'telemetry_freeze_steps'}.

    Длина серии одинаковых значений рабочего режима обучающего периода берётся по управляющим тегам (T5, F9, F32)."""
    thresholds = {"ho_feed_min_tph": otsu_threshold(ho_vals["F9"]), "ho_temp_min_c": otsu_threshold(ho_vals["T5"]),
                  "avt_feed_min": otsu_threshold(avt_vals["F65"])}
    running = (ho_vals["F9"] > thresholds["ho_feed_min_tph"]) & (ho_vals["T5"] > thresholds["ho_temp_min_c"])
    avt_running = avt_vals["F65"] > thresholds["avt_feed_min"]
    longest = 0
    for frame, mask, tags in ((ho_vals, running, control_tags["ho"]), (avt_vals, avt_running, control_tags["avt"])):
        train = frame.index < train_end
        for tag in tags:
            runs = trailing_run_length(frame[tag].where(mask & train))
            longest = max(longest, int(runs.max()))
    thresholds["telemetry_freeze_steps"] = longest + 1
    return thresholds
