"""Длительность пускового режима: постоянная времени релаксации серы после пуска по данным обучающего периода.

Среднее абсолютное отклонение часовых значений Q21 от медианы убывает с числом часов после пуска: сразу после пуска
установка выдаёт продукт с сильно отличающейся серой, затем режим устанавливается. Зависимость описывается экспонентой
`b + a·exp(-h/τ)`; `b` - уровень установившегося режима. Пусковой режим длится три постоянные времени
(95 % переходного процесса). Используется Q21: шум его измерения в десятки раз меньше, чем у ПАК (`calibrated.json`,
`sulfur.q21.r` и `sulfur.pak.r`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

TIME_CONSTANTS = 3  # длительность пускового режима, постоянных времени: затухание переходного процесса на 95 %


def _decay(hours: np.ndarray, a: float, tau: float, b: float) -> np.ndarray:
    return b + a * np.exp(-hours / tau)


def startup_relaxation(analyzer: pd.DataFrame, hours_since_start: pd.Series, train_end: pd.Timestamp) -> dict:
    """{'tau_h', 'window_h', 'amplitude', 'steady_deviation', 'n_hours'}; `analyzer` - `raw` и `reason` Q21 (шаг 10 мин)."""
    values = analyzer["raw"].where(analyzer["reason"] == "")[analyzer.index < train_end]
    hourly = values.resample("1h").mean()
    frame = pd.DataFrame({"s": hourly, "h": hours_since_start.resample("1h").first().reindex(hourly.index)}).dropna()
    frame["dev"] = (frame["s"] - frame["s"].median()).abs()
    by_hour = frame.groupby(np.floor(frame["h"]).astype(int))["dev"].agg(["mean", "count"])
    x, y, n = by_hour.index.to_numpy(float) + 0.5, by_hour["mean"].to_numpy(), by_hour["count"].to_numpy()
    (a, tau, b), _ = curve_fit(_decay, x, y, p0=[y[0] - y[-1], x[len(x) // 10] or 1.0, y[-1]], sigma=1 / np.sqrt(n),
                               bounds=([0.0, 0.5, 0.0], [np.inf, np.inf, np.inf]))
    return {"tau_h": float(tau), "window_h": float(TIME_CONSTANTS * tau), "amplitude": float(a), "steady_deviation": float(b),
            "n_hours": int(len(frame))}
