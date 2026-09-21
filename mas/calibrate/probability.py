"""Калибровка вероятности превышения нормы по сере: расчётная вероятность фильтра -> наблюдаемая частота превышений.

Вероятность фильтра `P = 1 - Φ((10 - S_hat) / S_sigma)` берётся по состоянию до самого анализа (без результата анализа),
метка - результат ЛИМС выше нормы, только установившийся режим обучающего периода. Отображение монотонное
(изотоническая регрессия): форма зависимости не предполагается.

Наблюдаемая частота превышений при малой расчётной вероятности почти не зависит от неё (нижняя граница задаётся погрешностью
лаборатории и быстрыми колебаниями серы, которых фильтр не предсказывает). Поэтому допустимость действия оценивается по
отображённой вероятности: требование «расчётная вероятность не выше 5 %» не отличает варианты, у которых наблюдаемая частота
превышений одна и та же.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from mas.common import v


def fit_probability_map(state: pd.DataFrame, lims: pd.DataFrame, running: pd.DataFrame, cfg: dict, train_end: pd.Timestamp,
                        window: float) -> dict:
    """`state` - оценка фильтра серы (`S_hat`, `S_sigma`), `window` - длительность пускового режима, ч."""
    limit = v(cfg["spec"]["sulfur_max_mgkg"])
    rows = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur") & (lims["time"] < train_end)].sort_values("time")
    idx = state.index.to_numpy()
    pos = np.clip(np.searchsorted(idx, rows["time"].to_numpy(), side="right") - 2, 0, len(idx) - 1)      # до самого анализа
    steady = (running["ho_running"] & (running["ho_hours_since_start"] > window)).reindex(state.index).fillna(False).to_numpy()[pos]
    p = 1 - norm.cdf((limit - state["S_hat"].to_numpy()[pos]) / state["S_sigma"].to_numpy()[pos])
    y = (rows["value"].to_numpy() > limit).astype(int)
    keep = steady & np.isfinite(p)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[keep], y[keep])
    return {"raw": [float(x) for x in iso.X_thresholds_], "calibrated": [float(x) for x in iso.y_thresholds_],
            "n_analyses": int(keep.sum())}
