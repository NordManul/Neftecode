"""Фоновая вероятность превышения норм по обучающим данным.

Доля анализов ЛИМС, превысивших норму, в установившемся режиме обучающего периода - это вероятность превышения,
которую установка имеет «по умолчанию», без всякой информации о текущем состоянии. Если она выше допустимой,
то нарушение нормы - свойство рабочего уровня режима, и корректирующее действие по единичному прогнозу его не
устраняет. Поэтому порог запуска действия отсчитывается от этого фона (`mas.common.action_threshold`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.common import v


def _share_above(lims: pd.DataFrame, running: pd.DataFrame, param: str, limit: float, train_end: pd.Timestamp,
                 startup_window_h: float) -> tuple[float, int]:
    rows = lims[(lims["point"] == "HT2") & (lims["parameter"] == param) & (lims["time"] < train_end)].sort_values("time")
    state = running[["ho_running", "ho_hours_since_start"]].reindex(rows["time"], method="ffill")
    steady = (state["ho_running"].fillna(False).astype(bool) & (state["ho_hours_since_start"] > startup_window_h)).to_numpy()
    values = rows["value"].to_numpy()[steady]
    return (float(np.mean(values > limit)) if len(values) else 0.0), int(len(values))


def sulfur_level_sd(lims: pd.DataFrame, running: pd.DataFrame, train_end: pd.Timestamp, window: float, sigma_lims: float) -> float:
    """Стандартное отклонение уровня серы в рабочем режиме обучающего периода, мг/кг: из дисперсии анализов ЛИМС вычтена
    дисперсия погрешности лабораторного определения (`sigma_lims`)."""
    rows = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur") & (lims["time"] < train_end)].sort_values("time")
    state = running[["ho_running", "ho_hours_since_start"]].reindex(rows["time"], method="ffill")
    steady = (state["ho_running"].fillna(False).astype(bool) & (state["ho_hours_since_start"] > window)).to_numpy()
    values = rows["value"].to_numpy()[steady]
    return float(np.sqrt(max(np.var(values) - sigma_lims ** 2, 0.0)))


def background_exceedance(lims: pd.DataFrame, running: pd.DataFrame, cfg: dict, train_end: pd.Timestamp,
                          window: float) -> dict:
    """`window` - длительность пускового режима, ч (`mas.common.startup_window_h`)."""
    sulfur, n_sulfur = _share_above(lims, running, "Mg.Sulfur", v(cfg["spec"]["sulfur_max_mgkg"]), train_end, window)
    t95, n_t95 = _share_above(lims, running, "95%.T", v(cfg["spec"]["t95_max_c"]), train_end, window)
    return {"sulfur_background": sulfur, "t95_background": t95, "n_sulfur": n_sulfur, "n_t95": n_t95}
