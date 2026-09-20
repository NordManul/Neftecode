"""Фоновая вероятность превышения норм и порог запуска действия."""
from __future__ import annotations

import pandas as pd
import pytest

from mas.calibrate.risk import background_exceedance
from mas.common import action_threshold, load_calib, load_cfg, v


def _lims(times, values, parameter="Mg.Sulfur"):
    return pd.DataFrame({"point": "HT2", "parameter": parameter, "time": times, "value": values})


def test_background_counts_only_steady_running_analyses_before_train_end():
    cfg = load_cfg()
    idx = pd.date_range("2024-01-01", periods=24 * 6 * 20, freq="10min")
    running = pd.DataFrame({"ho_running": True, "ho_hours_since_start": (pd.RangeIndex(len(idx)) / 6).to_numpy()}, index=idx)
    running.loc[idx[:6 * 24 * 2], "ho_running"] = False
    window = 72.0  # длительность пуска в этом синтетическом примере
    times = [idx[10], idx[int((window + 24) * 6)], idx[int((window + 48) * 6)], idx[int((window + 60) * 6)], idx[int((window + 72) * 6)]]
    lims = _lims(times, [15.0, 12.0, 8.0, 9.0, 11.0])
    train_end = idx[int((window + 70) * 6)]
    result = background_exceedance(lims, running, cfg, train_end, window)
    # первый анализ - установка остановлена, последний - после конца обучения; остаются 12, 8, 9 -> 1 из 3
    assert result["n_sulfur"] == 3
    assert result["sulfur_background"] == pytest.approx(1 / 3)


def test_action_threshold_is_max_of_alpha_and_background():
    cfg, calib = load_cfg(), load_calib()
    alpha = v(cfg["spec"]["risk_alpha"])
    saved = calib["risk"]["sulfur_background"]
    try:
        calib["risk"]["sulfur_background"] = alpha + 0.1
        assert action_threshold(cfg, "sulfur") == pytest.approx(alpha + 0.1)
        calib["risk"]["sulfur_background"] = alpha / 2
        assert action_threshold(cfg, "sulfur") == pytest.approx(alpha)
    finally:
        calib["risk"]["sulfur_background"] = saved
