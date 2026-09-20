"""Регламент ВАК: пригодность формулы - превосходство над тривиальными оценками, маска входных значений."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.models.vak import _mask_inputs, _metrics

ALPHA = 0.05


def _series(n_train: int, n_test: int, noise: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_train + n_test, freq="1D")
    actual = pd.Series(50.0 + np.cumsum(rng.normal(0, 3.0, len(idx))), index=idx)
    pred = actual + rng.normal(0, noise, len(idx))
    return pred, actual, idx[n_train]


def test_accurate_formula_is_suitable():
    pred, actual, train_end = _series(300, 200, noise=0.5)
    result = _metrics(pred, actual, train_end, ALPHA)
    assert result["verdict"] == "пригодна" and result["r2_test"] > 0.9
    assert result["mae_calibrated"] < result["mae_persistence"]


def test_formula_worse_than_last_analysis_is_not_suitable():
    pred, actual, train_end = _series(300, 200, noise=200.0)
    assert _metrics(pred, actual, train_end, ALPHA)["verdict"] == "не пригодна"


def test_too_few_points_are_reported_as_insufficient():
    pred, actual, train_end = _series(2, 2, noise=0.1)
    assert _metrics(pred, actual, train_end, ALPHA)["verdict"] == "недостаточно данных"


def test_inputs_outside_training_corridor_are_masked():
    idx = pd.date_range("2024-01-01", periods=2000, freq="10min")
    rng = np.random.default_rng(1)
    telemetry = pd.DataFrame({"T5": rng.normal(370, 5, len(idx))}, index=idx)
    telemetry.iloc[1500, 0] = 9999.0                               # единичный выброс датчика в тестовой части
    running = pd.Series(True, index=idx)
    masked = _mask_inputs(telemetry, running, ["T5"], idx[1000])
    lo, hi = telemetry["T5"].iloc[:1000].quantile([0.01, 0.99])
    assert np.isnan(masked["T5"].iloc[1500])
    assert masked["T5"].dropna().between(lo, hi).all()
