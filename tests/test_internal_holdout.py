"""Внутренняя проверка настройки фильтра: период подбора и метрики прогноза."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mas.calibrate import sulfur_filter as sf
from mas.common import load_cfg
from mas.evaluate.internal_holdout import _forecast_metrics


def test_fit_end_defaults_to_train_end_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv(sf.FIT_END_ENV, raising=False)
    assert sf.fit_end() == pd.Timestamp(load_cfg()["periods"]["train_end"])
    monkeypatch.setenv(sf.FIT_END_ENV, "2024-01-01")
    assert sf.fit_end() == pd.Timestamp("2024-01-01")


def test_forecast_metrics_on_calibrated_synthetic_state():
    rng = np.random.default_rng(0)
    n, mu, sigma_lims = 4000, 8.0, 1.5
    idx = pd.date_range("2024-01-01", periods=n, freq="10min")
    state = pd.DataFrame({"S_hat": mu, "S_slow": mu, "S_fast": 0.0, "P_ss": 0.0, "P_ff": 0.0, "P_sf": 0.0}, index=idx)
    params = {"mu": mu, "phi_s": 0.99, "q_s": 0.0, "phi_fast": 0.9, "q_fast": 0.0, "sigma_lims": sigma_lims}
    times = idx[100::5]
    lims_s = pd.DataFrame({"time": times, "value": mu + rng.normal(0, sigma_lims, len(times))})
    running = pd.Series(True, index=idx)
    m = _forecast_metrics(state, params, lims_s, running, horizon_steps=24)
    assert m["n"] == len(times)
    assert 0.85 < m["coverage_90"] < 0.95
    assert m["mae"] == pytest.approx(m["mae_persistence"])  # состояние постоянно: прогноз совпадает с текущей оценкой


def test_forecast_metrics_skip_targets_when_plant_was_stopped():
    idx = pd.date_range("2024-01-01", periods=600, freq="10min")
    state = pd.DataFrame({"S_hat": 8.0, "S_slow": 8.0, "S_fast": 0.0, "P_ss": 0.1, "P_ff": 0.0, "P_sf": 0.0}, index=idx)
    params = {"mu": 8.0, "phi_s": 0.99, "q_s": 0.0, "phi_fast": 0.9, "q_fast": 0.0, "sigma_lims": 1.0}
    running = pd.Series(True, index=idx)
    running.iloc[200:300] = False
    lims_s = pd.DataFrame({"time": [idx[250], idx[400]], "value": [8.0, 9.0]})
    assert _forecast_metrics(state, params, lims_s, running, horizon_steps=24)["n"] == 1
