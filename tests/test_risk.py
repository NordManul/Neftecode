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


def test_action_threshold_does_not_depend_on_calibration():
    cfg = load_cfg()
    assert action_threshold(cfg, "sulfur") == action_threshold(cfg, "t95") == 0.5
    assert action_threshold(cfg, "sulfur") > v(cfg["spec"]["risk_alpha"])


def test_level_sd_removes_lab_error_variance_and_uses_steady_running_training_analyses():
    import numpy as np
    from mas.calibrate.risk import sulfur_level_sd
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=24 * 6 * 400, freq="10min")
    running = pd.DataFrame({"ho_running": True, "ho_hours_since_start": (pd.RangeIndex(len(idx)) / 6).to_numpy() + 1000.0}, index=idx)
    n = 4000
    times = idx[rng.choice(len(idx) - 1, n, replace=False)].sort_values()
    level_sd, sigma_lims = 1.5, 0.9
    values = 8.5 + rng.normal(0, level_sd, n) + rng.normal(0, sigma_lims, n)
    lims = _lims(times, values)
    train_end = idx[-1] + pd.Timedelta(minutes=10)
    assert sulfur_level_sd(lims, running, train_end, 30.0, sigma_lims) == pytest.approx(level_sd, rel=0.08)
    assert sulfur_level_sd(lims, running, train_end, 30.0, 100.0) == 0.0     # погрешность больше разброса: нижняя граница нуль


def test_decision_uncertainty_limit_is_taken_from_calibration():
    from mas.common import max_sigma_for_decision
    assert max_sigma_for_decision(load_calib()) > 0
    assert max_sigma_for_decision({"uncertainty": {"sulfur_level_sd": 1.7}}) == 1.7


def test_probability_map_is_monotone_bounded_and_fitted_on_steady_training_analyses():
    import numpy as np
    from mas.calibrate.probability import fit_probability_map
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=24 * 6 * 300, freq="10min")
    s_hat = pd.Series(5.0 + 10.0 * ((np.arange(len(idx)) / 4000.0) % 1.0), index=idx)                # оценка серы равномерно от 5 до 15
    state = pd.DataFrame({"S_hat": s_hat, "S_sigma": 1.5})
    running = pd.DataFrame({"ho_running": True, "ho_hours_since_start": 500.0}, index=idx)
    times = idx[rng.choice(np.arange(20, len(idx) - 1), 800, replace=False)].sort_values()
    latent = s_hat.reindex(times, method="ffill").to_numpy() + rng.normal(0, 1.5, len(times))     # истинная сера отличается от оценки на sigma
    lims = _lims(times, latent)
    result = fit_probability_map(state, lims, running, load_cfg(), idx[-1] + pd.Timedelta(minutes=10), 30.0)
    raw, cal = np.array(result["raw"]), np.array(result["calibrated"])
    assert (np.diff(raw) > 0).all() and (np.diff(cal) >= 0).all() and cal.min() >= 0 and cal.max() <= 1
    assert result["n_analyses"] == 800
    # оценка согласована с меткой (истинная сера отличается от оценки на sigma): отображённая вероятность близка к расчётной
    grid = np.linspace(0.2, 0.8, 13)
    assert np.abs(np.interp(grid, raw, cal) - grid).max() < 0.12


def test_probability_map_ignores_startup_and_later_than_train_end():
    import numpy as np
    from mas.calibrate.probability import fit_probability_map
    idx = pd.date_range("2024-01-01", periods=24 * 6 * 40, freq="10min")
    state = pd.DataFrame({"S_hat": 9.0, "S_sigma": 1.0}, index=idx)
    hours = (pd.RangeIndex(len(idx)) / 6).to_numpy()
    running = pd.DataFrame({"ho_running": True, "ho_hours_since_start": hours}, index=idx)
    times = idx[[100, 1000, 2000, 3000, 4000]]
    lims = _lims(times, [5.0, 12.0, 6.0, 12.0, 4.0])
    train_end = idx[3500]
    result = fit_probability_map(state, lims, running, load_cfg(), train_end, 30.0)
    assert result["n_analyses"] == 3          # анализ в пусковом окне (16.7 ч) и после конца обучения не используются


def test_accepted_probability_level_matches_calibrated_alpha():
    import numpy as np
    from mas.common import accepted_raw_probability, calibrated_probability
    calib = load_calib()
    alpha = v(load_cfg()["spec"]["risk_alpha"])
    level = accepted_raw_probability(calib, alpha)
    assert 0 < level < action_threshold(load_cfg(), "sulfur")             # допустимый уровень ниже порога действия
    assert calibrated_probability(calib, level) <= alpha < calibrated_probability(calib, level + 0.01) + 1e-12
    assert (np.diff(calibrated_probability(calib, np.linspace(0, 1, 101))) >= 0).all()
