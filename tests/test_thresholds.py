"""Пороги, получаемые по данным: метод Оцу, длина заморозки, окно пуска, предельный возраст анализов."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mas.common import load_calib, load_thresholds, startup_window_h
from mas.prepare.thresholds import derive_thresholds, otsu_threshold


def test_otsu_separates_two_regimes():
    rng = np.random.default_rng(0)
    off, on = rng.normal(2.0, 1.0, 500), rng.normal(100.0, 5.0, 1500)
    thr = otsu_threshold(pd.Series(np.concatenate([off, on])))
    assert 10 < thr < 90                      # между режимом останова и рабочим режимом
    assert (off < thr).all() and (on > thr).all()


def test_otsu_ignores_gaps():
    x = pd.Series([1.0, 1.1, np.nan, 50.0, 50.2, np.nan, 0.9, 49.8])
    assert 2 < otsu_threshold(x) < 49


def _frames(freeze_len: int, n: int = 400):
    idx = pd.date_range("2024-01-01", periods=n, freq="10min")
    rng = np.random.default_rng(1)
    ho = pd.DataFrame({"F9": rng.normal(200, 3, n), "T5": rng.normal(370, 2, n)}, index=idx)
    avt = pd.DataFrame({"F65": rng.normal(600, 5, n), "F32": rng.normal(90, 1, n)}, index=idx)
    ho.loc[idx[50:50 + freeze_len], "F9"] = 200.0            # серия одинаковых значений в рабочем режиме
    ho.iloc[300:340] = 0.0                                    # останов: одинаковые нули порогом не считаются
    avt.iloc[300:340] = 0.0
    return avt, ho, idx[-1] + pd.Timedelta(minutes=10)


@pytest.mark.parametrize("freeze_len", [5, 12])
def test_freeze_threshold_exceeds_longest_healthy_run(freeze_len):
    avt, ho, train_end = _frames(freeze_len)
    thr = derive_thresholds(avt, ho, train_end, {"ho": ["F9", "T5"], "avt": ["F32"]})
    assert thr["telemetry_freeze_steps"] == freeze_len + 1
    assert 0 < thr["ho_feed_min_tph"] < 200 and 0 < thr["ho_temp_min_c"] < 370 and 0 < thr["avt_feed_min"] < 600


def test_freeze_threshold_uses_train_period_only():
    avt, ho, train_end = _frames(5)
    ho.iloc[350:380, ho.columns.get_loc("F9")] = 200.0       # длинная серия после обучающего периода
    thr = derive_thresholds(avt, ho, ho.index[200], {"ho": ["F9", "T5"], "avt": ["F32"]})
    assert thr["telemetry_freeze_steps"] == 6


def test_persisted_thresholds_are_consistent():
    thr = load_thresholds()
    assert {"ho_feed_min_tph", "ho_temp_min_c", "avt_feed_min", "telemetry_freeze_steps"} <= set(thr)
    assert thr["telemetry_freeze_steps"] >= 2


def test_startup_window_is_taken_from_calibration():
    assert startup_window_h(load_calib()) > 0
    assert startup_window_h({"startup": {"window_h": 30.0}}) == 30.0


def test_startup_relaxation_recovers_time_constant_of_synthetic_decay():
    from mas.calibrate.startup import startup_relaxation
    rng = np.random.default_rng(3)
    n_runs, run_len, tau = 40, 600, 12.0
    idx = pd.date_range("2023-01-01", periods=n_runs * run_len, freq="1h")
    hours = np.tile(np.arange(run_len, dtype=float), n_runs)
    sign = rng.choice([-1.0, 1.0], len(idx))
    level = 8.0 + sign * (4.0 * np.exp(-hours / tau) + 1.2) + rng.normal(0, 0.3, len(idx))
    analyzer = pd.DataFrame({"raw": level, "reason": ""}, index=idx)
    result = startup_relaxation(analyzer, pd.Series(hours, index=idx), idx[-1] + pd.Timedelta(hours=1))
    assert result["tau_h"] == pytest.approx(tau, rel=0.25)
    assert result["window_h"] == pytest.approx(3 * result["tau_h"])


def test_startup_relaxation_ignores_rejected_readings_and_period_after_training():
    from mas.calibrate.startup import startup_relaxation
    idx = pd.date_range("2023-01-01", periods=6000, freq="1h")
    hours = pd.Series(np.tile(np.arange(300, dtype=float), 20), index=idx)
    raw = 8.0 + 3.0 * np.exp(-hours / 10.0) * np.where(np.arange(len(idx)) % 2 == 0, 1.0, -1.0) + 1.0
    reason = pd.Series("", index=idx)
    reason.iloc[::7] = "заморозка"
    analyzer = pd.DataFrame({"raw": raw, "reason": reason}, index=idx)
    train_end = idx[3000]
    poisoned = analyzer.copy()
    poisoned.loc[poisoned.index >= train_end, "raw"] = 1000.0
    assert startup_relaxation(poisoned, hours, train_end)["tau_h"] == pytest.approx(startup_relaxation(analyzer, hours, train_end)["tau_h"])


def test_lab_age_limits_are_three_median_intervals():
    from mas.common import CACHE_DIR, load_cfg
    calib = load_calib()
    train_end = pd.Timestamp(load_cfg()["periods"]["train_end"])
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
    assert calib["lims_age_limit_h"]
    for key, limit in calib["lims_age_limit_h"].items():
        point, param = key.split(":")
        t = lims[(lims["point"] == point) & (lims["parameter"] == param) & (lims["time"] < train_end)]["time"].sort_values()
        assert limit == pytest.approx(3 * t.diff().dt.total_seconds().median() / 3600)
