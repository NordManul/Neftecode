"""Проверка модели отклика: ступенчатые изменения и согласие с действиями операторов."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mas.evaluate.effect_check import natural_experiments, operator_concordance

BETA = -0.03


def _synthetic():
    idx = pd.date_range("2024-01-01", periods=24 * 100, freq="1h")
    rng = np.random.default_rng(1)
    t5 = pd.Series(370.0, index=idx)
    for k in range(3, len(idx) - 30, 30):                  # ступень +4 °C на 15 ч через каждые 30 ч
        t5.iloc[k:k + 15] += 4.0
    lag = t5.shift(2).bfill()
    sulfur = pd.Series(8.0 * np.exp(BETA * (lag - 370.0)) * np.exp(rng.normal(0, 0.02, len(idx))), index=idx)
    return idx, t5, pd.Series(250.0, index=idx), sulfur, pd.Series(True, index=idx)


def test_natural_experiments_recover_known_response_and_report_ensemble_members():
    idx, t5, f9, sulfur, running = _synthetic()
    calib = {"kinetics": {"beta_t": {"weak": -0.01, "medium": BETA, "strong": -0.09}, "beta_f": {"weak": 0.5, "medium": 1.0, "strong": 1.5}}}
    rows = natural_experiments(t5, f9, {"синтетика": sulfur}, running, calib, idx[len(idx) // 2], idx[-1])
    temp = [r for r in rows if r["lever"].startswith("T5")]
    assert {r["period"] for r in temp} == {"обучение", "отложенный период"}
    for r in temp:
        assert r["n"] >= 5 and r["lo"] <= r["beta"] <= r["hi"]
        assert r["beta"] == pytest.approx(BETA, rel=0.25)
        assert "средняя" in r["ensemble"] and "сильная" not in r["ensemble"]
    assert not [r for r in rows if r["lever"].startswith("загрузка")]     # изменений загрузки в данных нет


def test_operator_concordance_detects_operator_reaction_after_signal():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="1h")
    times = idx[12:-12:6]
    signal = np.arange(len(times)) % 4 == 0
    t5 = pd.Series(370.0, index=idx)
    for t, s in zip(times, signal):
        if s:
            t5.loc[t + pd.Timedelta("2h"):] += 2.0             # оператор поднимает T5 через 2 ч после сигнала
            t5.loc[t + pd.Timedelta("20h"):] -= 2.0
    cycles = pd.DataFrame({"t": times, "status": np.where(signal, "КОРРЕКТИРУЮЩЕЕ ДЕЙСТВИЕ", "БЕЗ ИЗМЕНЕНИЙ"),
                           "P_now": np.where(signal, 0.9, 0.1)})
    res = operator_concordance(cycles, t5, pd.Series(250.0, index=idx), pd.Series(True, index=idx), 0.5)
    assert res["groups"][0]["n"] == int(signal.sum()) and res["difference"]["d_t5"] > 1.0
    assert res["difference"]["d_t5_ci"][0] > 0                   # интервал разности не содержит нуля
    assert res["difference"]["d_lnf9"] == pytest.approx(0.0, abs=1e-9)
