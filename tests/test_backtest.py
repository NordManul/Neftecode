"""Сопоставление сигнала о риске на циклах бэктеста с анализами ЛИМС."""
from __future__ import annotations

import pandas as pd
import pytest

from mas.evaluate.backtest import _detection, _failure_reason


def _cycles(flags: list[bool]) -> pd.DataFrame:
    t = pd.date_range("2025-01-01", periods=len(flags), freq="6h")
    status = ["КОРРЕКТИРУЮЩЕЕ ДЕЙСТВИЕ" if f else "БЕЗ ИЗМЕНЕНИЙ" for f in flags]
    return pd.DataFrame({"t": t, "status": status, "P_now": [0.9 if f else 0.1 for f in flags]})


def test_detection_counts_precision_recall_with_lead():
    cycles = _cycles([True, True, False, False, True, False])       # циклы каждые 6 ч
    # анализы через 3 ч после циклов: 1 и 3 превышают, остальные нет
    times = cycles["t"] + pd.Timedelta(hours=3)
    lab = pd.DataFrame({"time": times, "value": [12.0, 8.0, 8.0, 11.0, 12.0, 8.0]})
    row = next(r for r in _detection(cycles, lab, 10.0, 0.5) if r["lead_h"] == 1.0)
    assert row["n_analyses"] == 6
    assert row["share_flagged"] == pytest.approx(3 / 6)
    assert row["precision"] == pytest.approx(2 / 3)                   # сигналы 0, 1, 4: превышения у 0 и 4
    assert row["recall"] == pytest.approx(2 / 3)                      # превышения 0, 3, 4: найдены 0 и 4
    assert row["share_flagged_with_recommendation"] == 1.0


def test_detection_ignores_stopped_unit_and_analyses_before_first_cycle():
    cycles = _cycles([True, True, True])
    cycles.loc[1, "status"] = "НАБЛЮДЕНИЕ (установка не в работе)"
    times = [cycles["t"][0] - pd.Timedelta(hours=1), cycles["t"][1] + pd.Timedelta(hours=3), cycles["t"][2] + pd.Timedelta(hours=3)]
    lab = pd.DataFrame({"time": times, "value": [12.0, 12.0, 8.0]})
    row = next(r for r in _detection(cycles, lab, 10.0, 0.5) if r["lead_h"] == 1.0)
    assert row["n_analyses"] == 1 and row["precision"] == 0.0


def test_lead_excludes_cycles_too_close_to_the_analysis():
    cycles = _cycles([True, False])
    lab = pd.DataFrame({"time": [cycles["t"][1] + pd.Timedelta(minutes=30)], "value": [12.0]})
    by_lead = {r["lead_h"]: r for r in _detection(cycles, lab, 10.0, 0.5)}
    assert by_lead[1.0]["share_flagged"] == 1.0          # для анализа через 30 мин после цикла 6 ч берётся предыдущий цикл (с сигналом)
    assert by_lead[7.0]["n_analyses"] == 0 and by_lead[7.0]["precision"] is None      # до анализа нет цикла с таким упреждением


def test_recheck_refusal_is_a_named_reason():
    assert _failure_reason("Повторная проверка не пройдена: риск по T95 хуже, чем без изменений.") == "повторная проверка"
    assert _failure_reason("Нет допустимого варианта. Причины отсева: риск по сере выше допустимого: 105.") == "нет допустимого варианта"
