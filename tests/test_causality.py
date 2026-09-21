"""Отсутствие утечки из будущего - самая важная группа тестов."""
from __future__ import annotations

import pandas as pd

from mas.common import CACHE_DIR
from mas.prepare.quality_flags import trailing_run_length


def test_trailing_run_length_basic():
    s = pd.Series([1, 1, 1, 2, 2])
    assert list(trailing_run_length(s)) == [1, 2, 3, 1, 2]


def test_freeze_flag_unaffected_by_future():
    idx = pd.date_range("2024-01-01", periods=20, freq="10min")
    values = pd.Series([1.0] * 10 + [2.0] * 10, index=idx)
    t = idx[8]
    run_full = trailing_run_length(values)
    run_prefix = trailing_run_length(values.loc[:t])
    assert run_full.loc[t] == run_prefix.loc[t]
    changed = values.copy()
    changed.iloc[15:] = 99.0  # изменяем данные после t
    run_changed = trailing_run_length(changed)
    assert run_changed.loc[t] == run_full.loc[t]


def test_treq_7d_uses_only_previous_days():
    from mas.common import load_cfg
    from mas.models.catalyst import treq_series
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="1h")
    t5, feed, s_hat, running = (pd.Series(x, index=idx) for x in (370.0, 213.0, 8.0, True))
    calib = {"kinetics": {"beta_f": {"medium": 1.0}, "beta_t": {"medium": -0.03}}, "reference": {"feed_tph": 213.0}}
    cfg = load_cfg()
    base = treq_series(t5, feed, s_hat, running, calib, cfg)["treq_7d"]
    changed = t5.copy()
    changed.loc["2024-01-20":"2024-01-23"] += 10.0
    new = treq_series(changed, feed, s_hat, running, calib, cfg)["treq_7d"]
    assert new.loc["2024-01-20"] == base.loc["2024-01-20"]  # сутки D не видят собственных значений
    assert new.loc["2024-01-24"] != base.loc["2024-01-24"]  # а следующие сутки - видят


def test_lims_available_from_sample_time_and_not_before():
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
    ht2 = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")].sort_values("time")
    row = ht2.iloc[len(ht2) // 2]
    assert row["available_at"] == row["time"]
    from mas.agents.data_agent import DataAgent
    from mas.bus import MessageBus
    agent = DataAgent(MessageBus())
    before = agent._lims_field("HT2", "Mg.Sulfur", row["time"] - pd.Timedelta(seconds=1))
    at = agent._lims_field("HT2", "Mg.Sulfur", row["time"])
    assert before is None or before["sample_time"] < row["time"]
    assert at["sample_time"] == row["time"]  # снимок не видит анализ до его времени и видит с этого момента


def test_snapshot_independent_of_future_telemetry(tmp_path):
    from mas.bus import MessageBus
    from mas.agents.data_agent import DataAgent

    bus = MessageBus()
    agent = DataAgent(bus)
    t = pd.Timestamp("2025-06-01 12:00")
    snap_before = agent.snapshot(t)
    tail_backup = agent.ho.loc[t + pd.Timedelta(hours=1):].copy()
    agent.ho.loc[t + pd.Timedelta(hours=1):, "T5"] = 999.0
    snap_after = agent.snapshot(t)
    agent.ho.loc[tail_backup.index, "T5"] = tail_backup["T5"]
    assert snap_before.values["T5"] == snap_after.values["T5"]
