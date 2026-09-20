"""Сетка вариантов и порядок предпочтения: лексикографический, без весов и стоимостных показателей."""
from __future__ import annotations

import pandas as pd
import pytest

from mas.agents.optimization_agent import OptimizationAgent
from mas.bus import MessageBus
from mas.common import load_cfg, v
from mas.contracts import Snapshot


@pytest.fixture(scope="module")
def agent():
    return OptimizationAgent(MessageBus())


@pytest.fixture(scope="module")
def snap():
    return Snapshot(t=pd.Timestamp("2025-06-01 12:00"), ho_running=True, avt_running=True, hours_since_start=500.0,
                    values={"T5": 370.0, "F9": 220.0, "F32": 90.0})


def test_grid_is_full_product_of_configured_steps(agent, snap):
    dv = load_cfg()["decision_vars"]
    table = agent._build_table(snap)
    assert len(table) == len(dv["T5"]["grid"]) * len(dv["F9"]["grid_rel"]) * len(dv["F32"]["grid"]) == 105
    assert table["is_no_action"].sum() == 1
    assert (table["rF9"] <= 0).all()               # загрузка не увеличивается


def test_ranking_prefers_larger_output_then_smaller_change(agent, snap):
    table = agent._ranking(agent._build_table(snap), snap)
    assert set(table.columns) >= {"d_prod_tph", "moves", "rating"}
    best = table.loc[table["rating"].idxmax()]
    assert best["is_no_action"]                     # выпуск не снижен и изменений нет
    assert table["rating"].is_unique
    reduced = table[table["rF9"] < 0]
    same_output = table[table["rF9"] == 0]
    assert same_output["rating"].min() > reduced["rating"].max()     # выпуск важнее размера изменения
    within = same_output.sort_values(["moves", "id"])
    assert within["rating"].is_monotonic_decreasing                    # при равном выпуске меньше изменение лучше


def test_output_change_uses_yield_from_config(agent, snap):
    table = agent._ranking(agent._build_table(snap), snap)
    row = table[(table["rF9"] == -0.10) & (table["dT5"] == 0) & (table["dF32"] == 0)].iloc[0]
    assert row["d_prod_tph"] == pytest.approx(v(load_cfg()["process"]["ht_yield"]) * 220.0 * -0.10)


def test_moves_are_counted_in_grid_steps(agent, snap):
    table = agent._ranking(agent._build_table(snap), snap)
    row = table[(table["dT5"] == 2) & (table["rF9"] == -0.05) & (table["dF32"] == -5.0)].iloc[0]
    assert row["moves"] == pytest.approx(2 / 1 + 0.05 / 0.05 + 5.0 / 2.5)
