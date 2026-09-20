"""Сумма долей, ограничения, режимы и настройки смешения."""
from __future__ import annotations

import pandas as pd
import pytest

from mas.bus import MessageBus
from mas.agents.blending_agent import BlendingAgent

from blending_data import G, K, MAIN, TEST_BLENDING

MAIN_OK = {"sulfur_upper": 8.5, "t95_upper": 345.0, "density": 836.0, "density_age_h": 10,
           "cetane": 53.0, "cetane_age_h": 100}

@pytest.fixture()
def blending_data():
    """Подставляет тестовые данные компонентов в конфигурацию и восстанавливает её после теста."""
    import copy

    from mas.common import load_cfg
    bl = load_cfg()["blending"]
    saved = copy.deepcopy(bl)
    bl.update(copy.deepcopy(TEST_BLENDING))
    yield bl
    bl.clear()
    bl.update(saved)


@pytest.fixture()
def agent(blending_data):
    return BlendingAgent(MessageBus())


@pytest.fixture()
def default_agent():
    return BlendingAgent(MessageBus())


def test_default_configuration_has_no_component_data(default_agent):
    from mas.common import load_cfg
    bl = load_cfg()["blending"]
    assert bl["scenario_components"] == {} and bl["additives"] == {}
    result = default_agent.plan({"main": MAIN_OK, "mode": "scenario"})
    assert result.meta["mode"] == "product_only" and "warning" in result.meta
    assert result.table.iloc[0]["shares"] == {MAIN: 1.0}


def test_shares_sum_to_one_in_all_variants(agent):
    result = agent.plan({"main": MAIN_OK, "mode": "scenario"})
    sums = result.table["shares"].map(lambda s: sum(s.values()))
    assert (sums.sub(1.0).abs() < 1e-6).all()


def test_product_only_gives_exactly_one_variant(agent):
    result = agent.plan({"main": MAIN_OK, "mode": "product_only"})
    assert len(result.table) == 1
    assert result.table.iloc[0]["shares"] == {"ГО ДТ (24-2000)": 1.0}


def test_missing_fresh_analysis_marked_no_data_not_infeasible(agent):
    main = dict(MAIN_OK)
    del main["density"]
    result = agent.plan({"main": main, "mode": "product_only"})
    row = result.table.iloc[0]
    density_check = next(c for c in row["checks"] if c["check"] == "density")
    assert density_check["value"] == "нет данных"
    assert density_check["ok"] is True
    assert row["feasible"]


def test_main_component_alone_when_it_meets_all_norms(agent):
    main = dict(MAIN_OK, cetane=55.0, density=832.0)
    best = agent.plan({"main": main, "mode": "scenario"}).table.iloc[0]
    assert best["shares"]["ГО ДТ (24-2000)"] == pytest.approx(1.0)
    assert sum(best["additives"].values()) == 0.0


def test_minimal_additive_dose_among_equal_main_share(agent):
    table = agent.plan({"main": MAIN_OK, "mode": "scenario"}).table
    feasible = table[table["feasible"]]
    top_share = feasible["main_share"].max()
    at_top = feasible[feasible["main_share"] == top_share]
    assert feasible.iloc[0]["dose_total"] == at_top["dose_total"].min()


def test_stock_limits_share(agent):
    result = agent.plan({"main": MAIN_OK, "mode": "scenario"})
    batch_t = 1000.0
    kerosene_stock = 250.0
    over_stock = result.table[result.table["shares"].map(
        lambda s: s.get("Керосин гидроочищенный", 0) * batch_t > kerosene_stock + 1e-6)]
    assert (~over_stock["feasible"]).all()


@pytest.fixture()
def cfg_edit(blending_data):
    """Правка блока `blending` (после подстановки тестовых данных); восстановление - в `blending_data`."""
    return blending_data


def test_manual_recipe_is_normalised_and_evaluated(agent, cfg_edit):
    cfg_edit["manual_shares"]["value"] = {MAIN: 2.0, K: 1.0, G: 1.0}
    row = agent.plan({"main": MAIN_OK, "mode": "manual"}).table.iloc[0]
    assert row["mode"] == "manual"
    assert row["shares"] == pytest.approx({MAIN: 0.5, K: 0.25, G: 0.25})
    assert len(row["checks"]) == 4  # сера, T95, цетановое число, плотность (нормы по ПТФ нет)


def test_disabled_component_is_excluded(agent, cfg_edit):
    cfg_edit["scenario_components"][K]["enabled"] = False
    table = agent.plan({"main": MAIN_OK, "mode": "scenario"}).table
    assert all(K not in s for s in table["shares"])
    assert (table["shares"].map(lambda s: sum(s.values())).sub(1.0).abs() < 1e-6).all()


def test_share_range_limits_component(agent, cfg_edit):
    cfg_edit["scenario_components"][G]["share_range"] = [0.0, 0.1]
    table = agent.plan({"main": MAIN_OK, "mode": "scenario"}).table
    assert max(s[G] for s in table["shares"]) <= 0.1 + 1e-9


def test_component_property_changes_result(agent, cfg_edit):
    main = dict(MAIN_OK, cetane=48.0)  # ГО ДТ не проходит норму по ЦЧ, компоненты 2-3 и присадка A не хватают
    base = agent.plan({"main": main, "mode": "scenario"}).table.iloc[0]
    assert not base["feasible"]
    cfg_edit["scenario_components"][K]["cetane"] = 70.0  # высокоцетановый керосин делает смесь допустимой
    better = agent.plan({"main": main, "mode": "scenario"}).table.iloc[0]
    assert better["feasible"] and better["shares"][K] > 0


def test_cetane_margin_lowers_mixture_cetane(agent, cfg_edit):
    row = lambda: agent.plan({"main": MAIN_OK, "mode": "manual"}).table.iloc[0]["cetane"]  # noqa: E731
    base = row()
    cfg_edit["rule_margins"]["value"]["cetane"] = 3.0
    assert row() == pytest.approx(base - 2.0)


def test_main_defaults_used_without_fresh_analysis(agent, cfg_edit):
    main = {"sulfur_upper": 8.0, "t95_upper": 345.0}
    cfg_edit["main_defaults"]["value"]["cetane"] = 55.0
    cset = agent.plan({"main": main, "mode": "manual"})
    assert cset.meta["main_sources"]["cetane"] == "допущение"
    assert cset.table.iloc[0]["cetane"] > 50.0


def test_infeasible_fallback_keeps_sulfur_norm_and_lists_failures(agent, cfg_edit):
    for name in (K, G):
        cfg_edit["scenario_components"][name]["cetane"] = 30.0
    for add in cfg_edit["additives"].values():
        add["enabled"] = False
    main = dict(MAIN_OK, cetane=40.0)
    cset = agent.plan({"main": main, "mode": "scenario"})
    first = cset.table.iloc[0]
    assert cset.meta["n_feasible"] == 0 and not first["feasible"]
    assert next(c for c in first["checks"] if c["check"] == "S")["ok"]
    assert not next(c for c in first["checks"] if c["check"] == "cetane")["ok"]


def test_impossible_share_constraints_fall_back_with_warning(agent, cfg_edit):
    cfg_edit["main_share_range"]["value"] = [0.0, 0.2]
    for name in (K, G):
        cfg_edit["scenario_components"][name]["share_range"] = [0.0, 0.2]
    cset = agent.plan({"main": MAIN_OK, "mode": "scenario"})
    assert cset.meta["mode"] == "product_only" and "warning" in cset.meta


def test_unknown_mode_is_rejected(agent):
    with pytest.raises(ValueError):
        agent.plan({"main": MAIN_OK, "mode": "unknown"})


def test_block_lists_recipe_batch_checks_and_alternatives(agent):
    from mas.agents.blending_agent import blend_block
    block = blend_block(agent.plan({"main": MAIN_OK, "mode": "scenario"}))
    assert sum(block["доли"].values()) == pytest.approx(1.0)
    assert sum(block["партия, т"].values()) == pytest.approx(1000.0, abs=0.5)
    assert [c["показатель"] for c in block["проверки"]][0] == "Сера"
    assert len(block["альтернативы"]) == 3 and block["допустимых"] > 3


def test_main_component_stock_limits_its_share(agent, cfg_edit):
    cfg_edit["main_stock_t"]["value"] = 300.0  # партия 1000 т: доля ГО ДТ не выше 30 %
    table = agent.plan({"main": MAIN_OK, "mode": "scenario"}).table
    feasible = table[table["feasible"]]
    assert (feasible["shares"].map(lambda s: s[MAIN]) <= 0.3 + 1e-9).all()


@pytest.fixture()
def grade():
    """Выбор сорта товарного ДТ с восстановлением после теста."""
    from mas.common import load_cfg
    node = load_cfg()["spec"]["product_grade"]
    saved = node["value"]
    yield node
    node["value"] = saved


def test_winter_grade_uses_winter_density_and_cetane_norms(agent, grade):
    main = dict(MAIN_OK, density=805.0, cetane=50.0)
    summer = next(c for c in agent.plan({"main": main, "mode": "product_only"}).table.iloc[0]["checks"] if c["check"] == "density")
    assert not summer["ok"]                                    # 805 < 820 для летнего
    grade["value"] = "winter"
    checks = {c["check"]: c for c in agent.plan({"main": main, "mode": "product_only"}).table.iloc[0]["checks"]}
    assert checks["density"]["ok"]                             # 800-845 для зимнего
    assert checks["cetane"]["ok"]                              # 50 >= 49 для зимнего, но < 51 для летнего


def test_summer_grade_requires_cetane_51(agent, grade):
    main = dict(MAIN_OK, cetane=50.0)
    checks = {c["check"]: c for c in agent.plan({"main": main, "mode": "product_only"}).table.iloc[0]["checks"]}
    assert not checks["cetane"]["ok"]


def test_blend_block_shows_norms_of_selected_grade(grade):
    from mas.agents.blending_agent import blend_block
    agent_ = BlendingAgent(MessageBus())
    grade["value"] = "winter"
    block = blend_block(agent_.plan({"main": MAIN_OK, "mode": "product_only"}))
    norms = {c["показатель"]: c["норма"] for c in block["проверки"]}
    assert norms["Плотность при 15 °C"] == "800 … 845"
    assert norms["Цетановое число"] == "≥ 49"
