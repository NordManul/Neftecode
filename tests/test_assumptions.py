"""Редактируемые допущения интерактивного просмотра: проверка значений, применение и сброс."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from mas.agents.orchestrator import Orchestrator
from mas.common import CACHE_DIR, load_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "live_viewer"))
from assumptions import Assumptions  # noqa: E402

pytestmark = pytest.mark.skipif(not (CACHE_DIR / "state_sulfur.parquet").exists(), reason="нужен кэш: python run.py all")
T = pd.Timestamp("2025-10-17 20:00")


@pytest.fixture(scope="module")
def session():
    import copy

    from blending_data import TEST_BLENDING
    bl = load_cfg()["blending"]
    saved = copy.deepcopy(bl)
    bl.update(copy.deepcopy(TEST_BLENDING))      # компонентов в конфигурации проекта нет: подставляются тестовые
    a = Assumptions(Orchestrator())
    yield a
    a.apply(reset="all")
    bl.clear()
    bl.update(saved)


def _card(a: Assumptions) -> dict:
    rec = a.orch.cycle(T, save=False)
    return {"status": rec.status, "blocks": rec.blocks, "chosen": rec.chosen}


def test_invalid_values_are_rejected(session):
    assert "нижняя" in session.apply({"data_quality/pak_sulfur_valid": [5, 4]})["errors"]["data_quality/pak_sulfur_valid"]
    assert "spec/risk_alpha" in session.apply({"spec/risk_alpha": 0.9})["errors"]       # вне допустимых границ
    assert "неизвестный" in session.apply({"no/such": 1})["errors"]["no/such"]
    assert "не редактируется" in session.apply({"spec/t95_max_c": 400})["errors"]["spec/t95_max_c"]
    assert "неизвестный" in session.apply({"reliability/weights/t5": 5})["errors"]["reliability/weights/t5"]  # весов в системе нет
    assert "неизвестный" in session.apply({"spec/ensemble_weights": [1, 1, 1]})["errors"]["spec/ensemble_weights"]
    assert session.overrides == {}


def test_instant_parameter_changes_card_and_reset_restores_it(session):
    base = _card(session)
    assert session.apply({"process/ht_yield": 0.9, "uncertainty/sulfur_level_sd": 0.2})["recomputed"] is False
    changed = _card(session)
    assert changed != base
    assert load_cfg()["process"]["ht_yield"]["value"] == 0.9
    session.apply(reset="all")
    assert _card(session) == base
    assert load_cfg()["process"]["ht_yield"]["value"] == session.defaults["process/ht_yield"]


def test_instrument_range_recomputes_state_and_reset_restores_cache(session):
    base = _card(session)
    result = session.apply({"data_quality/pak_sulfur_valid": [0.05, 6.0]})
    assert result["ok"] and result["recomputed"]
    assert _card(session)["blocks"]["1. Время и состояние"]["оценка серы"] != base["blocks"]["1. Время и состояние"]["оценка серы"]
    assert session.apply(reset="all")["recomputed"]
    assert _card(session) == base


def test_value_equal_to_default_is_not_an_override(session):
    default = session.defaults["spec/risk_alpha"]
    assert session.apply({"spec/risk_alpha": default})["ok"]
    assert "spec/risk_alpha" not in session.overrides


def test_ensemble_members_are_equally_weighted(session):
    assert session.orch.quality.weights.sum() == pytest.approx(1.0)
    assert len(set(session.orch.quality.weights.round(12))) == 1


from blending_data import G, K, MAIN  # noqa: E402


def _blend(a: Assumptions) -> dict:
    return a.orch.cycle(T, save=False).blocks["8. Блендинг"]


def test_blending_mode_and_manual_recipe_take_effect(session):
    session.apply(reset="all")
    assert _blend(session)["режим"] == "product_only"     # режим по умолчанию
    assert session.apply({"blending/mode": "scenario"})["ok"] and _blend(session)["режим"] == "scenario"
    shares = {f"blending/manual_shares/{MAIN}": 0.6, f"blending/manual_shares/{K}": 0.2, f"blending/manual_shares/{G}": 0.2}
    assert session.apply({"blending/mode": "manual", **shares})["ok"]
    block = _blend(session)
    assert block["режим"] == "manual" and block["доли"][MAIN] == pytest.approx(0.6)
    assert session.apply({"blending/mode": "product_only"})["ok"]
    assert _blend(session)["доли"] == {MAIN: 1.0}
    session.apply(reset="all")


def test_component_switch_and_share_limit_change_recipe(session):
    session.apply(reset="all")
    session.apply({"blending/mode": "scenario"})
    assert session.apply({f"blending/scenario_components/{K}/enabled": False})["ok"]
    assert K not in _blend(session)["доли"]
    session.apply(reset="all")
    session.apply({"blending/mode": "scenario"})
    assert session.apply({f"blending/scenario_components/{G}/share_range": [0.0, 0.1]})["ok"]
    assert _blend(session)["доли"][G] <= 0.1 + 1e-9
    session.apply(reset="all")


def test_blending_validation(session):
    session.apply(reset="all")
    assert "возрастать" in session.apply({"blending/additives/Присадка A (цетаноповышающая)/doses_kg_t": [1, 0.5, 0.2]})["errors"][
        "blending/additives/Присадка A (цетаноповышающая)/doses_kg_t"]
    assert "нижняя" in session.apply({f"blending/scenario_components/{K}/share_range": [0.6, 0.2]})["errors"][
        f"blending/scenario_components/{K}/share_range"]
    assert session.apply({f"blending/scenario_components/{K}/enabled": "yes"})["ok"] is False
    zero = {f"blending/manual_shares/{n}": 0 for n in (MAIN, K, G)}
    assert "blending/mode" in session.apply({"blending/mode": "manual", **zero})["errors"]
    only_main = {f"blending/scenario_components/{n}/share_range": [0.0, 0.1] for n in (K, G)}
    assert "blending/main_share_range" in session.apply({"blending/main_share_range": [0.0, 0.3], **only_main})["errors"]
    assert session.overrides == {}


def test_product_grade_switch_applies_winter_norms_and_reset_restores(session):
    from mas.common import commercial_norms
    assert session.apply({"spec/product_grade": "autumn"})["ok"] is False
    assert session.apply({"spec/product_grade": "winter"})["ok"]
    norms = commercial_norms(load_cfg())
    assert norms["grade"] == "winter" and norms["d15_range"] == [800.0, 845.0] and norms["cetane_min"] == 49.0
    session.apply(reset="all")
    assert commercial_norms(load_cfg())["grade"] == "summer"
