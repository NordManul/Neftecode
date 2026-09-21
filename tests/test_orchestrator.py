"""Приоритеты, отказы, повторная проверка, карточка."""
from __future__ import annotations

import pandas as pd
import pytest

from mas.agents.orchestrator import Orchestrator
from mas.common import action_threshold, calibrated_probability, load_calib, v
from mas.contracts import QualityAssessment, STATUS_CORRECTIVE, STATUS_NO_ACTION


@pytest.fixture(scope="module")
def orch():
    return Orchestrator()


def _qa(p_exceed_now=0.02, p_t95_exceed=0.02) -> QualityAssessment:
    return QualityAssessment(S_now=8.0, S_sigma=0.6, p_exceed_now=p_exceed_now, horizon_h=4, S_forecast=8.5,
                              S_forecast_sigma=0.6, p_exceed_forecast=0.2, t95_est=345.0,
                              t95_sigma=3.0, p_t95_exceed=p_t95_exceed, specs=[], confidence=1.0,
                              confidence_label="высокая", warnings=[], risk="низкий")


def _table(rows: list[dict]) -> pd.DataFrame:
    base = {"id": "V000", "dT5": 0.0, "rF9": 0.0, "dF32": 0.0, "T5_new": 370.0, "F9_new": 220.0, "F32_new": 90.0,
            "is_no_action": False, "S_mid": 8.0, "S_upper_mix": 9.0, "P_exceed": 0.02, "P_t95": 0.02,
            "quality_margin": 2.0, "q_ok_sulfur": True, "q_ok_t95": True, "r_ok": True, "rating": 0.0,
            "d_prod_tph": 0.0, "severity_after": 0.3, "moves": 0.0, "feasible": True, "reject_reason": ""}
    out = []
    for r in rows:
        row = dict(base)
        row.update(r)
        row["feasible"] = row["q_ok_sulfur"] and row["q_ok_t95"] and row["r_ok"]
        out.append(row)
    return pd.DataFrame(out)


def test_high_rating_cannot_beat_quality_gate(orch):
    table = _table([
        {"id": "V000", "is_no_action": True, "rating": 0.0},
        {"id": "V001", "P_exceed": 0.30, "q_ok_sulfur": False, "rating": 1000.0, "reject_reason": "риск по сере"},
        {"id": "V002", "rating": 5.0},
    ])
    chosen, status, escalate, reason = orch._choose(_qa(), table)
    assert chosen["id"] != "V001"


def test_insensitivity_zone_low_risk_stays_no_action(orch):
    p = 0.3   # выше допустимой (откалиброванной) вероятности, ниже порога действия
    table = _table([{"id": "V000", "is_no_action": True, "P_exceed": p, "rating": 0.0}])
    chosen, status, escalate, reason = orch._choose(_qa(p_exceed_now=p), table)
    assert status == STATUS_NO_ACTION
    assert "наблюдения" in reason


def test_risk_below_neutral_threshold_does_not_trigger_action(orch):
    p = 0.4   # выше допустимой вероятности, но превышение менее вероятно, чем соблюдение
    table = _table([{"id": "V000", "is_no_action": True, "P_exceed": p, "q_ok_sulfur": True, "feasible": True},
                    {"id": "V001", "dT5": 3.0, "P_exceed": 0.03, "rating": 5.0}])
    chosen, status, escalate, reason = orch._choose(_qa(p_exceed_now=p), table)
    assert status == STATUS_NO_ACTION and chosen["id"] == "V000"


def test_risk_above_neutral_threshold_triggers_action(orch):
    p_high = action_threshold(orch.cfg, "sulfur") + 0.05
    table = _table([
        {"id": "V000", "is_no_action": True, "P_exceed": p_high, "q_ok_sulfur": False},
        {"id": "V001", "dT5": 1.0, "T5_new": 371.0, "P_exceed": 0.03, "rating": 2.0},
    ])
    chosen, status, escalate, reason = orch._choose(_qa(p_exceed_now=p_high), table)
    assert status == STATUS_CORRECTIVE
    assert chosen["id"] == "V001"


def test_action_threshold_is_neutral_probability(orch):
    assert action_threshold(orch.cfg, "sulfur") == 0.5 == action_threshold(orch.cfg, "t95")


def test_no_action_is_feasible_below_action_threshold_and_actions_need_alpha(orch):
    snap = orch.bus.request("t", "DataAgent", "snapshot", pd.Timestamp("2025-10-17 20:00"))
    qa = orch.bus.request("t", "QualityAgent", "assess", snap)
    ra = orch.bus.request("t", "ReliabilityAgent", "assess", snap)
    table = orch.bus.request("t", "OptimizationAgent", "optimize", {"snap": snap, "qa": qa, "ra": ra}).table
    alpha = v(orch.cfg["spec"]["risk_alpha"])
    no_action = table[table["is_no_action"]].iloc[0]
    assert bool(no_action["q_ok_sulfur"]) == (no_action["P_exceed"] <= action_threshold(orch.cfg, "sulfur"))
    actions = table[~table["is_no_action"]]
    assert (actions["q_ok_sulfur"] == (actions["P_cal"] <= alpha)).all()
    assert (actions["P_cal"] == calibrated_probability(load_calib(), actions["P_exceed"])).all()


def test_refusal_on_critical_data_problem(orch):
    rec = orch.cycle(pd.Timestamp("2026-07-02 12:00"), save=False)  # сценарий 5
    assert rec.status == "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ"


def test_refusal_no_feasible_variant_has_reference_direction(orch):
    rec = orch.cycle(pd.Timestamp("2025-03-17 12:00"), save=False)  # сценарий 6
    assert rec.status == "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ"
    assert "Справочно" in rec.blocks["7. Объяснение"] or "Причины отсева" in rec.blocks["7. Объяснение"]


def test_refusal_outside_model_scope(orch):
    rec = orch.cycle(pd.Timestamp("2026-04-09 12:00"), save=False)  # сценарий 7
    assert rec.status == "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ"
    assert "модели" in rec.blocks["7. Объяснение"]


def test_second_loop_catches_tampered_choice(orch):
    snap = orch.bus.request("t", "DataAgent", "snapshot", pd.Timestamp("2025-10-17 20:00"))
    qa = orch.bus.request("t", "QualityAgent", "assess", snap)
    bad_chosen = pd.Series({"id": "VFAKE", "is_no_action": False, "S_mid": 50.0, "S_upper_mix": 50.0,
                             "P_exceed": 0.9, "P_cal": 0.9, "P_t95": 0.01, "T95_mid": 340.0, "r_ok": True})
    blend = orch.bus.request("t", "BlendingAgent", "plan", {"main": orch._main_props(snap, bad_chosen), "mode": None})
    ok, reason = orch._recheck(qa, bad_chosen, blend, snap)
    assert not ok


def test_main_recommendation_never_increases_load(orch):
    for ts in ("2025-10-17 20:00", "2025-01-01 14:00", "2026-07-12 08:00"):
        rec = orch.cycle(pd.Timestamp(ts), save=False)
        if rec.chosen is not None:
            assert rec.chosen.get("rF9", 0) <= 1e-9


def test_card_has_all_8_blocks_and_bus_log(orch):
    rec = orch.cycle(pd.Timestamp("2025-10-17 20:00"), save=False)
    for i in range(1, 9):
        assert any(k.startswith(f"{i}.") for k in rec.blocks)
    assert len(rec.trace["bus_log"]) > 0


def test_cetane_of_hydrotreated_diesel_is_not_normed(orch):
    from mas.agents.quality_agent import QualityAgent  # noqa: F401  (агент зарегистрирован в оркестраторе)
    snap = orch.bus.request("t", "DataAgent", "snapshot", pd.Timestamp("2025-10-17 20:00"))
    snap.lims_latest["HT2:CetaneNumber"] = {"value": 40.0, "age_h": 5.0}
    qa = orch.bus.request("t", "QualityAgent", "assess", snap)
    cetane = next(s for s in qa.specs if s["param"] == "CetaneNumber")
    assert cetane["status"] == "не нормируется" and cetane["norm"] is None


def test_explanation_states_why_alternatives_are_worse(orch):
    table = _table([
        {"id": "V000", "is_no_action": True, "d_prod_tph": 0.0, "moves": 0.0},
        {"id": "V001", "d_prod_tph": -5.0, "moves": 2.0},
        {"id": "V002", "d_prod_tph": 0.0, "moves": 3.0},
        {"id": "V003", "d_prod_tph": 0.0, "moves": 1.0, "P_t95": 0.30, "q_ok_t95": True},
    ])
    table["rating"] = [3.0, 0.0, 1.0, 2.0]
    text = orch._explain(table.iloc[0], table, "проверка")
    assert "наибольший выпуск ГО ДТ (+0.0 т/ч" in text and "из 4 допустимых" in text.lower()
    assert "V001 не выбрана: выпуск ниже на 5.0 т/ч" in text
    assert "V002 не выбрана: крупнее изменение режима (3 шаг. сетки против 0)" in text
    assert "выше риск по T95 (30% против 2%)" in text


def test_expected_effect_shows_probability_under_weakest_response(orch):
    rec = orch.cycle(pd.Timestamp("2025-10-22 00:00"), save=False)          # сценарий 2: корректирующее действие
    b4 = rec.blocks["4. Ожидаемый эффект"]
    assert rec.status.startswith(STATUS_CORRECTIVE)
    p_after, _ = b4["P(S>10) c/без действия"]
    assert b4["P(S>10) при самом слабом отклике"] >= p_after           # худший член ансамбля не лучше смеси
