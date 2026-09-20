"""Индекс тяжести режима: факторы по границам из документов и обучающих данных, без весов."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mas.agents.reliability_agent import ReliabilityAgent
from mas.bus import MessageBus
from mas.common import load_calib, load_cfg, v
from mas.contracts import Snapshot


@pytest.fixture(scope="module")
def agent():
    return ReliabilityAgent(MessageBus())


def _snap(agent, t5=None, treq=np.nan, dp_ratio=1.0, d_t5=0.0, in_band=True) -> Snapshot:
    b = agent.bands
    t5 = b["T5"]["p50"] if t5 is None else t5
    values = {tag: (b[tag]["p5"] + b[tag]["p95"]) / 2 if in_band else b[tag]["p99"] * 2 + 1 for tag in agent._tags() if tag in b}
    values["T5"] = t5
    return Snapshot(t=pd.Timestamp("2025-06-01 12:00"), ho_running=True, avt_running=True, hours_since_start=1000.0,
                    values=values, values_4h_ago={"T5": t5 - d_t5}, sulfur_state={"dp_ratio": dp_ratio},
                    catalyst={"treq_7d": treq})


def test_t5_factor_spans_median_to_p99(agent):
    b = agent.bands["T5"]
    assert agent._t5_factor(b["p50"]) == 0.0
    assert agent._t5_factor(b["p99"]) == 1.0
    assert agent._t5_factor(b["p50"] - 5) == 0.0 and agent._t5_factor(b["p99"] + 5) == 1.0
    mid = agent._t5_factor((b["p50"] + b["p99"]) / 2)
    assert mid == pytest.approx(0.5)


def test_catalyst_factor_uses_cycle_temperatures_from_config(agent):
    cycle = v(load_cfg()["reliability"]["catalyst_cycle_c"])
    assert agent._factors(_snap(agent, treq=cycle["start"]))["catalyst"] == 0.0
    assert agent._factors(_snap(agent, treq=cycle["end"]))["catalyst"] == 1.0
    mid = (cycle["start"] + cycle["end"]) / 2
    assert agent._factors(_snap(agent, treq=mid))["catalyst"] == pytest.approx(0.5)
    assert np.isnan(agent._factors(_snap(agent))["catalyst"])            # Treq неизвестна: фактор не учитывается


def test_severity_is_plain_mean_of_known_factors(agent):
    snap = _snap(agent, t5=agent.bands["T5"]["p99"], treq=v(load_cfg()["reliability"]["catalyst_cycle_c"])["end"])
    factors = agent._factors(snap)
    known = [x for x in factors.values() if pd.notna(x)]
    assert agent.assess(snap).severity == pytest.approx(np.mean(known))
    assert len(known) == 5 - sum(pd.isna(x) for x in factors.values())


def test_severity_classes_are_thirds_of_scale(agent):
    low = agent.assess(_snap(agent))
    assert low.severity_class == "низкий" and low.severity < 1 / 3
    high = agent.assess(_snap(agent, t5=agent.bands["T5"]["p99"], treq=420.0, dp_ratio=10.0, d_t5=100.0, in_band=False))
    assert high.severity_class == "высокий" and high.severity >= 2 / 3
    assert set(high.factors) >= {"T5 близко к верхней границе", "катализатор близок к концу цикла"}


def test_unit_above_t5_corridor_is_not_admissible(agent):
    p99 = agent.bands["T5"]["p99"]
    assert agent.assess(_snap(agent, t5=p99)).admissible
    assert not agent.assess(_snap(agent, t5=p99 + 0.1)).admissible


def test_only_t5_factor_changes_with_candidate_action(agent):
    snap = _snap(agent, t5=agent.bands["T5"]["p50"] + 5, treq=380.0)
    ra = agent.assess(snap)
    table = pd.DataFrame({"T5_new": [snap.values["T5"] - 3, snap.values["T5"], snap.values["T5"] + 3],
                          "F9_new": [snap.values.get("F9", np.nan)] * 3, "F32_new": [snap.values.get("F32", np.nan)] * 3})
    out = agent.evaluate({"snap": snap, "ra": ra, "table": table}).table
    assert out["severity_after"].is_monotonic_increasing
    assert out["severity_after"].iloc[1] == pytest.approx(ra.severity)


def test_calibrated_boundaries_of_factors_are_present():
    rel = load_calib()["reliability"]
    assert rel["dp_ratio_p99"] > 1 and rel["t5_roc4h_p95"] > 0
