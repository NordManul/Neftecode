"""Датаклассы сообщений между агентами.

Каждый датакласс несёт свой brief() - краткую строку для журнала шины:
это делает журнал читаемым в карточке оператора без разбора JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

STATUS_NO_ACTION = "БЕЗ ИЗМЕНЕНИЙ"
STATUS_CORRECTIVE = "КОРРЕКТИРУЮЩЕЕ ДЕЙСТВИЕ"
STATUS_NO_RELIABLE = "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ"
STATUS_OBSERVE = "НАБЛЮДЕНИЕ (установка не в работе)"
ESCALATION_SUFFIX = " + ЭСКАЛАЦИЯ ТЕХНОЛОГУ"


@dataclass
class Issue:
    severity: str  # info | warning | critical
    source: str    # телеметрия | ПАК | ЛИМС | справочник | модель
    tag: str
    message: str

    def brief(self) -> str:
        return f"[{self.severity}] {self.tag}: {self.message}"


@dataclass
class Snapshot:
    t: pd.Timestamp
    ho_running: bool
    avt_running: bool
    hours_since_start: float
    values: dict[str, float] = field(default_factory=dict)
    values_4h_ago: dict[str, float] = field(default_factory=dict)
    analyzers: dict[str, Any] = field(default_factory=dict)
    sulfur_state: dict[str, float] = field(default_factory=dict)
    t95_samples: list[dict] = field(default_factory=list)
    lims_latest: dict[str, dict] = field(default_factory=dict)
    catalyst: dict[str, float] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def critical(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "critical"]

    def brief(self) -> str:
        state = "работает" if self.ho_running else "не в работе"
        n_crit = len(self.critical())
        s_hat = self.sulfur_state.get("S_hat")
        s_txt = f"S_hat={s_hat:.2f}" if s_hat is not None else "S_hat=?"
        return f"t={self.t} ГО {state}, {s_txt}, критических проблем: {n_crit}"


@dataclass
class QualityAssessment:
    S_now: float
    S_sigma: float
    p_exceed_now: float
    horizon_h: int
    S_forecast: float
    S_forecast_sigma: float
    p_exceed_forecast: float
    t95_est: float
    t95_sigma: float
    p_t95_exceed: float
    specs: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = "низкая"
    warnings: list[str] = field(default_factory=list)
    risk: str = "низкий"

    def brief(self) -> str:
        return (f"S={self.S_now:.2f}±{self.S_sigma:.2f} P(S>10)={self.p_exceed_now:.0%} "
                f"T95={self.t95_est:.0f} P(T95>360)={self.p_t95_exceed:.0%} риск={self.risk}")


@dataclass
class ReliabilityAssessment:
    severity: float
    severity_class: str
    factors: list[str] = field(default_factory=list)
    catalyst: dict = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    step_limits: dict[str, float] = field(default_factory=dict)
    admissible: bool = True

    def brief(self) -> str:
        margin = self.catalyst.get("margin_to_eor_c")
        m_txt = f"{margin:.0f}°C" if margin is not None else "?"
        return f"тяжесть={self.severity:.2f} ({self.severity_class}), запас до EOR={m_txt}, допустим={self.admissible}"


@dataclass
class CandidateSet:
    table: pd.DataFrame
    note: str = ""
    meta: dict = field(default_factory=dict)

    def brief(self) -> str:
        n = len(self.table)
        n_ok = int(self.table["feasible"].sum()) if "feasible" in self.table.columns else n
        return f"вариантов: {n}, допустимых: {n_ok}. {self.note}".strip()


@dataclass
class Recommendation:
    status: str
    t: pd.Timestamp
    blocks: dict[str, Any] = field(default_factory=dict)
    chosen: dict | None = None
    alternatives: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)

    def brief(self) -> str:
        return f"t={self.t} статус: {self.status}"
