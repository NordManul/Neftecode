"""ReliabilityAgent: тяжесть режима, катализатор, границы для оптимизатора (шаг 4).

Индекс тяжести - среднее пяти факторов, каждый в диапазоне 0-1; весов нет (материалы проекта их не задают), а границы
факторов берутся из документов и обучающих данных:

* близость T5 к верхней границе коридора: от медианы (0) до 99-го перцентиля (1) обучающего периода;
* положение катализатора в цикле: Treq между температурой начала (350 °C) и конца цикла (420 °C), ИТС 30-2021, табл. 2.36;
* доля контролируемых тегов вне коридора 5-95 % обучающего периода;
* быстрота изменения T5: изменение за 4 ч относительно 95-го перцентиля обучающего периода;
* рост приведённого перепада давления в реакторе: от базы (1) до 99-го перцентиля обучающего периода.

Класс тяжести - по третям шкалы 0-1; фактор считается выраженным, если он в верхней трети. Ограничения оптимизатора -
только явные границы: коридоры контролируемых тегов и сетка допустимых изменений `decision_vars`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.bus import MessageBus
from mas.common import load_calib, load_cfg, v
from mas.contracts import CandidateSet, ReliabilityAssessment

FACTOR_NAMES = {"t5": "T5 близко к верхней границе", "catalyst": "катализатор близок к концу цикла",
                "excursion": "много параметров вне коридора", "roc": "быстрое изменение T5",
                "dp": "рост перепада давления в реакторе"}


class ReliabilityAgent:
    def __init__(self, bus: MessageBus) -> None:
        self.cfg, self.calib = load_cfg(), load_calib()
        self.bands, self.rel = self.calib["bands"], self.calib["reliability"]
        bus.register("ReliabilityAgent", "assess", lambda snap: self.assess(snap))
        bus.register("ReliabilityAgent", "evaluate", lambda payload: self.evaluate(payload))

    def _tags(self) -> list[str]:
        monitored = self.cfg["reliability"]["monitored_tags"]
        return [*monitored["ho"], *monitored["avt"]]

    def _excursion(self, values: dict) -> float:
        """Доля контролируемых тегов вне коридора 5-95 % обучающего периода."""
        inside = outside = 0
        for tag in self._tags():
            x, b = values.get(tag), self.bands.get(tag)
            if x is None or b is None or pd.isna(x):
                continue
            outside += int(x < b["p5"] or x > b["p95"])
            inside += 1
        return outside / inside if inside else np.nan

    def _t5_factor(self, t5: float) -> float:
        b = self.bands["T5"]
        return float(np.clip((t5 - b["p50"]) / (b["p99"] - b["p50"]), 0, 1)) if pd.notna(t5) else np.nan

    def _factors(self, snap) -> dict[str, float]:
        cycle = v(self.cfg["reliability"]["catalyst_cycle_c"])
        treq = snap.catalyst.get("treq_7d", np.nan)
        d_t5 = abs(snap.values.get("T5", np.nan) - snap.values_4h_ago.get("T5", np.nan))
        dp_ratio, dp_p99 = snap.sulfur_state.get("dp_ratio", np.nan), self.rel["dp_ratio_p99"]
        return {
            "t5": self._t5_factor(snap.values.get("T5", np.nan)),
            "catalyst": float(np.clip((treq - cycle["start"]) / (cycle["end"] - cycle["start"]), 0, 1)) if pd.notna(treq) else np.nan,
            "excursion": self._excursion(snap.values),
            "roc": float(np.clip(d_t5 / self.rel["t5_roc4h_p95"], 0, 1)) if pd.notna(d_t5) else np.nan,
            "dp": float(np.clip((dp_ratio - 1) / (dp_p99 - 1), 0, 1)) if pd.notna(dp_ratio) and dp_p99 > 1 else np.nan,
        }

    def assess(self, snap) -> ReliabilityAssessment:
        cycle = v(self.cfg["reliability"]["catalyst_cycle_c"])
        t5 = snap.values.get("T5", np.nan)
        t5_max = self.bands["T5"]["p99"]
        factors = self._factors(snap)
        known = [x for x in factors.values() if pd.notna(x)]
        severity = float(np.mean(known)) if known else 0.0
        sev_class = "низкий" if severity < 1 / 3 else ("средний" if severity < 2 / 3 else "высокий")
        shown = [FACTOR_NAMES[k] for k, x in factors.items() if pd.notna(x) and x >= 2 / 3]

        treq = snap.catalyst.get("treq_7d", np.nan)
        dv = self.cfg["decision_vars"]
        bounds = {"T5": (self.bands["T5"]["p1"], t5_max), "F9": (self.bands["F9"]["p5"], self.bands["F9"]["p99"]),
                  "F32": (self.bands["F32"]["p5"], self.bands["F32"]["p99"])}
        step_limits = {"T5_up": float(max(dv["T5"]["grid"])), "T5_down": float(-min(dv["T5"]["grid"]))}
        catalyst = {"treq_7d": treq, "margin_to_eor_c": (cycle["end"] - treq) if pd.notna(treq) else np.nan,
                    "t5_headroom_c": (t5_max - t5) if pd.notna(t5) else np.nan,
                    "dp_ratio": snap.sulfur_state.get("dp_ratio"), "factors": factors,
                    "gor_now": snap.values.get("GOR"), "F2_now": snap.values.get("F2")}
        return ReliabilityAssessment(severity=severity, severity_class=sev_class, factors=shown, catalyst=catalyst,
                                      bounds=bounds, step_limits=step_limits,
                                      admissible=not (pd.notna(t5) and t5 > t5_max))

    def evaluate(self, payload: dict) -> CandidateSet:
        snap, ra, table = payload["snap"], payload["ra"], payload["table"].copy()
        t5_lo, t5_hi = ra.bounds["T5"]
        f9_lo, f9_hi = ra.bounds["F9"]
        f32_lo, f32_hi = ra.bounds["F32"]

        def _in_or_toward(new: pd.Series, cur: float, lo: float, hi: float) -> pd.Series:
            # Вариант, возвращающий параметр в коридор, разрешён, даже если
            # текущее значение вне него; «без изменений» не должен быть заблокирован тем же правилом.
            excursion_cur = max(lo - cur, cur - hi, 0.0)
            excursion_new = (lo - new).clip(lower=0) + (new - hi).clip(lower=0)
            return new.between(lo, hi) | (excursion_new <= excursion_cur)

        bounds_ok = (_in_or_toward(table["T5_new"], snap.values.get("T5", np.nan), t5_lo, t5_hi)
                     & _in_or_toward(table["F9_new"], snap.values.get("F9", np.nan), f9_lo, f9_hi)
                     & _in_or_toward(table["F32_new"], snap.values.get("F32", np.nan), f32_lo, f32_hi))

        # Из пяти факторов индекса действие меняет только положение T5 в коридоре.
        factors = ra.catalyst["factors"]
        n_known = max(sum(pd.notna(x) for x in factors.values()), 1)
        f_now = factors["t5"] if pd.notna(factors["t5"]) else 0.0
        f_new = table["T5_new"].map(self._t5_factor).fillna(f_now)
        table["severity_after"] = ra.severity + (f_new - f_now) / n_known
        table["r_ok"] = bounds_ok & ra.admissible
        table["r_reason"] = np.where(~bounds_ok, "вне модельных границ", "")
        return CandidateSet(table=table, note="проверка коридоров контролируемых параметров")
