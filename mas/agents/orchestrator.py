"""Orchestrator: итоговое решение, объяснение, альтернативы, отказ, карточка (шаг 8).

Лексикографический порядок: данные -> качество -> надёжность -> ранжирование. Основная
рекомендация никогда не увеличивает загрузку. Повторная независимая проверка
не доверяет таблице оптимизатора.
"""
from __future__ import annotations

import json

import pandas as pd

from mas.agents.blending_agent import EMPTY_BLOCK, BlendingAgent, blend_block
from mas.agents.data_agent import DataAgent
from mas.agents.optimization_agent import OptimizationAgent
from mas.agents.quality_agent import QualityAgent
from mas.agents.reliability_agent import ReliabilityAgent
from mas.bus import MessageBus
from mas.common import OUTPUTS_DIR, action_threshold, load_cfg, to_jsonable, v
from mas.contracts import ESCALATION_SUFFIX, STATUS_CORRECTIVE, STATUS_NO_ACTION, STATUS_NO_RELIABLE, STATUS_OBSERVE, Recommendation


class Orchestrator:
    def __init__(self) -> None:
        self.cfg = load_cfg()
        self.bus = MessageBus()
        self._log_start = 0
        self.data = DataAgent(self.bus)
        self.quality = QualityAgent(self.bus)
        self.reliability = ReliabilityAgent(self.bus)
        self.optimization = OptimizationAgent(self.bus)
        self.blending = BlendingAgent(self.bus)

    def _choose(self, qa, table):
        risk_alpha = v(self.cfg["spec"]["risk_alpha"])
        trigger_s, trigger_t95 = action_threshold(self.cfg, "sulfur"), action_threshold(self.cfg, "t95")
        feasible = table[table["feasible"]]
        no_action = table[table["is_no_action"]].iloc[0]

        p_t95_base = no_action["P_t95"] if no_action["feasible"] else qa.p_t95_exceed
        if pd.notna(p_t95_base) and p_t95_base > trigger_t95:
            pool = feasible[feasible["q_ok_sulfur"]]
            pool = pool if not pool.empty else feasible
            idx = pool.sort_values(["P_t95", "rating"], ascending=[True, False]).index[0]
            chosen = table.loc[idx]
            status = STATUS_NO_ACTION if chosen["is_no_action"] else STATUS_CORRECTIVE
            return chosen, status, True, "риск по T95 выше порога действия"

        p_s_base = no_action["P_exceed"] if no_action["feasible"] else qa.p_exceed_forecast
        if (pd.notna(p_s_base) and p_s_base > trigger_s) or not no_action["r_ok"]:
            idx = feasible["rating"].idxmax()
            return table.loc[idx], STATUS_CORRECTIVE, False, "риск по сере выше порога действия либо режим вне границ надёжности"

        watch = pd.notna(p_s_base) and risk_alpha < p_s_base <= trigger_s
        return no_action, STATUS_NO_ACTION, False, ("риск в зоне наблюдения, контролировать следующий анализ" if watch
                                                     else "текущий режим допустим, риск ниже порога действия")

    def _explain(self, chosen, table, reason: str) -> str:
        parts = [f"Выбран вариант {chosen['id']} ({reason})."]
        alts = table[table["feasible"] & (table["id"] != chosen["id"])].sort_values("rating", ascending=False).head(3)
        for _, alt in alts.iterrows():
            why = []
            if alt["quality_margin"] < chosen["quality_margin"]:
                why.append("меньший запас по качеству")
            if alt["P_t95"] > chosen["P_t95"]:
                why.append("выше риск по T95")
            parts.append(f"Альтернатива {alt['id']} не выбрана: {', '.join(why) or 'ниже в порядке предпочтения'}.")
        return " ".join(parts)

    def _main_props(self, snap, chosen) -> dict:
        # T95 - центральная оценка: жёсткий порог не применяется буквально (правило «не навреди»,
        # допустимость уже проверена вероятностно через P_t95.
        # Сера «без изменений» - тоже центральная оценка (для варианта без изменений
        # проверяется центральная оценка) - иначе зона наблюдения (5%<P<=10%) всегда
        # проваливала бы проверку смеси, хотя сама зона существует именно чтобы не отказывать здесь.
        sulfur_upper = chosen["S_mid"] if bool(chosen["is_no_action"]) else chosen["S_upper_mix"]
        props = {"sulfur_upper": sulfur_upper, "t95_upper": chosen["T95_mid"]}
        for key, lims_key in (("density", "HT2:D15"), ("cetane", "HT2:CetaneNumber")):
            f = snap.lims_latest.get(lims_key)
            if f:
                props[key], props[f"{key}_age_h"] = f["value"], f["age_h"]
        return props

    def _recheck(self, qa, chosen, blend, snap) -> tuple[bool, str]:
        risk_alpha, trigger_s = v(self.cfg["spec"]["risk_alpha"]), action_threshold(self.cfg, "sulfur")
        s_limit = v(self.cfg["spec"]["sulfur_max_mgkg"])
        no_action = bool(chosen["is_no_action"])
        if no_action:
            if not chosen["S_mid"] <= s_limit or not chosen["P_exceed"] <= trigger_s:
                return False, "сера (центр/шанс-ограничение при пороге действия) не проходит"
        else:
            if not (chosen["S_mid"] <= s_limit and chosen["S_upper_mix"] <= s_limit):
                return False, "сера по центральной или верхней оценке превышает норму"
            if not chosen["P_exceed"] <= risk_alpha:
                return False, "шанс-ограничение по сере не выполнено"
        if not (chosen["P_t95"] <= risk_alpha or chosen["P_t95"] <= qa.p_t95_exceed):
            return False, "риск по T95 хуже, чем без изменений"
        if not bool(chosen["r_ok"]):
            return False, "параметры вне модельных границ, шагов или соотношения газ/сырьё"
        row = blend.table.iloc[0]
        if abs(sum(row["shares"].values()) - 1.0) > 1e-6:
            return False, "сумма долей блендинга не равна 100%"
        # ЦЧ/D15/ПТФ - по свежим анализам, эскалация, а не блокировка; T95 уже проверен
        # вероятностно выше (правило «не навреди») - здесь жёстко блокирует только сера.
        if not next(c["ok"] for c in row["checks"] if c["check"] == "S"):
            return False, "спецификация смеси по сере не выполнена"
        if snap.critical() or snap.sulfur_state.get("S_sigma", 0) > v(self.cfg["sensor_fusion"]["max_sigma_for_decision"]):
            return False, "данных недостаточно для решения"
        return True, ""

    def _alternatives(self, table, chosen) -> list[dict]:
        feasible = table[table["feasible"] & (table["id"] != chosen["id"])]
        return feasible[feasible["pareto"]].sort_values("rating", ascending=False).head(3).to_dict("records")

    def _blocks(self, t, snap, qa, ra, chosen, blend, escalations: list[str]) -> dict:
        b1 = {"время": str(t), "ГО в работе": snap.ho_running, "АВТ в работе": snap.avt_running,
              "часов с пуска": snap.hours_since_start, "T5": snap.values.get("T5"), "F9": snap.values.get("F9"),
              "F32": snap.values.get("F32"), "оценка серы": snap.sulfur_state.get("S_hat"),
              "сигма серы": snap.sulfur_state.get("S_sigma"),
              "ПАК": {"значение": snap.analyzers["pak"]["raw"], "смещение": snap.sulfur_state.get("bias_pak"),
                      "статус": snap.analyzers["pak"]["reason"] or "ок"},
              "Q21": {"значение": snap.analyzers["q21"]["raw"], "смещение": snap.sulfur_state.get("bias_q21"),
                      "статус": snap.analyzers["q21"]["reason"] or "ок"},
              "возраст ЛИМС серы, ч": (snap.lims_latest.get("HT2:Mg.Sulfur") or {}).get("age_h")}
        b2 = {"риск": qa.risk if qa else None, "тяжесть режима": ra.severity_class if ra else None,
              "обнаружено": [i.message for i in snap.issues], "факторы надёжности": ra.factors if ra else [],
              "эскалации": escalations}
        if chosen is None or bool(chosen.get("is_no_action", True)):
            b3 = "Изменений не требуется"
        else:
            specs = [("T5", "Температура ГСС на выходе Р-201", "°C", snap.values.get("T5"), chosen["T5_new"]),
                     ("F9", "Расход сырья на установку", "т/ч", snap.values.get("F9"), chosen["F9_new"]),
                     ("F32", "Отбор фр. 240-290 °C с АВТ", "т/ч", snap.values.get("F32"), chosen["F32_new"])]
            b3 = [{"tag": tg, "desc": d, "unit": u, "current": c, "recommended": n, "change": n - c}
                  for tg, d, u, c, n in specs if abs(n - c) > 1e-6]
        b4 = {} if chosen is None else {
            "сера центр/верх": (chosen["S_mid"], chosen["S_upper_mix"]),
            "P(S>10) c/без действия": (chosen["P_exceed"], qa.p_exceed_forecast),
            "T95 центр/верх": (chosen["T95_mid"], chosen["T95_upper_worst"]),
            "P(T95>360) c/без действия": (chosen["P_t95"], qa.p_t95_exceed),
            "выпуск, т/ч": chosen["d_prod_tph"], "тяжесть после": chosen["severity_after"],
            "тепловой proxy, МВт": chosen["energy_proxy_mw"],
            "запаздывание эффекта": "сера 2-8 ч, T95 4 ч", "место в порядке предпочтения": int(-chosen["rating"]) + 1}
        b5 = [] if chosen is None else [
            {"check": "сера <= 10", "value": chosen["S_upper_mix"], "ok": bool(chosen["q_ok_sulfur"])},
            {"check": "шанс-ограничение", "value": chosen["P_exceed"], "ok": bool(chosen["q_ok_sulfur"])},
            {"check": "T95 не навреди", "value": chosen["P_t95"], "ok": bool(chosen["q_ok_t95"])},
            {"check": "границы контролируемых параметров", "value": None, "ok": bool(chosen["r_ok"])}]
        b6 = {"метка": "-", "балл": None, "предупреждения": []} if qa is None else \
            {"метка": qa.confidence_label, "балл": qa.confidence, "предупреждения": qa.warnings}
        b8 = blend_block(blend) if blend is not None and len(blend.table) else dict(EMPTY_BLOCK)
        return {"1. Время и состояние": b1, "2. Проблема / риск": b2, "3. Предлагаемое действие": b3,
                "4. Ожидаемый эффект": b4, "5. Проверка ограничений": b5, "6. Уверенность": b6,
                "7. Объяснение": "", "8. Блендинг": b8}

    def _finish(self, status, t, snap, qa=None, ra=None, cands=None, blend=None, alternatives=None,
                explanation="", save=True, tag=None, chosen=None) -> Recommendation:
        escalations = [explanation] if status.endswith(ESCALATION_SUFFIX) else []
        blocks = self._blocks(t, snap, qa, ra, chosen, blend, escalations)
        blocks["7. Объяснение"] = explanation
        trace = {"snapshot": snap.brief(), "issues": [i.brief() for i in snap.issues],
                 "quality": qa.brief() if qa else None, "reliability": ra.brief() if ra else None,
                 "candidates": cands.brief() if cands else None, "bus_log": self.bus.log[self._log_start:]}
        chosen_dict = None if chosen is None else {k: chosen[k] for k in chosen.index if k not in ()}
        rec = Recommendation(status=status, t=t, blocks=blocks, chosen=chosen_dict,
                              alternatives=alternatives or [], trace=trace)
        if save:
            self._save(rec, tag)
        return rec

    def _save(self, rec: Recommendation, tag: str | None) -> None:
        from mas.report.card import card_markdown
        name = tag or rec.t.strftime("%Y%m%d_%H%M")
        out_dir = OUTPUTS_DIR / "cycles"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(to_jsonable({"status": rec.status, "t": rec.t, "blocks": rec.blocks, "chosen": rec.chosen,
                                    "alternatives": rec.alternatives, "trace": rec.trace}), f, ensure_ascii=False, indent=2)
        with open(out_dir / f"{name}.md", "w", encoding="utf-8") as f:
            f.write(card_markdown(rec))

    def cycle(self, t, save: bool = True, tag: str | None = None, blend_mode: str | None = None,
              overrides: dict[str, float] | None = None) -> Recommendation:
        t, b = pd.Timestamp(t), self.bus
        self._log_start = len(b.log)  # журнал карточки - только за этот цикл, не накопленный
        snap = b.request("Orchestrator", "DataAgent", "snapshot", t)
        if overrides:  # сценарный расчёт «что если» - не часть рекомендации
            snap.values.update(overrides)
        if not snap.ho_running:
            return self._finish(STATUS_OBSERVE, t, snap, explanation="Установка не в работе.", save=save, tag=tag)

        qa = b.request("Orchestrator", "QualityAgent", "assess", snap)
        ra = b.request("Orchestrator", "ReliabilityAgent", "assess", snap)

        crit, max_sigma = snap.critical(), v(self.cfg["sensor_fusion"]["max_sigma_for_decision"])
        if crit or snap.sulfur_state.get("S_sigma", 0) > max_sigma:
            reasons = [i.message for i in crit] or [f"Неопределённость серы σ={snap.sulfur_state.get('S_sigma', 0):.2f} выше порога."]
            startup_or_transient = bool(crit) and all(i.tag in ("hours_since_start", "T5") for i in crit)
            direction = "Дождаться установившегося режима." if startup_or_transient else "Восстановить данные или провести внеочередной анализ."
            return self._finish(STATUS_NO_RELIABLE, t, snap, qa, ra, explanation=" ".join(reasons) + " " + direction,
                                 save=save, tag=tag)

        cands = b.request("Orchestrator", "OptimizationAgent", "optimize", {"snap": snap, "qa": qa, "ra": ra})
        table = cands.table

        if not ra.admissible:
            explanation = (f"Режим вне области применимости модели: T5={snap.values.get('T5', float('nan')):.1f} °C "
                            f"выше верхней границы коридора {ra.bounds['T5'][1]:.1f} °C (99-й перцентиль обучающего периода). "
                            f"Эскалация технологу: состояние катализатора и режим печи.")
            return self._finish(STATUS_NO_RELIABLE, t, snap, qa, ra, cands, explanation=explanation, save=save, tag=tag)

        if not table["feasible"].any():
            reasons = "; ".join(f"{k}: {n}" for k, n in table.loc[table["reject_reason"] != "", "reject_reason"].value_counts().items())
            pool = table[table["r_ok"]]
            hint = ""
            if not pool.empty:
                row = pool.loc[(pool["P_exceed"].fillna(1) + pool["P_t95"].fillna(1)).idxmin()]
                hint = (f" Справочно (не рекомендация): ΔT5={row['dT5']:+.0f} °C, ΔF9={row['rF9']:+.0%}, "
                        f"ΔF32={row['dF32']:+.1f} т/ч снижает риск сильнее прочих.")
            return self._finish(STATUS_NO_RELIABLE, t, snap, qa, ra, cands,
                                 explanation=f"Нет допустимого варианта. Причины отсева: {reasons}.{hint}", save=save, tag=tag)

        chosen, status, escalate, reason = self._choose(qa, table)
        explanation = self._explain(chosen, table, reason)
        blend = b.request("Orchestrator", "BlendingAgent", "plan", {"main": self._main_props(snap, chosen), "mode": blend_mode})

        ok, fail_reason = self._recheck(qa, chosen, blend, snap)
        if not ok:
            return self._finish(STATUS_NO_RELIABLE, t, snap, qa, ra, cands, blend,
                                 explanation=f"Повторная проверка не пройдена: {fail_reason}.", save=save, tag=tag)

        uncontrolled_bad = any(s["status"] == "нарушение" for s in qa.specs if s["param"] in ("D15", "FlashPoint"))
        if not escalate and (pd.notna(qa.p_t95_exceed) and qa.p_t95_exceed > action_threshold(self.cfg, "t95") or uncontrolled_bad):
            status, escalate = status + ESCALATION_SUFFIX, True
        elif escalate:
            status += ESCALATION_SUFFIX
        alternatives = self._alternatives(table, chosen)
        return self._finish(status, t, snap, qa, ra, cands, blend, alternatives, explanation, save, tag, chosen)
