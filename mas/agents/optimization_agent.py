"""OptimizationAgent: сетка 105 вариантов, физические показатели и порядок предпочтения, Парето (шаги 5-7).

Порядок предпочтения среди допустимых вариантов лексикографический, без весов: больший выпуск ГО ДТ, при равном выпуске -
меньший размер изменения режима (в шагах сетки). Энергозатраты и старение катализатора не оцениваются: для них в
материалах проекта нет подтверждённых параметров (теплоёмкость потока, температура на входе в печь, закон ускорения
старения). Тяжесть режима после действия входит в критерии Парето.

Оценки качества/надёжности запрашиваются ТОЛЬКО через шину - не вызывает агентов напрямую.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from mas.bus import MessageBus
from mas.common import load_cfg, v
from mas.contracts import CandidateSet


class OptimizationAgent:
    def __init__(self, bus: MessageBus) -> None:
        self.bus, self.cfg = bus, load_cfg()
        dv = self.cfg["decision_vars"]
        self.grid = list(itertools.product(dv["T5"]["grid"], dv["F9"]["grid_rel"], dv["F32"]["grid"]))
        step = lambda values: min(abs(b - a) for a, b in zip(sorted(values), sorted(values)[1:]))  # noqa: E731
        self.steps = (step(dv["T5"]["grid"]), step(dv["F9"]["grid_rel"]), step(dv["F32"]["grid"]))  # шаги сетки
        bus.register("OptimizationAgent", "optimize", lambda payload: self.optimize(payload))

    def _build_table(self, snap) -> pd.DataFrame:
        t5_now, f9_now, f32_now = snap.values["T5"], snap.values["F9"], snap.values["F32"]
        rows = [{"id": f"V{i:03d}", "dT5": dT5, "rF9": rF9, "dF32": dF32,
                 "T5_new": t5_now + dT5, "F9_new": f9_now * (1 + rF9), "F32_new": f32_now + dF32}
                for i, (dT5, rF9, dF32) in enumerate(self.grid)]
        table = pd.DataFrame(rows)
        table["is_no_action"] = (table["dT5"] == 0) & (table["rF9"] == 0) & (table["dF32"] == 0)
        return table

    def _ranking(self, table: pd.DataFrame, snap) -> pd.DataFrame:
        """Выпуск (т/ч), размер изменения (в шагах сетки) и лексикографический порядок `rating` (больше - предпочтительнее)."""
        f9_now = snap.values["F9"]
        s_t5, s_f9, s_f32 = self.steps
        table["d_prod_tph"] = v(self.cfg["process"]["ht_yield"]) * f9_now * table["rF9"]
        table["moves"] = table["dT5"].abs() / s_t5 + table["rF9"].abs() / s_f9 + table["dF32"].abs() / s_f32
        order = table.sort_values(["d_prod_tph", "moves", "id"], ascending=[False, True, True]).index
        table["rating"] = pd.Series(-np.arange(len(order), dtype=float), index=order).reindex(table.index)
        return table

    def _pareto(self, table: pd.DataFrame) -> pd.Series:
        feas = table[table["feasible"]]
        crit = pd.DataFrame({"production": feas["d_prod_tph"], "quality_margin": feas["quality_margin"],
                              "neg_severity": -feas["severity_after"], "neg_moves": -feas["moves"]}).to_numpy()
        is_pareto = np.ones(len(crit), dtype=bool)
        for i in range(len(crit)):
            dominates_i = np.all(crit >= crit[i], axis=1) & np.any(crit > crit[i], axis=1)
            dominates_i[i] = False
            if dominates_i.any():
                is_pareto[i] = False
        pareto = pd.Series(False, index=table.index)
        pareto.loc[feas.index[is_pareto]] = True
        return pareto

    def optimize(self, payload: dict) -> CandidateSet:
        snap, qa, ra = payload["snap"], payload["qa"], payload["ra"]
        table = self._build_table(snap)

        table = self.bus.request("OptimizationAgent", "QualityAgent", "evaluate", {"qa": qa, "table": table},
                                  summary=f"оценка {len(table)} вариантов по качеству").table
        table = self.bus.request("OptimizationAgent", "ReliabilityAgent", "evaluate",
                                  {"snap": snap, "ra": ra, "table": table},
                                  summary=f"оценка {len(table)} вариантов по надёжности").table

        table = self._ranking(table, snap)
        table["feasible"] = table["q_ok_sulfur"] & table["q_ok_t95"] & table["r_ok"]
        table["reject_reason"] = np.where(table["feasible"], "",
                                           np.where(table["q_reason"] != "", table["q_reason"], table["r_reason"]))
        table["pareto"] = self._pareto(table)
        return CandidateSet(table=table, note=f"{len(table)} вариантов, допустимых {int(table['feasible'].sum())}")
