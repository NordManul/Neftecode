"""BlendingAgent: план смешения, доли компонентов, присадки, спецификация смеси (шаг 8).

Режимы: `product_only` - товарное ДТ равно ГО ДТ (по умолчанию); `scenario` - подбор допустимой рецептуры с наибольшей
долей ГО ДТ и наименьшими дозами присадок; `manual` - расчёт заданной рецептуры. Свойства компонентов 2-3, присадок,
запасы и партия в материалах проекта не заданы, поэтому в конфигурации (`blending`) их нет: пока `scenario_components`
и `additives` пусты, любой режим сводится к `product_only`. При заполнении паспортными данными нужны также `share_step`,
`batch_t`, `rule_margins`, `main_stock_t`, `main_share_range`, `main_defaults` (свойства ГО ДТ при отсутствии свежего
анализа) и, для режима `manual`, `manual_shares`, `manual_doses`. Нормы товарного ДТ - по выбранному сорту
(`mas.common.commercial_norms`); норма по ПТФ в материалах проекта не задана и не проверяется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.bus import MessageBus
from mas.common import commercial_norms, lims_age_limit_h, load_calib, load_cfg, v
from mas.contracts import CandidateSet
from mas.models.blend import Additive, Component, Limits, best_first, dose_grid, evaluate, share_grid

MODES = ("product_only", "scenario", "manual")
HARD_LIMITS = [("S", "sulfur_max_mgkg", "<="), ("T95", "t95_max_c", "<="), ("cetane", "cetane_min", ">=")]
MAIN_PROPS = (("D15", "density", "D15"), ("cetane", "cetane", "CetaneNumber"))  # свойство, ключ, параметр ЛИМС ГО т.2
LABELS = {"S": ("Сера", "мг/кг"), "T95": ("T95", "°C"), "cetane": ("Цетановое число", ""),
          "CFPP": ("ПТФ", "°C"), "density": ("Плотность при 15 °C", "кг/м³")}
PROP_NAMES = {"S": "сера", "T95": "T95", "D15": "плотность", "cetane": "цетановое число", "CFPP": "ПТФ"}
SCENARIO_KEYS = ("share_step", "batch_t", "rule_margins", "main_stock_t", "main_share_range")
EMPTY_BLOCK = {"режим": "-", "доли": {}, "присадки": {}, "свойства": {}, "проверки": [], "не выполнено": [],
               "вариантов": 0, "допустимых": 0}


class BlendingAgent:
    def __init__(self, bus: MessageBus) -> None:
        self.cfg, self.calib = load_cfg(), load_calib()
        bus.register("BlendingAgent", "plan", lambda payload: self.plan(payload))

    def _checks(self, props: dict) -> list[dict]:
        spec, norms, checks = self.cfg["spec"], commercial_norms(self.cfg), []
        for key, cfg_key, op in HARD_LIMITS:
            val, norm_val = props.get(key), norms[cfg_key] if cfg_key in norms else v(spec[cfg_key])
            if val is None:
                checks.append({"check": key, "value": "нет данных", "ok": True})
            else:
                checks.append({"check": key, "value": val, "ok": bool(val <= norm_val if op == "<=" else val >= norm_val)})
        d15, lo_hi = props.get("density"), norms["d15_range"]
        checks.append({"check": "density", "value": d15 if d15 is not None else "нет данных",
                       "ok": True if d15 is None else bool(lo_hi[0] <= d15 <= lo_hi[1])})
        return checks

    def _fresh(self, main: dict, key: str, param: str) -> float | None:
        """Значение свойства ГО ДТ по последнему анализу ЛИМС, если он не старше предельного возраста."""
        limit = lims_age_limit_h(self.calib, "HT2", param)
        return main.get(key) if main.get(f"{key}_age_h", np.inf) <= limit else None

    def _product_only(self, main: dict) -> pd.DataFrame:
        props = {"S": main.get("sulfur_upper"), "T95": main.get("t95_upper")}
        props.update({("density" if k == "D15" else k): self._fresh(main, key, param) for k, key, param in MAIN_PROPS})
        checks = self._checks(props)
        name = v(self.cfg["blending"]["main_component"])
        row = {"id": "B000", "shares": {name: 1.0}, "additives": {}, **props, "checks": checks,
               "feasible": all(c["ok"] for c in checks), "dose_total": 0.0}
        return pd.DataFrame([row])

    def _required(self, *keys: str) -> None:
        bl = self.cfg["blending"]
        missing = [k for k in keys if k not in bl]
        if missing:
            raise ValueError("для расчёта смеси в config/settings.yaml (blending) не заданы: " + ", ".join(missing))

    def _main_component(self, main: dict) -> tuple[Component | None, dict]:
        """ГО ДТ: сера и T95 - из расчёта цикла, плотность и цетановое число - свежий анализ ЛИМС либо `main_defaults`."""
        bl = self.cfg["blending"]
        defaults = v(bl["main_defaults"]) if "main_defaults" in bl else {}
        lo_hi = v(bl["main_share_range"])
        vals, sources = {}, {"S": "прогноз цикла", "T95": "прогноз цикла"}
        for prop, key, param in MAIN_PROPS:
            fresh = self._fresh(main, key, param)
            if fresh is not None:
                vals[prop], sources[prop] = fresh, "ЛИМС"
            elif prop in defaults:
                vals[prop], sources[prop] = defaults[prop], "допущение"
            else:
                return None, sources
        # ПТФ ГО ДТ не используется (нормы по ПТФ нет); значение нужно только модели смешения
        comp = Component(v(bl["main_component"]), main["sulfur_upper"], vals["D15"], vals["cetane"], main["t95_upper"],
                         defaults.get("CFPP", 0.0), v(bl["main_stock_t"]), lo_hi[0], lo_hi[1])
        return comp, sources

    def _components(self, main: dict) -> tuple[list[Component], dict]:
        comp, sources = self._main_component(main)
        if comp is None:
            return [], sources
        comps = [comp]
        for name, p in self.cfg["blending"]["scenario_components"].items():
            if p.get("enabled", True):
                lo, hi = p.get("share_range", [0.0, 1.0])
                comps.append(Component(name, p["S"], p["D15"], p["cetane"], p["T95"], p.get("CFPP", 0.0), p["stock_t"], lo, hi))
        return comps, sources

    def _additives(self) -> list[Additive]:
        return [Additive(name, p["param"], tuple(v(p["doses_kg_t"])), v(p["effect_per_kg_t"]))
                for name, p in self.cfg["blending"]["additives"].items() if p.get("enabled", True)]

    def _limits(self) -> Limits:
        s, norms = self.cfg["spec"], commercial_norms(self.cfg)
        return Limits(v(s["sulfur_max_mgkg"]), v(s["t95_max_c"]), norms["cetane_min"], None, tuple(norms["d15_range"]))

    def _mixtures(self, main: dict, mode: str) -> tuple[pd.DataFrame, dict]:
        bl = self.cfg["blending"]
        self._required(*SCENARIO_KEYS)
        comps, sources = self._components(main)
        if not comps:
            return self._product_only(main), {"warning": "нет свежего анализа свойств ГО ДТ для расчёта смеси: показан режим product_only"}
        adds = self._additives()
        if mode == "manual":
            self._required("manual_shares", "manual_doses")
            weights = np.array([max(0.0, v(bl["manual_shares"]).get(c.name, 0.0)) for c in comps])
            shares = (weights / weights.sum())[None, :] if weights.sum() > 0 else None
            doses = np.array([[max(0.0, v(bl["manual_doses"]).get(a.name, 0.0)) for a in adds]], dtype=float).reshape(1, len(adds))
        else:
            grid, dgrid = share_grid(comps, v(bl["share_step"])), dose_grid(adds)
            shares = np.repeat(grid, len(dgrid), axis=0) if len(grid) else None
            doses = np.tile(dgrid, (len(grid), 1))
        if shares is None:
            return self._product_only(main), {"warning": "нет допустимых сочетаний долей: показан режим product_only"}
        table = evaluate(comps, adds, shares, doses, margins=v(bl["rule_margins"]), batch_t=v(bl["batch_t"]),
                         limits=self._limits())
        return best_first(table), {"main_sources": sources}

    def plan(self, payload: dict) -> CandidateSet:
        bl = self.cfg["blending"]
        main = payload["main"]
        mode = payload.get("mode") or v(bl["mode"])
        if mode not in MODES:
            raise ValueError(f"неизвестный режим блендинга: {mode}")
        no_components = not bl["scenario_components"] and not bl["additives"]
        if mode == "product_only" or no_components:
            table, meta = self._product_only(main), {}
            if mode != "product_only":
                meta["warning"] = "компоненты и присадки в конфигурации не заданы: показан режим product_only"
        else:
            table, meta = self._mixtures(main, mode)
        table["mode"] = mode if "warning" not in meta else "product_only"
        meta.update(mode=table["mode"].iloc[0], n_feasible=int(table["feasible"].sum()))
        if "batch_t" in bl:
            meta["batch_t"] = v(bl["batch_t"])
        return CandidateSet(table=table, note=f"режим {meta['mode']}, вариантов {len(table)}", meta=meta)


def _limit_text(key: str, spec: dict, norms: dict) -> str:
    if key == "density":
        lo, hi = norms["d15_range"]
        return f"{lo:g} … {hi:g}"
    cfg_key, op = next((c, o) for k, c, o in HARD_LIMITS if k == key)
    return f"{'≤' if op == '<=' else '≥'} {(norms[cfg_key] if cfg_key in norms else v(spec[cfg_key])):g}"


def blend_block(blend: CandidateSet) -> dict:
    """Блок 8 карточки: рецептура, свойства смеси относительно норм, альтернативы (партия - если задана)."""
    cfg = load_cfg()
    spec, norms, row, meta = cfg["spec"], commercial_norms(cfg), blend.table.iloc[0], blend.meta
    checks = [{"показатель": LABELS[c["check"]][0], "единицы": LABELS[c["check"]][1], "значение": c["value"],
               "норма": _limit_text(c["check"], spec, norms), "ok": bool(c["ok"])} for c in row["checks"]]
    # альтернативы - рецептуры с другими долями (для каждой - вариант с наименьшей дозой присадок), в порядке предпочтения
    seen, alts = {tuple(row["shares"].values())}, []
    for _, cand in blend.table.iloc[1:][lambda t: t["feasible"] & (t["mode"] == "scenario")].iterrows():
        shares_key = tuple(cand["shares"].values())
        if shares_key not in seen and len(alts) < 3:
            seen.add(shares_key)
            alts.append(cand)
    batch = meta.get("batch_t")
    order = lambda shares: [{"компонент": k, "доля": x, **({"партия, т": round(x * batch, 1)} if batch else {})}  # noqa: E731
                            for k, x in shares.items()]
    doses = lambda adds: [{"присадка": k, "доза, кг/т": d} for k, d in adds.items()]  # noqa: E731
    return {"режим": row["mode"], "доли": row["shares"], "состав": order(row["shares"]), "дозы": doses(row["additives"]),
            **({"партия, т": {k: round(x * batch, 1) for k, x in row["shares"].items()}} if batch else {}),
            "присадки": row["additives"], "свойства": {k: row.get(k) for k in ("S", "T95", "cetane", "density")},
            "проверки": checks, "не выполнено": [c["показатель"] for c in checks if not c["ok"]],
            "вариантов": len(blend.table), "допустимых": meta.get("n_feasible", 0),
            "источники свойств ГО ДТ": [{"свойство": PROP_NAMES[k], "источник": x} for k, x in meta.get("main_sources", {}).items()],
            "предупреждение": meta.get("warning", ""),
            "альтернативы": [{"доли": r["shares"], "состав": order(r["shares"]), "дозы": doses(r["additives"]),
                              "присадки": r["additives"]} for r in alts]}
