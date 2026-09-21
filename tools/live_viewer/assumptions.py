"""Редактируемые допущения интерактивного просмотра: реестр параметров, проверка и применение.

Значения меняются только в памяти процесса (общие словари `load_cfg()` и `load_calib()`); файлы
`config/settings.yaml`, `config/calibrated.json` и кэш не изменяются. Параметры, влияющие на
производные ряды (пороги приборов, признак работы, ЛИМС, параметры фильтра), приводят к пересчёту
фильтра серы и суточных рядов Treq; остальные действуют со следующего расчёта карточки.
"""
from __future__ import annotations

import copy
import math
import threading
import time
from typing import Any

from mas.common import load_calib, load_cfg, node_refs
from mas.models.kinetics import response_ensemble
from mas.state.derived import load_derived, rebuild_derived

CFG, CALIB = "cfg", "calib"
INSTANT, STATE = "cycle", "state"
TABS = ["Приборы и данные", "Качество и риск", "Надёжность", "Выпуск", "Блендинг"]
LIMS_LABELS = {  # показатели ЛИМС, для которых заданы области применения методов и которые используются в решениях
    "Mg.Sulfur": ("Сера, мг/кг (ГОСТ ISO 20846, ГОСТ ISO 20884)", 0, 1000),
    "95%.T": ("T95, °C (ГОСТ ISO 3405)", 0, 600),
}


def _p(pid, tab, section, label, kind, unit="", lo=None, hi=None, step=None, effect=INSTANT, hint="",
       root=CFG, src=None, labels=None, size=None, norm=False, ascending=False, equal=False) -> dict:
    """Описание параметра. `size` - длина вектора, `norm` - нормировка на единицу, `ascending` - возрастающие
    значения, `equal` - допустимо равенство границ диапазона."""
    return {"id": pid, "tab": tab, "section": section, "label": label, "kind": kind, "unit": unit, "min": lo,
            "max": hi, "step": step, "effect": effect, "hint": hint, "root": root, "src": src, "labels": labels,
            "size": size, "norm": norm, "ascending": ascending, "equal": equal}


def _registry(cfg: dict) -> list[dict]:
    dev, lab = TABS[0], "Лабораторный анализ (ЛИМС)"
    noise = ("Дисперсия случайной погрешности прибора, не описанной быстрой компонентой сигнала. "
             "Чем она больше, тем меньший вес имеют показания прибора в фильтре Калмана.")
    lag = ("Запаздывание показаний анализатора относительно потока, шагов по 10 мин. Увеличивает допустимую "
           "неопределённость при сопоставлении с лабораторным анализом.")
    rows = [
        _p("data_quality/pak_sulfur_valid", dev, "ПАК (сера)", "Допустимый диапазон показаний", "range", "мг/кг", 0, 1000,
           0.05, STATE, "Показание не выше нижней границы считается отказом прибора, не ниже верхней - насыщением шкалы. "
           "Такие точки исключаются из фильтра серы. Границы найдены по данным (плато насыщения и значения отказа)."),
        _p("sulfur/pak/r", dev, "ПАК (сера)", "Дисперсия шума измерения", "number", "(мг/кг)²", 0.0001, 25, 0.01, STATE,
           noise, root=CALIB, src="DATA"),
        _p("sulfur/pak/delay", dev, "ПАК (сера)", "Запаздывание показаний", "int", "шагов", 0, 36, 1, STATE, lag,
           root=CALIB, src="DATA"),
        _p("sulfur/pak/shift", dev, "ПАК (сера)", "Сдвиг показаний по времени", "int", "шагов", 0, 36, 1, STATE,
           "Показание ПАК относится к моменту на указанное число шагов (10 мин) раньше: транспортное запаздывание "
           "между точкой измерения и точкой отбора пробы. Подбирается по качеству прогноза анализов ЛИМС.",
           root=CALIB, src="DATA"),
        _p("data_quality/q21_sulfur_valid", dev, "Q21 (сера на выходе гидроочистки, ppm)", "Допустимый диапазон показаний",
           "range", "мг/кг", 0, 1000, 0.05, STATE, "Границы отказа и насыщения поточного анализатора Q21, найденные по "
           "данным: плато насыщения шкалы и значения отказа."),
        _p("sulfur/q21/r", dev, "Q21 (сера на выходе гидроочистки, ppm)", "Дисперсия шума измерения", "number", "(мг/кг)²",
           0.0001, 25, 0.01, STATE, noise, root=CALIB, src="DATA"),
        _p("sulfur/q21/delay", dev, "Q21 (сера на выходе гидроочистки, ppm)", "Запаздывание показаний", "int", "шагов", 0, 36, 1,
           STATE, lag, root=CALIB, src="DATA"),
        _p("sulfur/sigma_lims", dev, "Фильтр серы", "Погрешность лабораторного анализа (СКО)", "number", "мг/кг", 0.05, 10,
           0.05, STATE, "Стандартное отклонение погрешности лабораторного определения серы. Определяет вес результата "
           "ЛИМС в фильтре. Значение получено калибровкой по прогнозам фильтра на обучающем периоде и включает не только "
           "погрешность метода, но и несоответствие момента и места отбора пробы состоянию, которое видят анализаторы. "
           "Ориентир на 10 мг/кг: СКО воспроизводимости ГОСТ ISO 20846 около 0.8 мг/кг, ГОСТ ISO 20884 около 1.1 мг/кг.",
           root=CALIB, src="DATA"),
        _p("sulfur/q_b", dev, "Фильтр серы", "Дисперсия дрейфа смещения анализатора за шаг", "number", "(мг/кг)²", 1e-6, 0.01, 1e-5,
           STATE, "Скорость изменения систематического смещения показаний анализатора между лабораторными поправками. "
           "Чем больше, тем быстрее фильтр «забывает» накопленное смещение.", root=CALIB, src="DATA"),
        _p("uncertainty/sulfur_level_sd", dev, "Фильтр серы", "Предельная неопределённость оценки серы", "number",
           "мг/кг", 0.1, 10, 0.1, INSTANT, "При большей неопределённости оценки серы система отказывается от рекомендации. "
           "Значение - разброс уровня серы установки в рабочем режиме обучающего периода (стандартное отклонение анализов ЛИМС "
           "за вычетом погрешности лабораторного определения): оценка неопределённее этого разброса не информативнее "
           "климатологии.", root=CALIB, src="DATA"),
    ]
    for key, (name, lo, hi) in LIMS_LABELS.items():
        rows.append(_p(f"data_quality/lims_ranges/{key}", dev, "Области применения методов ЛИМС", name, "range", "", lo,
                       hi, 0.1, STATE, "Записи вне области метода считаются ошибкой ввода и отбраковываются; нижняя "
                       "граница не включается. Границы - области применения стандартов методов испытаний."))
    tq, risk = TABS[1], "Нормы и риск"
    rows += [
        _p("spec/sulfur_max_mgkg", tq, "Нормы продукта (не редактируются)", "Сера, не более", "readonly", "мг/кг"),
        _p("spec/t95_max_c", tq, "Нормы продукта (не редактируются)", "T95, не более", "readonly", "°C"),
        _p("spec/flash_min_c", tq, "Нормы продукта (не редактируются)", "Температура вспышки, не менее", "readonly", "°C"),
        _p("spec/d15_range", tq, "Нормы продукта (не редактируются)", "Плотность при 15 °C: ДТ с гидроочистки и летнее товарное",
           "readonly", "кг/м³"),
        _p("spec/cetane_min", tq, "Нормы продукта (не редактируются)",
           "Цетановое число, не менее: летнее товарное (для ДТ с гидроочистки не нормируется)", "readonly"),
        _p("spec/winter/d15_range", tq, "Нормы продукта (не редактируются)", "Плотность при 15 °C: зимнее товарное", "readonly",
           "кг/м³"),
        _p("spec/winter/cetane_min", tq, "Нормы продукта (не редактируются)", "Цетановое число, не менее: зимнее товарное",
           "readonly"),
        _p("spec/product_grade", tq, risk, "Сорт товарного ДТ для проверки смеси", "enum", effect=INSTANT,
           labels=["summer", "winter"], hint="summer - летнее товарное ДТ (плотность 820-845, цетановое число не менее 51), "
           "winter - зимнее (плотность 800-845, цетановое число не менее 49). Нормы применяются к смеси в блоке 8 карточки; "
           "анализы ДТ с гидроочистки проверяются по плотности 820-845, цетановое число для них не нормируется."),
        _p("spec/risk_alpha", tq, risk, "Допустимая вероятность превышения нормы", "number", "доля", 0.001, 0.5, 0.005,
           INSTANT, "Управляющее воздействие принимается, если откалиброванная вероятность превышения нормы по сере после него "
           "(наблюдаемая частота превышений при такой расчётной вероятности, по анализам обучающего периода) "
           "не больше заданной (по T95 - расчётная вероятность не выше, чем без изменений). Действие запускается, когда превышение "
           "нормы вероятнее, чем соблюдение (вероятность выше 1/2): стоимостных данных для иного порога нет."),
        _p("sensor_fusion/forecast_horizon_h", tq, risk, "Горизонт прогноза", "number", "ч", 1, 12, 1, INSTANT,
           "Горизонт прогноза серы в карточке. Решение принимается по текущей оценке: прогноз на 4 ч не лучше климатологии."),
    ]
    rl, sev = TABS[2], "Катализатор и оборудование"
    rows += [
        _p("reliability/catalyst_cycle_c/start", rl, sev, "Температура на входе в реактор в начале цикла катализатора", "number",
           "°C", 300, 450, 1, INSTANT, "ИТС 30-2021, табл. 2.36. Начало шкалы фактора «катализатор близок к концу цикла»."),
        _p("reliability/catalyst_cycle_c/end", rl, sev, "Температура на входе в реактор в конце цикла (до регенерации)", "number",
           "°C", 300, 450, 1, INSTANT, "ИТС 30-2021, табл. 2.36. Конец шкалы фактора «катализатор близок к концу цикла»; "
           "запас до конца цикла считается от Treq."),
    ]
    rk, proc = TABS[3], "Технологические константы"
    rows += [
        _p("process/ht_yield", rk, proc, "Выход гидроочищенного ДТ от сырья", "number", "доля", 0.5, 1, 0.005, INSTANT,
           "Отношение массового расхода продукта к расходу сырья; определяет изменение выпуска при изменении F9."),
    ]
    rows += _blending_rows(cfg)
    return rows


COMPONENT_PROPS = (("S", "Сера", "мг/кг", 0, 50, 0.1), ("D15", "Плотность при 15 °C", "кг/м³", 700, 950, 1),
                   ("cetane", "Цетановое число", "", 30, 80, 0.5), ("T95", "T95", "°C", 150, 450, 1),
                   ("CFPP", "ПТФ", "°C", -80, 20, 1), ("stock_t", "Запас", "т", 0, 1000000, 10))
MAIN_DEFAULTS = (("D15", "Плотность при 15 °C", "кг/м³", 700, 950, 1), ("cetane", "Цетановое число", "", 30, 80, 0.5),
                 ("CFPP", "ПТФ", "°C", -80, 20, 1))
PARAM_LABELS = {"cetane": "цетановое число", "CFPP": "ПТФ"}


def _blending_rows(cfg: dict) -> list[dict]:
    """Параметры вкладки «Блендинг»: режим и, если в конфигурации заданы данные компонентов и присадок, рецептура и подбор."""
    tab, bl = TABS[4], cfg["blending"]
    main = bl["main_component"]
    sec = "Режим смешения"
    rows = [_p("blending/mode", tab, sec, "Режим смешения", "enum", effect=INSTANT, labels=["product_only", "scenario", "manual"],
               hint="product_only - товарное ДТ равно гидроочищенному ДТ; scenario - подбор рецептуры с наибольшей "
                    "долей ГО ДТ и наименьшими дозами присадок по компонентам из конфигурации; manual - расчёт заданной "
                    "рецептуры. Компоненты и присадки в материалах проекта не описаны, поэтому без данных в "
                    "config/settings.yaml (blending) любой режим сводится к product_only.")]
    if "manual_shares" in bl:
        sec = "Ручная рецептура (режим manual)"
        for name in [main, *bl["scenario_components"]]:
            rows.append(_p(f"blending/manual_shares/{name}", tab, sec, f"Доля: {name}", "number", "доля", 0, 1, 0.01,
                           hint="Доля компонента в заданной рецептуре; доли нормируются на 100 %. Компоненты, "
                                "исключённые из смеси, не учитываются."))
        for name in bl["additives"]:
            rows.append(_p(f"blending/manual_doses/{name}", tab, sec, f"Доза: {name}", "number", "кг/т", 0, 20, 0.05,
                           hint="Дозировка присадки в заданной рецептуре."))
    sec = f"Основной компонент: {main}"
    if "main_share_range" in bl:
        rows.append(_p("blending/main_share_range", tab, sec, "Допустимая доля в смеси", "range", "доля", 0, 1, 0.05, equal=True,
                       hint="Ограничение доли основного компонента при подборе рецептуры."))
    if "main_stock_t" in bl:
        rows.append(_p("blending/main_stock_t", tab, sec, "Запас", "number", "т", 0, 1000000, 100,
                       hint="Доступное количество основного компонента для партии смешения."))
    for key, name, unit, lo, hi, step in MAIN_DEFAULTS:
        if key in bl.get("main_defaults", {}).get("value", {}):
            rows.append(_p(f"blending/main_defaults/{key}", tab, sec, f"{name} при отсутствии свежего анализа", "number", unit, lo,
                           hi, step, hint="Значение свойства ГО ДТ, если свежего анализа ЛИМС нет. Сера и T95 берутся из "
                                          "прогноза цикла."))
    for cname in bl["scenario_components"]:
        sec, base = f"Компонент: {cname}", f"blending/scenario_components/{cname}"
        rows.append(_p(f"{base}/enabled", tab, sec, "Используется в смеси", "bool",
                       hint="Исключённый компонент не участвует в подборе рецептуры и в расчёте заданной рецептуры."))
        for key, label, unit, lo, hi, step in COMPONENT_PROPS:
            rows.append(_p(f"{base}/{key}", tab, sec, label, "number", unit, lo, hi, step,
                           hint="Свойство компонента из паспорта или лабораторных данных."))
        rows.append(_p(f"{base}/share_range", tab, sec, "Допустимая доля в смеси", "range", "доля", 0, 1, 0.05, equal=True,
                       hint="Ограничение доли компонента при подборе рецептуры."))
    for aname, add in bl["additives"].items():
        sec, base = f"Присадка: {aname}", f"blending/additives/{aname}"
        target, n_doses = PARAM_LABELS.get(add["param"], add["param"]), len(add["doses_kg_t"])
        rows += [
            _p(f"{base}/enabled", tab, sec, "Используется в смеси", "bool", hint="Исключённая присадка не участвует в расчёте."),
            _p(f"{base}/effect_per_kg_t", tab, sec, f"Эффект на показатель «{target}» на 1 кг/т", "number", "", -50, 50, 0.1,
               hint="Изменение показателя при дозировке 1 кг присадки на тонну смеси (по паспорту присадки)."),
            _p(f"{base}/doses_kg_t", tab, sec, "Перебираемые дозировки", "vector", "кг/т", 0, 20, 0.05, size=n_doses,
               ascending=True, labels=[f"доза {i + 1}" for i in range(n_doses)],
               hint="Дозировки, из которых подбирается рецептура; значения возрастают, первая обычно равна нулю (без присадки)."),
        ]
    sec = "Подбор рецептуры и продукт"
    for key, label, unit, lo, hi, step, hint in (
            ("batch_t", "Размер партии", "т", 100, 100000, 100, "Партия смешения: определяет расход компонентов и проверку запасов."),
            ("share_step", "Шаг сетки долей", "доля", 0.02, 0.5, 0.01, "Шаг перебора долей компонентов; меньший шаг увеличивает число вариантов.")):
        if key in bl:
            rows.append(_p(f"blending/{key}", tab, sec, label, "number", unit, lo, hi, step, hint=hint))
    for key, label, unit, hi, hint in (
            ("T95", "Запас по T95 на нелинейность смешения", "°C", 20, "Смесь из нескольких компонентов оценивается с запасом: линейное смешение T95 неточно."),
            ("CFPP", "Запас по ПТФ на нелинейность смешения", "°C", 20, "Запас для предельной температуры фильтруемости смеси из нескольких компонентов."),
            ("cetane", "Запас по цетановому числу на нелинейность смешения", "ед.", 10, "Цетановое число смеси из нескольких компонентов уменьшается на заданный запас.")):
        if key in bl.get("rule_margins", {}).get("value", {}):
            rows.append(_p(f"blending/rule_margins/{key}", tab, sec, label, "number", unit, 0, hi, 0.5, hint=hint))
    return rows


def _resolve(root: dict, path: str) -> tuple[dict, str, str, str, list]:
    """(контейнер, ключ, источник, примечание, ссылки на источники): значение лежит в container[key]."""
    node, src, note, refs, keys = root, "", "", [], path.split("/")
    for i, key in enumerate(keys):
        if isinstance(node, dict) and key not in node and "value" in node:
            node = node["value"]
        nxt = node[key]
        if isinstance(nxt, dict) and "src" in nxt:
            src, note, refs = nxt["src"], nxt.get("note", ""), node_refs(nxt)
        if i == len(keys) - 1:
            if isinstance(nxt, dict) and "value" in nxt:
                return nxt, "value", src, note, refs
            return node, key, src, note, refs
        node = nxt
    raise KeyError(path)


def _number(x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError("ожидается число")
    return float(x)


def _clean(p: dict, raw: Any) -> Any:
    """Приведение значения к типу параметра с проверкой границ; ValueError с пояснением при нарушении."""
    lo, hi, kind = p["min"], p["max"], p["kind"]

    def bounded(x: float) -> float:
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            raise ValueError(f"значение должно быть в пределах {lo} … {hi}")
        return x

    if kind == "number":
        return bounded(_number(raw))
    if kind == "int":
        x = _number(raw)
        if x != int(x):
            raise ValueError("ожидается целое число")
        return int(bounded(x))
    if kind == "bool":
        if not isinstance(raw, bool):
            raise ValueError("ожидается значение да/нет")
        return raw
    if kind in ("range", "vector"):
        size = 2 if kind == "range" else p["size"]
        if not isinstance(raw, (list, tuple)) or len(raw) != size:
            raise ValueError("неверное число значений")
        vals = [bounded(_number(x)) for x in raw]
        if kind == "range" and not (vals[0] <= vals[1] if p["equal"] else vals[0] < vals[1]):
            raise ValueError("нижняя граница должна быть меньше верхней" if not p["equal"] else "нижняя граница не должна превышать верхнюю")
        if p["ascending"] and any(a >= b for a, b in zip(vals, vals[1:])):
            raise ValueError("значения должны возрастать")
        if p["norm"]:
            if sum(vals) <= 0:
                raise ValueError("сумма весов должна быть положительной")
            vals = [x / sum(vals) for x in vals]
        return vals
    if kind == "enum":
        if raw not in p["labels"]:
            raise ValueError("недопустимое значение")
        return raw
    raise ValueError("параметр не редактируется")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    return a == b


class Assumptions:
    """Реестр допущений, текущие переопределения и применение их к агентам."""

    def __init__(self, orchestrator) -> None:
        self.orch, self.lock = orchestrator, threading.RLock()
        self.roots = {CFG: load_cfg(), CALIB: load_calib()}
        self.params = _registry(self.roots[CFG])
        self.by_id = {p["id"]: p for p in self.params}
        self.defaults, self.overrides = {}, {}
        for p in self.params:
            container, key, src, note, refs = _resolve(self.roots[p["root"]], p["id"])
            self.defaults[p["id"]] = copy.deepcopy(container[key])
            p["src"], p["note"] = p["src"] or src or "", note
            p["refs"] = refs or node_refs({"ref": "data_project"})
            if p["kind"] == "readonly":
                p["value"] = self.defaults[p["id"]]

    def _effective(self, overrides: dict, pid: str) -> Any:
        return overrides[pid] if pid in overrides else self.defaults[pid]

    def describe(self) -> dict:
        rows = []
        for p in self.params:
            row = {k: p[k] for k in ("id", "tab", "section", "label", "kind", "unit", "min", "max", "step", "effect", "hint",
                                     "src", "note", "labels", "refs")}
            row["default"] = self.defaults[p["id"]]
            row["value"] = self._effective(self.overrides, p["id"])
            row["changed"] = p["id"] in self.overrides
            rows.append(row)
        return {"tabs": TABS, "params": rows, "overrides": self.overrides}

    def _cross_checks(self, merged: dict) -> dict:
        """Проверки согласованности значений, зависящих друг от друга; ключ - параметр, к которому относится ошибка."""
        eff = lambda pid: self._effective(merged, pid)  # noqa: E731
        errors = {}
        if eff("reliability/catalyst_cycle_c/start") >= eff("reliability/catalyst_cycle_c/end"):
            errors["reliability/catalyst_cycle_c/start"] = "температура начала цикла должна быть ниже температуры конца цикла"
        prefix = "blending/scenario_components/"
        active = [i[:-len("/enabled")] for i in self.by_id if i.startswith(prefix) and i.endswith("/enabled") and eff(i)]
        if "blending/main_share_range" in self.by_id:
            ranges = [eff("blending/main_share_range")] + [eff(f"{base}/share_range") for base in active]
            if sum(r[0] for r in ranges) > 1 + 1e-9:
                errors["blending/main_share_range"] = "сумма минимальных долей включённых компонентов превышает 100 %"
            elif sum(r[1] for r in ranges) < 1 - 1e-9:
                errors["blending/main_share_range"] = "сумма максимальных долей включённых компонентов меньше 100 %"
        if eff("blending/mode") == "manual" and "blending/manual_shares/" + self.roots[CFG]["blending"]["main_component"] in self.by_id:
            names = [i[len(prefix):-len("/enabled")] for i in self.by_id if i.startswith(prefix) and i.endswith("/enabled") and eff(i)]
            main = self.roots[CFG]["blending"]["main_component"]
            if sum(eff(f"blending/manual_shares/{n}") for n in [main, *names]) <= 0:
                errors["blending/mode"] = "в заданной рецептуре нет ненулевых долей включённых компонентов"
        return errors

    def _validate(self, changes: dict) -> tuple[dict, dict]:
        cleaned, errors = {}, {}
        for pid, raw in changes.items():
            p = self.by_id.get(pid)
            if p is None:
                errors[pid] = "неизвестный параметр"
                continue
            try:
                cleaned[pid] = _clean(p, raw)
            except ValueError as exc:
                errors[pid] = str(exc)
        if not errors:
            errors = self._cross_checks({**self.overrides, **cleaned})
        return cleaned, errors

    def _materialise(self) -> None:
        for p in self.params:
            if p["kind"] == "readonly":
                continue
            container, key = _resolve(self.roots[p["root"]], p["id"])[:2]
            container[key] = copy.deepcopy(self._effective(self.overrides, p["id"]))
        self.orch.quality.members, self.orch.quality.weights = response_ensemble(self.roots[CALIB]["kinetics"])

    def apply(self, changes: dict | None = None, reset: list[str] | str | None = None) -> dict:
        """Применяет изменения и/или сброс к значениям по умолчанию; возвращает итог и длительность пересчёта."""
        with self.lock:
            cleaned, errors = self._validate(changes or {})
            if errors:
                return {"ok": False, "errors": errors}
            started, before = time.time(), dict(self.overrides)
            ids = list(self.overrides) if reset == "all" else list(reset or [])
            for pid in ids:
                self.overrides.pop(pid, None)
            for pid, val in cleaned.items():
                if _same(val, self.defaults[pid]):
                    self.overrides.pop(pid, None)
                else:
                    self.overrides[pid] = val
            changed = [i for i in {*before, *self.overrides}
                       if not _same(self._effective(before, i), self._effective(self.overrides, i))]
            self._materialise()
            state_ids = [i for i in self.overrides if self.by_id[i]["effect"] == STATE]
            recomputed = any(self.by_id[i]["effect"] == STATE for i in changed)
            if recomputed:
                frames = (rebuild_derived(self.roots[CFG], self.roots[CALIB])
                          if state_ids else load_derived())
                self.orch.data.set_derived(frames)
            return {"ok": True, "changed": changed, "recomputed": recomputed,
                    "elapsed_s": round(time.time() - started, 1)}
