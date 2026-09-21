"""Формирование `docs/05_ИСТОЧНИКИ_ДОПУЩЕНИЙ.md` из config/settings.yaml и config/references.yaml.

Для каждого допущения указан источник: нормативный документ, литература, данные проекта либо отметка
«официального источника нет». Числа берутся из конфигурации, вручную не вводятся.
"""
from __future__ import annotations

from mas.common import DOCS_DIR, load_cfg, load_refs, node_refs

KIND_TEXT = {"official": "нормативный документ", "literature": "литература (не норматив)", "data": "данные проекта",
             "none": "официального источника нет"}
SECTION_TEXT = {"data_quality": "Данные и приборы", "spec": "Нормы и допустимый риск", "sensor_fusion": "Фильтр серы",
                "decision_vars": "Управляющие воздействия",
                "reliability": "Надёжность", "kinetics": "Кинетика", "process": "Технологические константы",
                "blending": "Блендинг"}


def _rows(cfg: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    def walk(node, path):
        if isinstance(node, dict) and "src" in node:
            value = node["value"] if "value" in node else {k: x for k, x in node.items() if k not in ("src", "note", "ref")}
            text = "; ".join(f"{k}={x}" for k, x in value.items()) if isinstance(value, dict) else str(value)
            out.setdefault(path.split(".")[0], []).append({"path": path, "value": text, "src": node["src"], "refs": node_refs(node)})
        elif isinstance(node, dict):
            for k, x in node.items():
                walk(x, f"{path}.{k}" if path else k)

    walk(cfg, "")
    return out


def _local(ref: dict) -> str | None:
    """Путь копии документа относительно docs/ (копии вне docs/ по ссылке не открываются)."""
    file = ref.get("file") or ""
    return file[len("docs/"):] if file.startswith("docs/") else None


def _link(ref: dict) -> str:
    if ref.get("url"):
        return f"[{ref['short']}]({ref['url']})"
    return f"[{ref['short']}]({_local(ref)})" if _local(ref) else ref["short"]


def _derived_section() -> list[str]:
    """Значения, которые не заданы в конфигурации, а найдены по данным обучающего периода (`prepare`, `calibrate`)."""
    import json

    from mas.common import CACHE_DIR, ROOT
    rows: list[tuple[str, str, str]] = []
    thr_path, cal_path = CACHE_DIR / "thresholds.json", ROOT / "config" / "calibrated.json"
    if thr_path.exists():
        thr = json.loads(thr_path.read_text(encoding="utf-8"))
        rows += [("признак работы ГО: расход сырья F9, т/ч", "метод Оцу по ряду", f"{thr['ho_feed_min_tph']:.1f}"),
                 ("признак работы ГО: температура T5, °C", "метод Оцу по ряду", f"{thr['ho_temp_min_c']:.1f}"),
                 ("признак работы АВТ: расход F65", "метод Оцу по ряду", f"{thr['avt_feed_min']:.1f}"),
                 ("заморозка телеметрии, шагов", "самая длинная серия одинаковых значений управляющих тегов в рабочем режиме обучения, плюс один",
                  str(thr["telemetry_freeze_steps"]))]
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        ages = cal.get("lims_age_limit_h", {})
        rows += [("длительность пускового режима, ч", "три постоянные времени релаксации серы после пуска (Q21)",
                  f"{cal['startup']['window_h']:.1f}"),
                 ("предельная неопределённость оценки серы, мг/кг", "разброс уровня серы в рабочем режиме за вычетом погрешности лаборатории",
                  f"{cal['uncertainty']['sulfur_level_sd']:.2f}"),
                 ("опорный расход сырья, т/ч", "медиана F9 рабочего режима обучения", f"{cal['reference']['feed_tph']:.1f}"),
                 ("граница фактора перепада давления", "99-й перцентиль приведённого перепада", f"{cal['reliability']['dp_ratio_p99']:.2f}"),
                 ("граница фактора быстроты изменения T5, °C за 4 ч", "95-й перцентиль изменения T5 за 4 ч", f"{cal['reliability']['t5_roc4h_p95']:.2f}"),
                 ("предельный возраст анализов ЛИМС, ч", "три медианных интервала между анализами",
                  "; ".join(f"{k} — {x:.0f}" for k, x in ages.items()))]
    if not rows:
        return []
    out = ["## Значения, найденные по данным", "", "Не заданы в конфигурации; вычисляются шагами `prepare` и `calibrate` по обучающему периоду.", "",
           "| Величина | Метод | Значение |", "|---|---|---|"]
    out += [f"| {name} | {method} | {value} |" for name, method, value in rows]
    return out + [""]


def build_references_doc() -> str:
    cfg, refs = load_cfg(), load_refs()
    parts = ["# Источники значений", "",
             "У каждого значения из `config/settings.yaml` указан источник (поле `ref`, реестр — `config/references.yaml`, копии "
             "документов — `docs/источники/`). Значений без источника нет: параметр, для которого источника нет, удалён из системы "
             "либо вычисляется по данным (раздел «Значения, найденные по данным»). Перечень удалённых, выведенных и принятых значений — "
             "`docs/02_ОБОСНОВАНИЕ.md`, раздел 16.", "",
             "Тип значения (`src`): **GOST** — норматив (ГОСТ 32511-2013); **REQUIREMENT** — нормативное требование к продукту; "
             "**TASK** — требование или разрешение технического задания либо ответ организаторов; **DATA** — получено по данным; "
             "**LIT** — число взято из документа, указанного в источнике.", "",
             "Типы источников: " + "; ".join(f"**{v}**" for k, v in KIND_TEXT.items() if k != "none") + ".", ""]
    for section, rows in _rows(cfg).items():
        parts += [f"## {SECTION_TEXT.get(section, section)}", "", "| Параметр | Значение | Тип значения | Источник |", "|---|---|---|---|"]
        for r in rows:
            value = r["value"] if len(r["value"]) <= 90 else r["value"][:87] + "..."
            parts.append(f"| `{r['path']}` | {value} | {r['src']} | {'; '.join(_link(x) for x in r['refs'])} |")
        parts.append("")
    parts += _derived_section()
    parts += ["## Использованные источники", ""]
    for rid, r in refs.items():
        title = f"[{r['title']}]({r['url']})" if r.get("url") else (f"[{r['title']}]({_local(r)})" if _local(r) else r["title"])
        if r.get("file"):
            title += f" (копия в проекте: `{r['file']}`)"
        parts.append(f"* **{KIND_TEXT[r['kind']]}** — {title}. {r.get('note', '')}")
    text = "\n".join(parts) + "\n"
    (DOCS_DIR / "05_ИСТОЧНИКИ_ДОПУЩЕНИЙ.md").write_text(text, encoding="utf-8")
    return text
