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


def _link(ref: dict) -> str:
    return f"[{ref['short']}]({ref['url']})" if ref.get("url") else ref["short"]


def build_references_doc() -> str:
    cfg, refs = load_cfg(), load_refs()
    parts = ["# Источники допущений", "",
             "У каждого допущения из `config/settings.yaml` указан источник (поле `ref`, реестр — `config/references.yaml`). "
             "Если официального источника нет, это указано явно, а значение не выдаётся за нормативное.", "",
             "Типы источников: " + "; ".join(f"**{v}**" for v in KIND_TEXT.values()) + ".", ""]
    for section, rows in _rows(cfg).items():
        parts += [f"## {SECTION_TEXT.get(section, section)}", "", "| Параметр | Значение | Тип значения | Источник |", "|---|---|---|---|"]
        for r in rows:
            value = r["value"] if len(r["value"]) <= 90 else r["value"][:87] + "..."
            parts.append(f"| `{r['path']}` | {value} | {r['src']} | {'; '.join(_link(x) for x in r['refs'])} |")
        parts.append("")
    parts += ["## Использованные источники", ""]
    for rid, r in refs.items():
        title = f"[{r['title']}]({r['url']})" if r.get("url") else r["title"]
        if r.get("file") and not r.get("url"):
            title += f" (файл {r['file']})"
        parts.append(f"* **{KIND_TEXT[r['kind']]}** — {title}. {r.get('note', '')}")
    text = "\n".join(parts) + "\n"
    (DOCS_DIR / "05_ИСТОЧНИКИ_ДОПУЩЕНИЙ.md").write_text(text, encoding="utf-8")
    return text
