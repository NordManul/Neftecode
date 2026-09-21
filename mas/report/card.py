"""Карточка оператора в Markdown из 8 блоков `Recommendation.blocks`.

Деловой стиль, разумная точность: сера 2 знака, температуры 1,
вероятности в процентах без дробей.
"""
from __future__ import annotations

from typing import Any


def _fmt(x: Any, pct: bool = False) -> str:
    """Числа с двумя знаками; вероятности (`pct`) - в процентах без дробей."""
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "да" if x else "нет"
    if x == "" or (isinstance(x, (list, tuple)) and not x):
        return "нет"
    if isinstance(x, float):
        return f"{x:.0%}" if pct else f"{x:.2f}"
    if isinstance(x, (list, tuple)):
        return ", ".join(_fmt(v, pct) for v in x)
    if isinstance(x, dict):
        return "; ".join(f"{k}={_fmt(v)}" for k, v in x.items())
    return str(x)


def _is_probability(key: str) -> bool:
    return key.startswith("P(") or key in ("балл",)


def _dict_lines(d: dict) -> str:
    return "\n".join(f"- **{k}**: {_fmt(v, _is_probability(k))}" for k, v in d.items())


def _block3(b3) -> str:
    if isinstance(b3, str):
        return b3
    if not b3:
        return "Изменений не требуется"
    lines = ["| Тег | Описание | Текущее | Рекомендуемое | Изменение |", "|---|---|---|---|---|"]
    lines += [f"| {a['tag']} | {a['desc']} | {_fmt(a['current'])} {a['unit']} | {_fmt(a['recommended'])} {a['unit']} | {_fmt(a['change'])} |"
              for a in b3]
    return "\n".join(lines)


def _block5(b5) -> str:
    if not b5:
        return "-"
    lines = ["| Проверка | Значение | Результат |", "|---|---|---|"]
    lines += [f"| {c['check']} | {_fmt(c['value'], 'шанс' in c['check'] or 'навреди' in c['check'])} | "
              f"{'выполнено' if c['ok'] else 'нарушено'} |" for c in b5]
    return "\n".join(lines)


def _alternatives_md(alts: list[dict]) -> str:
    if not alts:
        return "Нет допустимых альтернатив."
    lines = ["| id | ΔT5, °C | ΔF9 | ΔF32, т/ч | P(S>10) | P(T95>360) | запас по сере, мг/кг |", "|---|---|---|---|---|---|---|"]
    for a in alts:
        lines.append(f"| {a.get('id')} | {_fmt(a.get('dT5'))} | {_fmt(a.get('rF9'), True)} | {_fmt(a.get('dF32'))} | "
                     f"{_fmt(a.get('P_exceed'), True)} | {_fmt(a.get('P_t95'), True)} | {_fmt(a.get('quality_margin'))} |")
    return "\n".join(lines)


def _bus_log_md(log: list[dict]) -> str:
    lines = ["| # | Направление | Отправитель | Получатель | Тема | Содержание | мс |", "|---|---|---|---|---|---|---|"]
    for e in log:
        lines.append(f"| {e['n']} | {e['dir']} | {e['sender']} | {e['receiver']} | {e['topic']} | {e['summary']} | {e['ms'] or ''} |")
    return "\n".join(lines)


def _block8(b8: dict) -> str:
    if not b8.get("доли"):
        return "Расчёт блендинга не выполнялся."
    head = [f"- **Режим**: {b8['режим']}", f"- **Вариантов / допустимых**: {b8['вариантов']} / {b8['допустимых']}"]
    if b8.get("предупреждение"):
        head.append(f"- **Предупреждение**: {b8['предупреждение']}")
    batch = b8.get("партия, т")
    recipe = ["", "| Компонент | Доля | Партия, т |" if batch else "| Компонент | Доля |", "|---|---|---|" if batch else "|---|---|"]
    recipe += [f"| {k} | {v:.0%} | {batch.get(k, 0):g} |" if batch else f"| {k} | {v:.0%} |" for k, v in b8["доли"].items()]
    recipe += [f"| {k} | {v:g} кг/т | - |" if batch else f"| {k} | {v:g} кг/т |" for k, v in b8["присадки"].items()]
    props = ["", "| Показатель | Значение | Норма | Результат |", "|---|---|---|---|"]
    for c in b8["проверки"]:
        val = c["значение"] if isinstance(c["значение"], str) else f"{c['значение']:.2f} {c['единицы']}".strip()
        props.append(f"| {c['показатель']} | {val} | {c['норма']} | {'выполнено' if c['ok'] else 'нарушено'} |")
    alts = b8.get("альтернативы") or []
    lines = head + recipe + props
    if alts:
        lines += ["", "Альтернативы:"]
        for a in alts:
            recipe_text = "; ".join(f"{name} {share:.0%}" for name, share in a["доли"].items() if share > 0)
            doses = "; ".join(f"{name} {dose:g} кг/т" for name, dose in a["присадки"].items() if dose > 0) or "без присадок"
            lines.append(f"- {recipe_text}; {doses}")
    return "\n".join(lines)


def card_markdown(rec) -> str:
    b = rec.blocks
    parts = [f"# Карточка оператора: {rec.t}", "", f"**Статус: {rec.status}**", ""]
    parts += ["## 1. Время и состояние", _dict_lines(b["1. Время и состояние"]), ""]
    parts += ["## 2. Проблема / риск", _dict_lines(b["2. Проблема / риск"]), ""]
    parts += ["## 3. Предлагаемое действие", _block3(b["3. Предлагаемое действие"]), ""]
    parts += ["## 4. Ожидаемый эффект", _dict_lines(b["4. Ожидаемый эффект"]) if b["4. Ожидаемый эффект"] else "-", ""]
    parts += ["## 5. Проверка ограничений", _block5(b["5. Проверка ограничений"]), ""]
    parts += ["## 6. Уверенность", _dict_lines(b["6. Уверенность"]), ""]
    parts += ["## 7. Объяснение", b["7. Объяснение"] or "-", ""]
    parts += ["## 8. Блендинг", _block8(b["8. Блендинг"]), ""]
    parts += ["## Альтернативы", _alternatives_md(rec.alternatives), ""]
    parts += ["## Журнал шины", _bus_log_md(rec.trace.get("bus_log", [])), ""]
    return "\n".join(parts)
