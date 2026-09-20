"""10 демонстрационных сценариев из отложенного периода."""
from __future__ import annotations

import json

import pandas as pd

from mas.agents.orchestrator import Orchestrator
from mas.common import OUTPUTS_DIR, get_logger, to_jsonable
from mas.report.card import card_markdown

log = get_logger("demo")

SCENARIOS = [
    (1, "устойчивый_режим", "2026-07-12 08:00", None, "БЕЗ ИЗМЕНЕНИЙ"),
    (2, "риск_по_сере", "2025-10-10 12:00", None, "КОРРЕКТИРУЮЩЕЕ ДЕЙСТВИЕ"),
    (3, "риск_по_t95", "2025-05-31 20:00", None, "КОРРЕКТИРУЮЩЕЕ ДЕЙСТВИЕ + ЭСКАЛАЦИЯ ТЕХНОЛОГУ"),
    (4, "пак_застыл", "2025-03-23 08:00", None, "БЕЗ ИЗМЕНЕНИЙ"),
    (5, "после_пуска_нет_данных", "2026-07-02 12:00", None, "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ"),
    (6, "сера_выше_нормы", "2025-01-21 12:00", None, "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ"),
    (7, "конец_цикла_катализатора", "2026-04-09 12:00", None, "НАДЁЖНОЙ РЕКОМЕНДАЦИИ НЕТ"),
    (8, "простой", "2026-04-20 12:00", None, "НАБЛЮДЕНИЕ (установка не в работе)"),
    (9, "запас_по_качеству", "2025-01-01 14:00", None, "БЕЗ ИЗМЕНЕНИЙ"),
    (10, "блендинг_сценарий", "2025-05-31 20:00", "scenario", None),
]


def run_demo() -> list[dict]:
    orch = Orchestrator()
    out_dir = OUTPUTS_DIR / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for num, name, ts, mode, expected in SCENARIOS:
        rec = orch.cycle(pd.Timestamp(ts), save=False, blend_mode=mode)
        tag = f"{num:02d}_{name}"
        (out_dir / f"{tag}.md").write_text(card_markdown(rec), encoding="utf-8")
        payload = {"status": rec.status, "t": rec.t, "blocks": rec.blocks, "chosen": rec.chosen,
                   "alternatives": rec.alternatives, "trace": rec.trace}
        (out_dir / f"{tag}.json").write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        match = expected is None or expected == rec.status
        if not match:
            log.warning("Сценарий %d (%s): получено %r, ожидалось %r", num, name, rec.status, expected)
        results.append({"n": num, "name": name, "status": rec.status, "expected": expected, "match": match})
    return results
