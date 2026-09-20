"""Шина сообщений между агентами: единственный канал вызова.

Прямые вызовы одного агента из другого запрещены - только bus.request().
Каждый вызов пишет в журнал запись с кратким содержанием ответа (brief()),
это делает журнал читаемым в карточке оператора без разбора трассы.
"""
from __future__ import annotations

import time
from typing import Any, Callable


class MessageBus:
    def __init__(self) -> None:
        self._handlers: dict[tuple[str, str], Callable[[Any], Any]] = {}
        self.log: list[dict] = []

    def register(self, agent_name: str, topic: str, fn: Callable[[Any], Any]) -> None:
        self._handlers[(agent_name, topic)] = fn

    def request(self, sender: str, receiver: str, topic: str, payload: Any, summary: str = "") -> Any:
        key = (receiver, topic)
        if key not in self._handlers:
            raise KeyError(f"Нет обработчика {receiver}.{topic} (шина не может доставить запрос)")
        n = len(self.log) // 2 + 1
        self.log.append({
            "n": n, "dir": "запрос", "sender": sender, "receiver": receiver,
            "topic": topic, "summary": summary or f"{sender} -> {receiver}.{topic}", "ms": None,
        })
        t0 = time.perf_counter()
        result = self._handlers[key](payload)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        brief = result.brief() if hasattr(result, "brief") else str(result)
        self.log.append({
            "n": n, "dir": "ответ", "sender": receiver, "receiver": sender,
            "topic": topic, "summary": brief, "ms": ms,
        })
        return result
