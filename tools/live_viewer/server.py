"""Интерактивный просмотр карточки оператора: локальный HTTP-сервер на стандартной библиотеке.

Отдельный инструмент (не входит в пакет `mas`): использует агентов `Orchestrator` и кэш, собранный
`run.py all`. Возможности: воспроизведение времени, сценарные значения T5, F9, F32 и редактирование
допущений (пороги приборов, нормы, надёжность, ранжирование, блендинг) с пересчётом в памяти; файлы
конфигурации и кэша не изменяются.

Запуск:
    python tools/live_viewer/server.py [порт]        (по умолчанию 8899)
Адрес: http://localhost:<порт>
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from assumptions import Assumptions  # noqa: E402
from mas.agents.orchestrator import Orchestrator  # noqa: E402
from mas.common import load_cfg, to_jsonable  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}
_assumptions: Assumptions | None = None


def _get_assumptions() -> Assumptions:
    global _assumptions
    if _assumptions is None:
        _assumptions = Assumptions(Orchestrator())
    return _assumptions


def _cycle_payload(t: pd.Timestamp, overrides: dict[str, float] | None) -> dict:
    session = _get_assumptions()
    with session.lock:
        rec = session.orch.cycle(t, save=False, overrides=overrides or None)
    return {
        "t": str(rec.t), "status": rec.status, "blocks": rec.blocks,
        "chosen": rec.chosen, "alternatives": rec.alternatives,
        "bus_log": rec.trace.get("bus_log", []),
        "assumptions_changed": len(session.overrides),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str, code: int = 200, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, code: int = 200, extra: dict[str, str] | None = None) -> None:
        body = json.dumps(to_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", code, extra)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/timerange":
                cfg = load_cfg()
                self._send_json({"start": cfg["periods"]["train_end"], "end": cfg["periods"]["test_end"]})
            elif parsed.path == "/api/cycle":
                overrides = {k: float(qs[k][0]) for k in ("T5", "F9", "F32") if qs.get(k) and qs[k][0] != ""}
                self._send_json(_cycle_payload(pd.Timestamp(qs["t"][0]), overrides))
            elif parsed.path == "/api/assumptions":
                self._send_json(_get_assumptions().describe())
            elif parsed.path == "/api/assumptions/export":
                payload = {"overrides": _get_assumptions().overrides}
                self._send_json(payload, extra={"Content-Disposition": 'attachment; filename="assumptions.json"'})
            elif parsed.path in ("/", "/index.html"):
                self._send((STATIC_DIR / "index.html").read_bytes(), CONTENT_TYPES[".html"])
            else:
                candidate = STATIC_DIR / parsed.path.lstrip("/")
                if candidate.suffix in (".js", ".css") and candidate.is_file():
                    self._send(candidate.read_bytes(), CONTENT_TYPES[candidate.suffix])
                else:
                    self._send_json({"error": "не найдено"}, code=404)
        except Exception as exc:  # noqa: BLE001 - причина возвращается клиенту
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, code=500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if urlparse(self.path).path != "/api/assumptions":
                self._send_json({"error": "не найдено"}, code=404)
                return
            body = self._read_json()
            session = _get_assumptions()
            result = session.apply(body.get("changes"), body.get("reset"))
            self._send_json({**result, **session.describe()}, code=200 if result["ok"] else 422)
        except Exception as exc:  # noqa: BLE001 - причина возвращается клиенту
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, code=500)

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    print("Инициализация агентов...")
    _get_assumptions()
    print(f"Сервер запущен: http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
