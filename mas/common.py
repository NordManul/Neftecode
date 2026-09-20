"""Пути, конфигурация, helper v(), безопасная сериализация, логгер. Все допущения - в settings.yaml."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
CACHE_DIR = ROOT / "cache"
OUTPUTS_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"

_cfg_cache: dict | None = None
_calib_cache: dict | None = None
_refs_cache: dict | None = None

def get_data_dir() -> Path:
    """Папка с исходными файлами: --data аргумент прокидывается через env NEFTEKOD_DATA."""
    raw = os.environ.get("NEFTEKOD_DATA")
    if not raw:
        raise FileNotFoundError("Не указана папка с данными: передайте --data <папка> или NEFTEKOD_DATA")
    p = Path(raw)
    if not p.is_dir():
        raise FileNotFoundError(f"Папка с данными не найдена: {p}")
    return p


def load_cfg() -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        with open(CONFIG_DIR / "settings.yaml", "r", encoding="utf-8") as f:
            _cfg_cache = yaml.safe_load(f)
    return _cfg_cache


def load_calib() -> dict:
    global _calib_cache
    if _calib_cache is None:
        path = CONFIG_DIR / "calibrated.json"
        if not path.exists():
            raise FileNotFoundError("config/calibrated.json не найден: выполните `python run.py calibrate`")
        with open(path, "r", encoding="utf-8") as f:
            _calib_cache = json.load(f)
    return _calib_cache

def load_refs() -> dict:
    """Реестр источников допущений (config/references.yaml)."""
    global _refs_cache
    if _refs_cache is None:
        with open(CONFIG_DIR / "references.yaml", "r", encoding="utf-8") as f:
            _refs_cache = yaml.safe_load(f)
    return _refs_cache


def node_refs(node: Any) -> list[dict]:
    """Источники узла конфигурации: список {id, kind, short, title, url?, note}."""
    ids = node.get("ref", []) if isinstance(node, dict) else []
    refs = load_refs()
    return [{"id": i, **refs[i]} for i in (ids if isinstance(ids, list) else [ids])]


def v(node: Any) -> Any:
    """Разворачивает узел {value, src, note} конфигурации либо возвращает значение как есть."""
    return node["value"] if isinstance(node, dict) and "value" in node else node


def action_threshold(cfg: dict, kind: str) -> float:
    """Порог запуска корректирующего действия: не ниже допустимой вероятности `spec.risk_alpha` и не ниже фоновой
    вероятности превышения нормы (`kind` = sulfur | t95; доля анализов выше нормы в рабочем режиме обучающего периода,
    `calibrated.json`): риск, обычный для установки, действием не устраняется, реакция нужна на его рост."""
    background = load_calib().get("risk", {}).get(f"{kind}_background", 0.0)
    return float(max(v(cfg["spec"]["risk_alpha"]), background))


def load_thresholds() -> dict:
    """Пороги, полученные по данным шагом prepare (`cache/thresholds.json`): признак работы и заморозка телеметрии."""
    with open(CACHE_DIR / "thresholds.json", "r", encoding="utf-8") as f:
        return json.load(f)


def startup_window_h(calib: dict) -> float:
    """Длительность пускового режима, ч: три постоянные времени релаксации серы после пуска (`mas/calibrate/startup.py`).
    Тот же масштаб времени задаёт окно информации: возраст показаний анализаторов, после которого они считаются устаревшими."""
    return float(calib["startup"]["window_h"])


def lims_age_limit_h(calib: dict, point: str, param: str) -> float:
    """Предельный возраст анализа ЛИМС, ч: три медианных интервала между анализами в обучающем периоде."""
    return float(calib["lims_age_limit_h"][f"{point}:{param}"])


def commercial_norms(cfg: dict) -> dict:
    """Нормы товарного ДТ для проверки смеси: летнего (`spec.*`) либо зимнего (`spec.winter.*`) по `spec.product_grade`."""
    spec = cfg["spec"]
    source = spec["winter"] if v(spec["product_grade"]) == "winter" else spec
    return {"grade": "winter" if source is not spec else "summer", "d15_range": v(source["d15_range"]),
            "cetane_min": v(source["cetane_min"])}


def to_jsonable(obj: Any) -> Any:
    """Безопасная сериализация для JSON: NaN/NaT -> None, Timestamp -> ISO, numpy -> python."""
    if obj is None:
        return None
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(x) for k, x in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_jsonable(x) for x in list(obj)]
    return obj


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dirs() -> None:
    for d in (CACHE_DIR, OUTPUTS_DIR / "cycles", OUTPUTS_DIR / "demo", OUTPUTS_DIR / "data_report", DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
