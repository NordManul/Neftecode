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
    """Папка с исходными файлами: --data (через env NEFTEKOD_DATA); если не указана - папка `data/` в корне репозитория."""
    raw = os.environ.get("NEFTEKOD_DATA") or (str(ROOT / "data") if (ROOT / "data").is_dir() else None)
    if not raw:
        raise FileNotFoundError("Не указана папка с данными: передайте --data <папка>, задайте NEFTEKOD_DATA или создайте папку data/ в корне репозитория")
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


NEUTRAL_RISK = 0.5


def action_threshold(cfg: dict | None = None, kind: str | None = None) -> float:
    """Порог запуска корректирующего действия: вероятность 1/2.

    Действие снижает загрузку и потому уменьшает выпуск. Стоимостных данных, по которым можно сопоставить потерю выпуска с
    ущербом от превышения нормы, нет, поэтому применяется нейтральная граница решения: действие назначается, когда
    превышение нормы вероятнее, чем соблюдение (вероятность выше 1/2). Оценка вероятности откалибрована по данным
    (Brier ниже климатологии, docs/03), поэтому вероятность 1/2 означает, что превышение ожидается примерно в каждом втором
    таком случае. Пока вероятность ниже порога, режим без изменений допустим; выбранное действие обязано вернуть риск к
    допустимой вероятности `spec.risk_alpha`."""
    return NEUTRAL_RISK


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


def calibrated_probability(calib: dict, p: Any) -> np.ndarray:
    """Расчётная вероятность превышения нормы по сере, приведённая к наблюдаемой частоте превышений: монотонное отображение,
    подобранное по анализам ЛИМС обучающего периода (`calibrated.json`, раздел `probability`, `mas/calibrate/probability.py`).
    Расчётная вероятность ниже примерно 0.15 не отличает состояния друг от друга: наблюдаемая частота там 2-3 % при любом
    значении, поэтому допустимость действия оценивается по отображённой вероятности."""
    m = calib["probability"]
    return np.interp(np.asarray(p, dtype=float), m["raw"], m["calibrated"])


def accepted_raw_probability(calib: dict, alpha: float) -> float:
    """Наибольшая расчётная вероятность превышения, при которой откалиброванная вероятность не выше `alpha`."""
    grid = np.linspace(0.0, 1.0, 10001)
    return float(grid[calibrated_probability(calib, grid) <= alpha].max())


def max_sigma_for_decision(calib: dict) -> float:
    """Предельная неопределённость оценки серы, мг/кг: разброс уровня серы в рабочем режиме обучающего периода
    (`calibrated.json`, `uncertainty.sulfur_level_sd`). Оценка неопределённее этого разброса не информативнее климатологии."""
    return float(calib["uncertainty"]["sulfur_level_sd"])


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
