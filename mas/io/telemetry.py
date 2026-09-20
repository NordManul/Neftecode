"""Чтение CSV телеметрии АВТ и 24-2000. Кэш в parquet, читаем один раз.

Формат: колонка `date` + теги, служебные `Unnamed: *` удаляются.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from mas.common import get_data_dir, load_cfg


def _find_one(pattern: str) -> str:
    data_dir = get_data_dir()
    matches = glob.glob(os.path.join(data_dir, pattern))
    if not matches:
        raise FileNotFoundError(f"Не найден файл по шаблону {pattern} в {data_dir}")
    return sorted(matches)[0]


def _read_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df.astype(float)


def read_telemetry() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (avt, ho): индекс date, все теги float."""
    cfg = load_cfg()
    avt = _read_one(_find_one(cfg["files"]["avt"]))
    ho = _read_one(_find_one(cfg["files"]["ho"]))
    return avt, ho
