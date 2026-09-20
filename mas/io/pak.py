"""Чтение выгрузки ПАК: поточные анализаторы серы и плотности.

Формат листа: пары колонок [время, значение] для «24-2000:Mg.Sulfur» и «24-2000:D15»,
разделённые пустой колонкой; строки 0-1 - заголовок и единица, данные с строки 2.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from mas.common import get_data_dir, load_cfg


def _find_file() -> str:
    cfg = load_cfg()
    matches = glob.glob(os.path.join(get_data_dir(), cfg["files"]["pak"]))
    if not matches:
        raise FileNotFoundError("Не найден файл ПАК по шаблону из config.files.pak")
    return sorted(matches)[0]


def _pair_to_series(raw: pd.DataFrame, col_time: int, col_value: int) -> pd.Series:
    pair = raw.iloc[:, [col_time, col_value]].dropna()
    idx = pd.DatetimeIndex(pd.to_datetime(pair.iloc[:, 0]), name="date")
    s = pd.Series(pair.iloc[:, 1].astype(float).to_numpy(), index=idx)
    return s.sort_index()


def read_pak() -> dict[str, pd.Series]:
    """Возвращает {"sulfur_ppm": Series, "d15": Series}, индекс - время анализа."""
    raw = pd.read_excel(_find_file(), header=None, skiprows=2, engine="openpyxl")
    return {
        "sulfur_ppm": _pair_to_series(raw, 0, 1),
        "d15": _pair_to_series(raw, 3, 4),
    }
