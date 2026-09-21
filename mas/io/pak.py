"""Чтение выгрузки ПАК: поточный анализатор серы («24-2000:Mg.Sulfur»).

Формат листа: пары колонок [время, значение], разделённые пустой колонкой; строки 0-1 - заголовок и единица, данные с строки 2.
Расчётная плотность D15 из выгрузки (вторая пара колонок) не читается: плотность продукта проверяется по анализам ЛИМС.
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


def read_pak() -> pd.Series:
    """Показания анализатора серы, мг/кг, индекс - время анализа."""
    raw = pd.read_excel(_find_file(), header=None, skiprows=2, engine="openpyxl", usecols=[0, 1])
    return _pair_to_series(raw, 0, 1)
