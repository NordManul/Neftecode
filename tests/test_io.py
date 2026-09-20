"""Разбор шапки ЛИМС и единиц на малых синтетических наборах."""
from __future__ import annotations

import os

import openpyxl
import pandas as pd
import pytest

from mas.prepare.lims_clean import clean_lims


@pytest.fixture()
def synthetic_lims_dir(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    # 3 точки, по 2 показателя, пары времени/значения разной длины; одно значение вне диапазона,
    # одно нечисловое, строка единиц у второй точки сдвинута (кг/м3 вместо °C).
    ws.append(["Установка 'Гидроочистка'. Точка отбора '2'. Продукт 'X'", None,
               None, None, "Установка 'АВТ'. Точка отбора '1'. Продукт 'Y'", None])
    ws.append(["Mg.Sulfur", None, "95%.T", None, "CFPP", None])
    ws.append(["мг/кг", None, "°C", None, "кг/м3", None])
    ws.append(["Количество значений:", 3, "Количество значений:", 2, "Количество значений:", 4])
    rows = [
        [pd.Timestamp("2023-01-01 10:00"), 8.5, pd.Timestamp("2023-01-01 10:00"), 340.0, pd.Timestamp("2023-01-01 10:00"), -6.0],
        [pd.Timestamp("2023-01-02 10:00"), 2120.0, pd.Timestamp("2023-01-02 10:00"), 345.0, pd.Timestamp("2023-01-02 10:00"), -7.0],
        [pd.Timestamp("2023-01-03 10:00"), "Pt Created", None, None, pd.Timestamp("2023-01-03 10:00.5"), -8.0],
        [None, None, None, None, pd.Timestamp("2023-01-04 10:00"), -9.0],
    ]
    for r in rows:
        ws.append(r)
    path = tmp_path / "sample_ЛИМС.xlsx"
    wb.save(path)
    os.environ["NEFTEKOD_DATA"] = str(tmp_path)
    yield tmp_path
    del os.environ["NEFTEKOD_DATA"]


def test_lims_header_parsing(synthetic_lims_dir):
    from mas.io.lims import read_lims
    long_df, units_df, n_non_numeric = read_lims()
    assert n_non_numeric == 1  # "Pt Created"
    assert set(long_df["point"]) == {"HT2", "AVT1"}
    assert long_df[(long_df.point == "HT2") & (long_df.parameter == "Mg.Sulfur")].shape[0] == 2
    assert long_df[(long_df.point == "AVT1") & (long_df.parameter == "CFPP")].shape[0] == 4


def test_lims_units_conflict_recorded_but_code_dict_used(synthetic_lims_dir):
    from mas.io.lims import UNITS, read_lims
    _, units_df, _ = read_lims()
    row = units_df[(units_df.point == "AVT1") & (units_df.parameter == "CFPP")].iloc[0]
    assert row["declared_unit"] == "кг/м3"  # сдвинутая единица зафиксирована как есть
    assert row["fixed_unit"] == UNITS["CFPP"] == "°C"  # но используется словарь в коде


def test_lims_out_of_range_and_nonnumeric_rejected(synthetic_lims_dir):
    from mas.io.lims import read_lims
    long_df, _, _ = read_lims()
    cfg = {"data_quality": {"lims_ranges": {"Mg.Sulfur": [0, 500], "95%.T": [0, 400]}, "lims_dedup_minutes": 5}}
    clean, rejected = clean_lims(long_df, cfg)
    assert (rejected["reason"] == "вне области метода").any()
    assert 2120.0 not in clean["value"].to_numpy()
    assert (clean["available_at"] == clean["time"]).all()  # отдельного времени выдачи результата в файле нет
