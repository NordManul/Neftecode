"""Очистка ЛИМС: диапазоны методов, дубли выгрузки, available_at, отчёты об отбраковке.

Возраст анализа считается от `time` (время измерения по файлу ЛИМС); отдельного времени выдачи результата в файле нет,
поэтому анализ считается доступным с момента `time` (`available_at = time`). ЛИМС не отбрасывается фильтром никогда,
здесь отбраковываются только значения вне области применения метода и дубли выгрузки. Нижняя граница диапазона не
включается (ноль - заполнитель пропуска), верхняя включается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.common import v


def _range_rejects(df: pd.DataFrame, ranges: dict) -> pd.Series:
    lo = df["parameter"].map(lambda p: ranges.get(p, [np.nan, np.nan])[0])
    hi = df["parameter"].map(lambda p: ranges.get(p, [np.nan, np.nan])[1])
    has_range = lo.notna()
    return has_range & ((df["value"] <= lo) | (df["value"] > hi))


def clean_lims(L: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(чистые записи + available_at, отбраковка с причиной)."""
    ranges = v(cfg["data_quality"]["lims_ranges"])
    dedup_min = v(cfg["data_quality"]["lims_dedup_minutes"])

    df = L.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["point", "parameter", "time"]).reset_index(drop=True)

    out_of_range = _range_rejects(df, ranges)
    rejected_parts = [df[out_of_range].assign(reason="вне области метода")]
    df = df[~out_of_range].sort_values(["point", "parameter", "time"])

    gap_min = df.groupby(["point", "parameter"])["time"].diff().dt.total_seconds() / 60.0
    is_dup = gap_min.notna() & (gap_min < dedup_min)
    rejected_parts.append(df[is_dup].assign(reason="дубль выгрузки (< 5 мин)"))
    df = df[~is_dup].copy()

    df["available_at"] = df["time"]
    rejected = pd.concat(rejected_parts, ignore_index=True) if rejected_parts else df.iloc[:0]
    return df.reset_index(drop=True), rejected.reset_index(drop=True)
