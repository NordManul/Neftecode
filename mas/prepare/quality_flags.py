"""Причинные флаги качества сигналов: код 307, заморозка, насыщение, отказ.

Все детекторы смотрят только в прошлое: длина «замороженного»
хвоста, а не длина всего плоского участка - иначе в момент t использовалась бы информация
из будущего о том, сколько ещё продлится плоский участок.
"""
from __future__ import annotations

import pandas as pd

from mas.common import v


def trailing_run_length(s: pd.Series) -> pd.Series:
    """Длина хвоста повторяющихся значений, включая текущее (тест: [1,1,1,2,2] -> [1,2,3,1,2])."""
    same = s.diff() == 0
    grp = (~same).cumsum()
    return (same.groupby(grp).cumsum() + 1).astype(int)


def telemetry_flags(values: pd.DataFrame, is_sentinel: pd.DataFrame, freeze_steps: int) -> pd.DataFrame:
    """Флаги int8: 0 норма, 1 код ошибки источника, 2 заморозка (причинный хвост одинаковых значений)."""
    flags = pd.DataFrame(0, index=values.index, columns=values.columns, dtype="int8")
    for col in values.columns:
        run = trailing_run_length(values[col])
        flags.loc[(run >= freeze_steps) & ~is_sentinel[col], col] = 2
        flags.loc[is_sentinel[col], col] = 1
    return flags


def clean_telemetry(df: pd.DataFrame, cfg: dict, freeze_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(значения с заменённым кодом ошибки на NaN, флаги telemetry_flags)."""
    is_sentinel = df.isin(set(v(cfg["data_quality"]["sentinel_values"])))
    values = df.mask(is_sentinel)
    return values, telemetry_flags(values, is_sentinel, freeze_steps)


def clean_analyzer(s: pd.Series, valid_range: tuple[float, float]) -> pd.DataFrame:
    """Брак анализатора: отказ (<= нижней границы), насыщение (>= верхней), заморозка (повтор точно того же значения).

    Показание анализатора - число с плавающей точкой: точное повторение предыдущего значения означает, что прибор
    выдаёт прежний результат (заморозка), поэтому повторы отбраковываются без дополнительного порога."""
    low, high = valid_range
    run_len = trailing_run_length(s)
    reason = pd.Series("", index=s.index, dtype=object)
    reason[s <= low] = "отказ"
    reason[s >= high] = "насыщение шкалы"
    reason[(reason == "") & (run_len >= 2)] = "заморозка"
    return pd.DataFrame({"raw": s, "reason": reason, "run_len": run_len})
