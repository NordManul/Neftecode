"""Производные ряды, зависящие от допущений по приборам: загрузка из кэша и пересчёт в памяти.

Пересчёт использует только сохранённые значения (телеметрия без кодов ошибки, показания
анализаторов, ЛИМС и отбракованные записи ЛИМС) и файлы кэша не изменяет: интерактивный просмотр
может менять пороги и сразу видеть результат, не затрагивая воспроизводимый расчёт.
"""
from __future__ import annotations

import pandas as pd

from mas.common import CACHE_DIR, OUTPUTS_DIR, load_thresholds, v
from mas.prepare.lims_clean import clean_lims
from mas.prepare.pipeline import hours_since_start, running_flags
from mas.prepare.quality_flags import clean_analyzer
from mas.state.builder import compute_state

KEYS = ("avt_flags", "ho_flags", "running", "pak_sulfur", "q21_sulfur", "lims", "state_sulfur", "state_treq")


def load_derived() -> dict[str, pd.DataFrame]:
    """Ряды, рассчитанные шагами prepare и state, как они записаны в кэше."""
    return {k: pd.read_parquet(CACHE_DIR / f"{k}.parquet") for k in KEYS}


def _lims_source(lims_clean: pd.DataFrame) -> pd.DataFrame:
    """Исходная таблица ЛИМС: принятые записи плюс отбракованные шагом prepare."""
    rejected = pd.read_csv(OUTPUTS_DIR / "data_report" / "lims_rejected.csv", parse_dates=["time"])
    return pd.concat([lims_clean.drop(columns=["available_at"]), rejected.drop(columns=["reason"])], ignore_index=True)


def rebuild_derived(cfg: dict, calib: dict) -> dict[str, pd.DataFrame]:
    """Пересчёт производных рядов при текущих `cfg` и `calib` (в памяти)."""
    base = load_derived()
    dq = cfg["data_quality"]
    avt, ho = pd.read_parquet(CACHE_DIR / "avt.parquet"), pd.read_parquet(CACHE_DIR / "ho.parquet")
    avt_flags, ho_flags = base["avt_flags"], base["ho_flags"]

    avt_running, ho_running = running_flags(avt, ho, load_thresholds())
    running = pd.DataFrame({
        "avt_running": avt_running, "ho_running": ho_running,
        "ho_hours_since_start": hours_since_start(ho_running),
    })
    pak = clean_analyzer(base["pak_sulfur"]["raw"], tuple(v(dq["pak_sulfur_valid"])))
    q21 = clean_analyzer(base["q21_sulfur"]["raw"], tuple(v(dq["q21_sulfur_valid"])))
    lims, _ = clean_lims(_lims_source(base["lims"]), cfg)
    state, _, treq = compute_state(cfg, calib, ho, running, pak, q21, lims)
    return {"avt_flags": avt_flags, "ho_flags": ho_flags, "running": running, "pak_sulfur": pak,
            "q21_sulfur": q21, "lims": lims, "state_sulfur": state, "state_treq": treq}
