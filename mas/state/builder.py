"""Причинные ряды состояния в кэш: сера (фильтр), Treq/катализатор, индекс перепада.

Единственное место, где модели вызываются один раз на весь период - агенты только читают срез.
"""
from __future__ import annotations

import pandas as pd

from mas.common import CACHE_DIR, load_cfg
from mas.models.catalyst import dp_index, treq_series
from mas.models.kalman import kalman_sulfur


def compute_state(cfg: dict, calib: dict, ho_vals: pd.DataFrame, running: pd.DataFrame, pak: pd.DataFrame,
                  q21: pd.DataFrame, lims: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(состояние серы, априорные оценки перед ЛИМС, суточные ряды Treq) - расчёт без обращения к файлам."""
    ho_running = running["ho_running"]
    state, priors = kalman_sulfur({"pak": pak, "q21": q21}, lims, ho_running, calib["sulfur"])

    ho_running_1h = ho_running.resample("1h").max()
    p8_1h = ho_vals["P8"].where(ho_running).resample("1h").mean()
    feed_1h = ho_vals["F9"].where(ho_running).resample("1h").mean()
    dp = dp_index(p8_1h, feed_1h, ho_running_1h, calib["reference"]["feed_tph"], calib["dp_norm_base"])
    state["dp_ratio"] = dp["dp_ratio"].reindex(state.index, method="ffill")

    t5_1h = ho_vals["T5"].where(ho_running).resample("1h").mean()
    s_hat_1h = state["S_hat"].resample("1h").mean()
    # Причинность: `treq_7d` уже смещён на сутки внутри treq_series.
    return state, priors, treq_series(t5_1h, feed_1h, s_hat_1h, ho_running_1h, calib, cfg)[["treq_7d"]]


def build_state(calib: dict) -> None:
    cfg = load_cfg()
    ho_vals = pd.read_parquet(CACHE_DIR / "ho.parquet")
    running = pd.read_parquet(CACHE_DIR / "running.parquet")
    pak = pd.read_parquet(CACHE_DIR / "pak_sulfur.parquet")
    q21 = pd.read_parquet(CACHE_DIR / "q21_sulfur.parquet")
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")

    state, priors, daily = compute_state(cfg, calib, ho_vals, running, pak, q21, lims)
    priors.to_parquet(CACHE_DIR / "kalman_priors.parquet")
    state.to_parquet(CACHE_DIR / "state_sulfur.parquet")
    daily.to_parquet(CACHE_DIR / "state_treq.parquet")
