"""Причинные ряды состояния в кэш: сера (фильтр), Treq/катализатор, индекс перепада.

Единственное место, где модели вызываются один раз на весь период - агенты только читают срез.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mas.common import CACHE_DIR, load_cfg, v
from mas.models.catalyst import dp_index, treq_series
from mas.models.kalman import kalman_sulfur


def _causal_fusion_weight(state: pd.DataFrame, q21: pd.DataFrame, lims: pd.DataFrame,
                          train_end: pd.Timestamp) -> float:
    """Выбирает вес фильтра против очищенного Q21 только по train-анализам.

    Оба ряда оцениваются строго до момента лабораторного анализа. Q21 добавляет
    устойчивую краткосрочную информацию, а фильтр сохраняет задержки, пропуски и
    неопределённость ансамбля.
    """
    targets = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")
                   & (lims["time"] < train_end)].sort_values("time")
    if targets.empty:
        return 1.0
    q = q21["raw"].where(q21["reason"] == "").rolling("1h", min_periods=1).mean()
    times = targets["time"].to_numpy()

    def align(series: pd.Series) -> np.ndarray:
        idx = series.index.to_numpy()
        pos = np.clip(np.searchsorted(idx, times, side="right") - 2, 0, len(idx) - 1)
        return series.to_numpy()[pos]

    filt, q_clean = align(state["S_hat"]), align(q)
    y = targets["value"].to_numpy()
    ok = np.isfinite(filt) & np.isfinite(q_clean) & np.isfinite(y)
    if ok.sum() < 30:
        return 1.0
    weights = np.linspace(0.0, 1.0, 101)
    errors = [np.abs(w * filt[ok] + (1.0 - w) * q_clean[ok] - y[ok]).mean() for w in weights]
    return float(weights[int(np.argmin(errors))])


def compute_state(cfg: dict, calib: dict, ho_vals: pd.DataFrame, running: pd.DataFrame, pak: pd.DataFrame,
                  q21: pd.DataFrame, lims: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(состояние серы, априорные оценки перед ЛИМС, суточные ряды Treq) - расчёт без обращения к файлам."""
    ho_running = running["ho_running"]
    state, priors = kalman_sulfur({"pak": pak, "q21": q21}, lims, ho_running, calib["sulfur"])
    fusion_weight = _causal_fusion_weight(
        state, q21, lims, pd.Timestamp(cfg["periods"]["train_end"])
    )
    q21_clean = q21["raw"].where(q21["reason"] == "").rolling("1h", min_periods=1).mean()
    q21_clean = q21_clean.reindex(state.index, method="ffill")
    state["S_hat_filter"] = state["S_hat"]
    fused = fusion_weight * state["S_hat"] + (1.0 - fusion_weight) * q21_clean
    state["S_hat_fusion"] = fused.combine_first(state["S_hat"])
    state["sulfur_fusion_weight"] = fusion_weight

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
