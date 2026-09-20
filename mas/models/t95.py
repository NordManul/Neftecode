"""Оценка T95 товарного ДТ: взвешенная по обратной дисперсии комбинация анализов.

Резервный источник (нет анализа продукта) - сырьё гидроочистки HT1:95%.T со смещением
и добавочной дисперсией: готовится `feed_fallback_sample`, комбинирует - `t95_estimate`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def recent_samples(lims_ht2: pd.DataFrame, f32: pd.Series, t: pd.Timestamp, max_age_h: float, n: int = 3) -> list[dict]:
    """До `n` последних анализов T95 продукта, доступных на момент `t` (по `available_at`) и не старше `max_age_h`,
    с расходом F32 в момент отбора. `lims_ht2` отсортирована по `available_at`."""
    pos = lims_ht2["available_at"].to_numpy().searchsorted(np.datetime64(t), side="right")
    sub = lims_ht2.iloc[:pos]
    sub = sub[(t - sub["time"]).dt.total_seconds() / 3600 <= max_age_h].sort_values("time", ascending=False).head(n)
    return [{"source": "HT2", "value": float(row["value"]), "extra_sd": 0.0,
             "age_h": (t - row["time"]).total_seconds() / 3600,
             "F32_at_sample": float(f32.loc[row["time"] - pd.Timedelta(hours=1):row["time"]].mean())}
            for _, row in sub.iterrows()]


def feed_fallback_sample(ht1_95t_value: float, age_h: float, f32_at_sample: float, calib: dict) -> dict:
    """Проба сырья ГО как замена анализу продукта, с поправкой offset и доп. дисперсией sd_diff²."""
    return {
        "source": "HT1_fallback", "value": ht1_95t_value + calib["t95_offset"],
        "extra_sd": calib["t95_sd_diff"], "age_h": age_h, "F32_at_sample": f32_at_sample,
    }


def t95_estimate(samples: list[dict], f32_now: float, calib: dict) -> tuple[float, float]:
    """Пробы (последние три в пределах возраста `lims_age_limit_h`), приведённые к текущему отбору F32 -> (t95_est, t95_sigma)."""
    if not samples:
        return float("nan"), float("nan")
    b95_mid = calib["b95"]["medium"]
    sigma_a2 = calib["t95_sigma"] ** 2
    drift = calib["t95_drift"]
    num, den = 0.0, 0.0
    for s in samples:
        value_adj = s["value"] + b95_mid * (f32_now - s["F32_at_sample"])
        var = sigma_a2 + s.get("extra_sd", 0.0) ** 2 + drift * s["age_h"] / 24.0
        num += value_adj / var
        den += 1.0 / var
    return num / den, (1.0 / den) ** 0.5
