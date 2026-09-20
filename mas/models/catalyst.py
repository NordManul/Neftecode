"""Treq (нормированная требуемая температура катализатора) и индекс перепада давления.

Суточная величина доступна только с начала суток D+1: `treq_7d` - медиана за 7 суток, смещённая на сутки.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def _daily_treq(t5_1h: pd.Series, feed_1h: pd.Series, s_hat_1h: pd.Series, running_1h: pd.Series,
                 beta_f_mid: float, beta_ref: float, feed_ref: float) -> pd.Series:
    ok = running_1h & feed_1h.notna() & s_hat_1h.notna() & t5_1h.notna() & (s_hat_1h > 0)
    treq = t5_1h + (np.log(s_hat_1h / 10.0) + beta_f_mid * np.log(feed_ref / feed_1h)) / abs(beta_ref)
    return treq.where(ok).resample("1D").median()


def treq_series(t5_1h: pd.Series, feed_1h: pd.Series, s_hat_1h: pd.Series, running_1h: pd.Series,
                 calib: dict, cfg: dict) -> pd.DataFrame:
    """Суточные ряды: `treq` - значение суток D (для калибровки, где сдвиг в пределах train не важен) и
    `treq_7d` - причинная величина для решений (медиана за 7 суток, доступная с начала следующих суток)."""
    treq = _daily_treq(t5_1h, feed_1h, s_hat_1h, running_1h, calib["kinetics"]["beta_f"]["medium"],
                       abs(calib["kinetics"]["beta_t"]["medium"]), calib["reference"]["feed_tph"])
    treq_7d = treq.shift(1).rolling(7, min_periods=4).median()
    return pd.DataFrame({"treq": treq, "treq_7d": treq_7d})


def dp_index(p8_1h: pd.Series, feed_1h: pd.Series, running_1h: pd.Series, feed_ref: float,
             base_dp_norm: float) -> pd.DataFrame:
    dp_norm = (p8_1h * (feed_ref / feed_1h) ** 2).where(running_1h & feed_1h.notna() & p8_1h.notna())
    dp_norm_7d = dp_norm.resample("1D").median().rolling(7, min_periods=4).median()
    dp_ratio = dp_norm_7d / base_dp_norm
    return pd.DataFrame({"dp_norm_7d": dp_norm_7d, "dp_ratio": dp_ratio})
