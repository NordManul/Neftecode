"""Событийный анализ отклика серы (βT, βF), блочный бутстреп b95 (F32→T95), γ.

Только обучающий период, почасовые средние в режиме работы. Бутстреп - фиксированный
seed (воспроизводимость).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 0
N_BOOT = 300


def _hourly(s: pd.Series, running: pd.Series, train_end: pd.Timestamp) -> pd.Series:
    return s.where(running).resample("1h").mean().loc[:train_end]


def _dedup(times: pd.DatetimeIndex, min_gap_h: int) -> pd.DatetimeIndex:
    kept, last = [], None
    for t in times:
        if last is None or (t - last) >= pd.Timedelta(hours=min_gap_h):
            kept.append(t)
            last = t
    return pd.DatetimeIndex(kept)


def _find_events(main_delta: pd.Series, other_delta: pd.Series, main_thr: float, other_thr: float,
                  dedup_h: int = 12) -> pd.DatetimeIndex:
    mask = (main_delta.abs() >= main_thr) & (other_delta.abs() < other_thr)
    return _dedup(main_delta.index[mask.fillna(False)], dedup_h)


def _slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    return float((x * y).sum() / (x ** 2).sum())


def _response_slope(sulfur_h: pd.Series, events: pd.DatetimeIndex, x_series: pd.Series) -> dict:
    """Регрессия ln(S1/S0) на воздействие через начало координат + бутстреп CI 90%."""
    xs, ys = [], []
    for t in events:
        s0 = sulfur_h.loc[t - pd.Timedelta("6h"):t - pd.Timedelta("3h")].mean()
        s1 = sulfur_h.loc[t + pd.Timedelta("2h"):t + pd.Timedelta("8h")].mean()
        x = x_series.get(t, np.nan)
        if s0 > 0 and s1 > 0 and np.isfinite(x) and x != 0:
            xs.append(x)
            ys.append(np.log(s1 / s0))
    x_arr, y_arr = np.array(xs), np.array(ys)
    if len(x_arr) < 5:
        return {"beta": float("nan"), "ci90": (float("nan"), float("nan")), "n": 0}
    beta = _slope_through_origin(x_arr, y_arr)
    rng = np.random.default_rng(SEED)
    boots = [_slope_through_origin(x_arr[i], y_arr[i]) for i in
              (rng.integers(0, len(x_arr), len(x_arr)) for _ in range(N_BOOT))]
    lo, hi = np.percentile(boots, [5, 95])
    return {"beta": beta, "ci90": (float(lo), float(hi)), "n": len(x_arr)}


def calibrate_sulfur_response(ho_vals: pd.DataFrame, pak: pd.DataFrame, ho_running: pd.Series,
                               train_end: pd.Timestamp) -> dict:
    t5_h = _hourly(ho_vals["T5"], ho_running, train_end)
    f9_h = _hourly(ho_vals["F9"], ho_running, train_end)
    sulfur_h = _hourly(pak["raw"].where(pak["reason"] == ""), ho_running, train_end)

    d_t5 = t5_h - t5_h.shift(3)
    d_lnf9 = np.log(f9_h) - np.log(f9_h.shift(3))

    temp_events = _find_events(d_t5, d_lnf9, 3.0, 0.03)
    load_events = _find_events(d_lnf9, d_t5, 0.05, 1.0)
    return {"beta_t": _response_slope(sulfur_h, temp_events, d_t5),
            "beta_f": _response_slope(sulfur_h, load_events, d_lnf9)}


def literature_beta(ea_kj_mol: tuple[float, float], s_now: float, s_in_ppm: float, t_ref_k: float) -> dict:
    """Аррениус: βT(Ea) = -2·(1-√(S/Sвх))·Ea/(R·T²); βF = 2·(1-√(S/Sвх))."""
    r_gas, ratio = 8.314, 1 - (s_now / s_in_ppm) ** 0.5
    ea_lo, ea_hi = ea_kj_mol
    return {"beta_t_medium": -2 * ratio * (ea_lo * 1000) / (r_gas * t_ref_k ** 2),
            "beta_t_strong": -2 * ratio * (ea_hi * 1000) / (r_gas * t_ref_k ** 2),
            "beta_f_strong": 2 * ratio}


def calibrate_b95(avt_vals: pd.DataFrame, avt_running: pd.Series, lims: pd.DataFrame,
                   train_end: pd.Timestamp) -> dict:
    """b95: T95(продукт ГО ДТ, HT2) ~ F32_lag4h, блочный бутстреп по месяцам.

    Обе стороны детрендированы (скользящая медиана 31 сутки, только назад) как и Treq в
    `calibrate_gamma`: без этого общий медленный тренд загрузки завышает наклон (~0.24 вместо ~0.165).
    """
    f32_daily = avt_vals["F32"].where(avt_running).resample("1h").mean().shift(4).resample("1D").mean()
    f32_dt = f32_daily - f32_daily.rolling(31, min_periods=10).median()

    ht2 = lims[(lims["point"] == "HT2") & (lims["parameter"] == "95%.T") & (lims["time"] < train_end)]
    day = ht2["time"].dt.floor("D")
    y_daily = pd.Series(ht2["value"].to_numpy(), index=day.to_numpy()).groupby(level=0).mean()
    y_dt = y_daily - y_daily.rolling(31, min_periods=10).median()

    x = f32_dt.reindex(day).to_numpy()
    y, times = y_dt.reindex(day).to_numpy(), ht2["time"].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, times = x[valid], y[valid], times[valid]
    month = pd.PeriodIndex(pd.DatetimeIndex(times), freq="M").to_numpy()
    slope = _slope_through_origin(x, y)

    rng = np.random.default_rng(SEED)
    months_unique = np.unique(month)
    boots = []
    for _ in range(N_BOOT):
        draw = rng.choice(months_unique, size=len(months_unique), replace=True)
        idxs = np.concatenate([np.flatnonzero(month == m) for m in draw])
        if len(idxs) >= 10:
            boots.append(_slope_through_origin(x[idxs], y[idxs]))
    lo, hi = np.percentile(boots, [5, 95])

    years = pd.DatetimeIndex(times).year
    by_year = {int(yr): _slope_through_origin(x[years == yr], y[years == yr])
               for yr in sorted(set(years)) if (years == yr).sum() >= 20}
    return {"medium": slope, "weak": float(lo), "strong": float(hi), "n": int(len(x)), "by_year": by_year}


def calibrate_gamma(treq_daily: pd.Series, ht1_t95_daily: pd.Series, beta_ref: float) -> float:
    """γ = |βref|·(dTreq/dT95_сырья), детрендированная суточная Treq.

    Окно строго назад (без center=True), даже офлайн: агрегаты по будущему
    исключены без оговорок.
    """
    detrended = treq_daily - treq_daily.rolling(31, min_periods=15).median()
    df = pd.DataFrame({"treq": detrended, "t95": ht1_t95_daily}).dropna()
    slope = float(np.polyfit(df["t95"], df["treq"], 1)[0])
    return abs(beta_ref) * slope
