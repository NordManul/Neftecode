"""Метрики оценок на отложенном периоде: сера (7 оценщиков), риск T95, регламент ВАК.

Все оценки на отложенном периоде строятся только по данным, доступным на момент оценки;
значимо лучший результат по сравнению с простыми оценщиками считается поводом проверить причинность.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from mas.agents.data_agent import DataAgent
from mas.agents.quality_agent import QualityAgent
from mas.bus import MessageBus
from mas.common import (CACHE_DIR, OUTPUTS_DIR, accepted_raw_probability, action_threshold, calibrated_probability, load_calib,
                        load_cfg, startup_window_h, to_jsonable, v)
from mas.models.kalman import kalman_sulfur
from mas.models.vak import evaluate_vak


def _mae_auc(pred: pd.Series, lims_s: pd.DataFrame, tol_min: int = 30) -> tuple[float, float, int]:
    """«Оценка перед анализом»: берётся последняя точка НЕ ПОЗЖЕ времени отбора (без утечки)."""
    times = lims_s["time"].to_numpy()
    idx = pred.index.to_numpy()
    pos = np.clip(np.searchsorted(idx, times, side="right") - 1, 0, len(idx) - 1)
    matched = pred.to_numpy()[pos]
    gap_min = np.abs(idx[pos] - times) / np.timedelta64(1, "m")
    ok = (gap_min <= tol_min) & np.isfinite(matched)
    y_true = (lims_s["value"].to_numpy() > 10).astype(int)
    mae = float(np.abs(matched[ok] - lims_s["value"].to_numpy()[ok]).mean())
    if len(set(y_true[ok])) < 2:
        return mae, float("nan"), int(ok.sum())
    order = np.argsort(matched[ok])
    y_sorted = y_true[ok][order]
    n1, n0 = y_sorted.sum(), len(y_sorted) - y_sorted.sum()
    ranks = np.arange(1, len(y_sorted) + 1)
    auc = (ranks[y_sorted == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")
    return mae, float(auc), int(ok.sum())


GBM_TAGS = {"ho": ["T5", "T6", "T11", "P3", "P13", "P8", "F9", "F2"], "avt": ["T55", "T33", "P22", "T20", "F65", "F32"]}


def _gbm_regime_baseline(ho: pd.DataFrame, avt: pd.DataFrame, lims_all: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    """Градиентный бустинг серы только по режимным тегам (1-часовые и суточные средние, причинно), без анализаторов."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    feats = []
    for df, tags in ((ho, GBM_TAGS["ho"]), (avt, GBM_TAGS["avt"])):
        for window in ("1h", "24h"):
            feats.append(df[tags].rolling(window, min_periods=1).mean().add_suffix(f"_{window}"))
    x_all = pd.concat(feats, axis=1)
    pos = np.clip(x_all.index.searchsorted(lims_all["time"].to_numpy(), side="right") - 1, 0, len(x_all) - 1)
    x = x_all.iloc[pos].to_numpy()
    y = lims_all["value"].to_numpy()
    train = (lims_all["time"] < train_end).to_numpy()
    model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0).fit(x[train], y[train])
    pred = pd.Series(model.predict(x[~train]), index=lims_all["time"][~train].to_numpy())
    mae, auc, n = _mae_auc(pred, lims_all[~train])
    return {"mae": mae, "auc": auc, "n": n}


def _sulfur_estimators(pak: pd.DataFrame, q21: pd.DataFrame, state: pd.DataFrame, lims_s: pd.DataFrame) -> dict:
    """Все базовые оценки причинны: скользящее окно строго назад от момента отбора (без будущих показаний)."""
    pak_clean_h = pak["raw"].where(pak["reason"] == "").rolling("1h", min_periods=1).mean()
    q21_clean_h = q21["raw"].where(q21["reason"] == "").rolling("1h", min_periods=1).mean()
    mean_two = (pak_clean_h + q21_clean_h) / 2
    last_lims = pd.Series(lims_s["value"].to_numpy(), index=lims_s["available_at"].to_numpy()).shift(1).reindex(
        pak.index.union(lims_s["available_at"]), method="ffill").reindex(pak.index)  # предыдущий анализ, не сам
    out = {}
    for name, series in (("ПАК как есть", pak["raw"]), ("ПАК, среднее за час, очищенный", pak_clean_h),
                          ("Q21, среднее за час, очищенный", q21_clean_h), ("Среднее двух приборов, очищенное", mean_two),
                          ("Последний анализ ЛИМС", last_lims), ("Фильтр по двум приборам", state["S_hat"].shift(1))):
        mae, auc, n = _mae_auc(series, lims_s)
        out[name] = {"mae": mae, "auc": auc, "n": n}
    return out


def _sulfur_only_pak(pak: pd.DataFrame, running: pd.Series, lims: pd.DataFrame, calib: dict) -> dict:
    params = dict(calib["sulfur"])
    params.pop("q21", None)
    state, _ = kalman_sulfur({"pak": pak}, lims, running, params)
    mae, auc, n = _mae_auc(state["S_hat"].shift(1), lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")])
    return {"mae": mae, "auc": auc, "n": n}


def _coverage_brier(state: pd.DataFrame, lims_s: pd.DataFrame, climatology_rate: float, sigma_lims: float
                     ) -> tuple[float, float, float, dict]:
    """Сравнение идёт с шумной лабораторной меткой, поэтому к P00 добавляется σ_ЛИМС² - та же
    дисперсия инновации, что фильтр использует при собственном обновлении по ЛИМС."""
    idx = state.index.to_numpy()
    pos = np.clip(np.searchsorted(idx, lims_s["time"].to_numpy(), side="right") - 2, 0, len(idx) - 1)  # до самого анализа
    s_hat = state["S_hat"].to_numpy()[pos]
    s_sigma = np.sqrt(state["S_sigma"].to_numpy()[pos] ** 2 + sigma_lims ** 2)
    lo, hi = norm.ppf(0.05, s_hat, s_sigma), norm.ppf(0.95, s_hat, s_sigma)
    y = lims_s["value"].to_numpy()
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    p_exceed = 1 - norm.cdf((10 - s_hat) / s_sigma)
    brier = float(np.mean((p_exceed - (y > 10)) ** 2))
    brier_clim = float(np.mean((climatology_rate - (y > 10)) ** 2))
    # сигнал системы о риске: вероятность превышения физической серы (без погрешности лаборатории) выше порога действия
    p_decision = 1 - norm.cdf((10 - s_hat) / state["S_sigma"].to_numpy()[pos])
    flagged, exceeded = p_decision > action_threshold(), y > 10
    alert = {"share_flagged": float(flagged.mean()), "recall": float(flagged[exceeded].mean()) if exceeded.any() else float("nan"),
             "precision": float(exceeded[flagged].mean()) if flagged.any() else float("nan"), "n_flagged": int(flagged.sum()),
             "n_exceeded": int(exceeded.sum())}
    return coverage, brier, brier_clim, alert


PROBABILITY_BINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)


def _probability_calibration(state: pd.DataFrame, lims_s: pd.DataFrame, running: pd.DataFrame, calib: dict) -> dict:
    """Наблюдаемая частота превышений нормы по анализам отложенного периода (установившийся режим) в интервалах расчётной
    вероятности фильтра и для двух правил допустимости действия: расчётная вероятность не выше `risk_alpha` (строгое правило) и
    откалиброванная вероятность не выше `risk_alpha` (принятое правило)."""
    limit, alpha = v(load_cfg()["spec"]["sulfur_max_mgkg"]), v(load_cfg()["spec"]["risk_alpha"])
    idx = state.index.to_numpy()
    pos = np.clip(np.searchsorted(idx, lims_s["time"].to_numpy(), side="right") - 2, 0, len(idx) - 1)      # до самого анализа
    steady = (running["ho_running"] & (running["ho_hours_since_start"] > startup_window_h(calib))).reindex(state.index).fillna(False).to_numpy()[pos]
    p = 1 - norm.cdf((limit - state["S_hat"].to_numpy()[pos]) / state["S_sigma"].to_numpy()[pos])
    y = (lims_s["value"].to_numpy() > limit)
    keep = steady & np.isfinite(p)
    p, y = p[keep], y[keep]

    def group(mask: np.ndarray) -> dict:
        return {"n": int(mask.sum()), "n_exceeded": int(y[mask].sum()), "frequency": float(y[mask].mean()) if mask.any() else float("nan"),
                "mean_p": float(p[mask].mean()) if mask.any() else float("nan")}

    bins = [{"from": lo, "to": hi, **group((p > lo) & (p <= hi) if lo > 0 else (p <= hi))}
            for lo, hi in zip(PROBABILITY_BINS[:-1], PROBABILITY_BINS[1:])]
    level = accepted_raw_probability(calib, alpha)
    return {"n": int(keep.sum()), "bins": bins, "accepted_raw_level": level,
            "accepted": group(p <= level), "strict": group(p <= alpha),
            "brier_calibrated": float(np.mean((calibrated_probability(calib, p) - y) ** 2)),
            "brier_raw": float(np.mean((p - y) ** 2))}


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    n1, n0 = int(label.sum()), int(len(label) - label.sum())
    if not n1 or not n0:
        return float("nan")
    ranks = np.empty(len(score))
    ranks[np.argsort(score)] = np.arange(1, len(score) + 1)
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _sulfur_forecast_metrics(train_end: pd.Timestamp, sigma_lims: float) -> dict:
    """Прогноз серы на горизонт решения: оценка по снимку за `horizon` часов до отбора проверяется по анализу ЛИМС.

    Сравнение: прогноз системы, «персистентность» (оценка на момент снимка без прогноза) и климатология."""
    bus = MessageBus()
    data, quality = DataAgent(bus), QualityAgent(bus)
    horizon = int(v(load_cfg()["sensor_fusion"]["forecast_horizon_h"]))
    lims_s = data.lims[(data.lims["point"] == "HT2") & (data.lims["parameter"] == "Mg.Sulfur") & (data.lims["time"] >= train_end)]
    y, fore, sig, now = [], [], [], []
    for _, row in lims_s.iterrows():
        snap = data.snapshot(row["time"] - pd.Timedelta(hours=horizon))
        if not snap.ho_running or snap.critical():
            continue
        s_f, sig_f = quality._forecast_sulfur(snap)
        y.append(row["value"]), fore.append(s_f), sig.append(sig_f), now.append(snap.sulfur_state.get("S_hat", np.nan))
    y, fore, sig, now = map(np.array, (y, fore, sig, now))
    ok = np.isfinite(fore) & np.isfinite(sig) & np.isfinite(now)
    y, fore, sig, now = y[ok], fore[ok], sig[ok], now[ok]
    total = np.sqrt(sig ** 2 + sigma_lims ** 2)
    label = (y > 10).astype(int)
    p = 1 - norm.cdf((10 - fore) / total)
    return {"horizon_h": horizon, "n": int(ok.sum()), "mae": float(np.abs(y - fore).mean()),
            "mae_persistence": float(np.abs(y - now).mean()), "auc": _auc(fore, label),
            "coverage_90": float(np.mean((y >= norm.ppf(0.05, fore, total)) & (y <= norm.ppf(0.95, fore, total)))),
            "brier": float(np.mean((p - label) ** 2)), "brier_climatology": float(np.mean((label.mean() - label) ** 2))}


def _t95_metrics(train_end: pd.Timestamp) -> dict:
    """Причинная оценка T95 перед каждым анализом продукта, через DataAgent/QualityAgent."""
    bus = MessageBus()
    data, quality = DataAgent(bus), QualityAgent(bus)
    lims_t95 = data.lims[(data.lims["point"] == "HT2") & (data.lims["parameter"] == "95%.T")
                          & (data.lims["time"] >= train_end)]
    est, sig, actual = [], [], []
    for _, row in lims_t95.iterrows():
        snap = data.snapshot(row["time"] - pd.Timedelta(seconds=1))  # анализ на момент оценки ещё неизвестен
        qa = quality.assess(snap)
        est.append(qa.t95_est)
        sig.append(qa.t95_sigma)
        actual.append(row["value"])
    est, sig, actual = np.array(est), np.array(sig), np.array(actual)
    ok = np.isfinite(est) & np.isfinite(sig)
    mae = float(np.abs(est[ok] - actual[ok]).mean())
    y = (actual[ok] > 360).astype(int)
    p_exceed = 1 - norm.cdf((360 - est[ok]) / sig[ok])
    if len(set(y)) < 2:
        auc = float("nan")
    else:
        order = np.argsort(p_exceed)
        y_sorted = y[order]
        n1, n0 = y_sorted.sum(), len(y_sorted) - y_sorted.sum()
        ranks = np.arange(1, len(y_sorted) + 1)
        auc = float((ranks[y_sorted == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
    # новый анализ тоже содержит шум σ: интервал для лабораторного значения шире интервала для истинного T95
    total = np.sqrt(sig[ok] ** 2 + load_calib()["t95"]["sigma"] ** 2)
    lo, hi = norm.ppf(0.05, est[ok], total), norm.ppf(0.95, est[ok], total)
    coverage = float(np.mean((actual[ok] >= lo) & (actual[ok] <= hi)))
    brier = float(np.mean((1 - norm.cdf((360 - est[ok]) / total) - y) ** 2))
    brier_clim = float(np.mean((y.mean() - y) ** 2))
    risky = p_exceed > v(load_cfg()["spec"]["risk_alpha"])          # риск выше допустимой вероятности
    rate_risky = float(y[risky].mean()) if risky.any() else float("nan")
    rate_safe = float(y[~risky].mean()) if (~risky).any() else float("nan")
    high = p_exceed > action_threshold()                              # риск выше порога действия
    return {"n": int(ok.sum()), "share_above_360": float(y.mean()), "mae": mae, "auc": auc,
            "coverage_90": coverage, "brier": brier, "brier_climatology": brier_clim,
            "rate_when_risky": rate_risky, "rate_when_safe": rate_safe,
            "n_above_action_threshold": int(high.sum()),
            "rate_above_action_threshold": float(y[high].mean()) if high.any() else float("nan"),
            "mean_p_above_action_threshold": float(p_exceed[high].mean()) if high.any() else float("nan")}


def validation_report() -> dict:
    cfg, calib = load_cfg(), load_calib()
    train_end = pd.Timestamp(cfg["periods"]["train_end"])
    pak = pd.read_parquet(CACHE_DIR / "pak_sulfur.parquet")
    q21 = pd.read_parquet(CACHE_DIR / "q21_sulfur.parquet")
    running_frame = pd.read_parquet(CACHE_DIR / "running.parquet")
    running = running_frame["ho_running"]
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
    state = pd.read_parquet(CACHE_DIR / "state_sulfur.parquet")

    lims_s = lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur") & (lims["time"] >= train_end)]
    sulfur_metrics = _sulfur_estimators(pak, q21, state, lims_s)
    sulfur_metrics["Градиентный бустинг по режимным тегам"] = _gbm_regime_baseline(
        pd.read_parquet(CACHE_DIR / "ho.parquet"), pd.read_parquet(CACHE_DIR / "avt.parquet"),
        lims[(lims["point"] == "HT2") & (lims["parameter"] == "Mg.Sulfur")].sort_values("time"), train_end)
    sulfur_metrics["Фильтр только по ПАК"] = _sulfur_only_pak(pak, running, lims, calib)
    clim_rate = float((lims_s["value"] > 10).mean())
    coverage, brier, brier_clim, alert = _coverage_brier(state, lims_s, clim_rate, calib["sulfur"]["sigma_lims"])

    probability_metrics = _probability_calibration(state, lims_s, running_frame, calib)
    t95_metrics = _t95_metrics(train_end)
    forecast_metrics = _sulfur_forecast_metrics(train_end, calib["sulfur"]["sigma_lims"])
    summary_txt = (f"сера n={len(lims_s)}, доля>10={clim_rate:.1%}, Brier={brier:.4f} (климатология {brier_clim:.4f}); "
                    f"T95 n={t95_metrics['n']}, AUC={t95_metrics['auc']:.3f}")

    vak_df = evaluate_vak()
    vak_df.to_csv(OUTPUTS_DIR / "vak_regulation.csv", index=False)

    report = {
        "period": {"start": str(train_end), "end": str(pak.index.max())},
        "sulfur": {"n_analyses": len(lims_s), "share_above_10": clim_rate, "estimators": sulfur_metrics,
                   "coverage_90": coverage, "brier": brier, "brier_climatology": brier_clim, "alert": alert,
                   "probability_calibration": probability_metrics},
        "sulfur_forecast": forecast_metrics, "t95_risk": t95_metrics,
        "vak": {"n_comparisons": len(vak_df), "n_fit": int((vak_df["verdict"] == "пригодна").sum())},
        "summary": summary_txt,
    }
    with open(OUTPUTS_DIR / "validation.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)
    return report
