"""Проверка модели отклика на записанных данных: ступенчатые изменения и согласие с действиями операторов.

Действие рекомендации на установке проверить нельзя, поэтому отклик серы на изменение T5 и загрузки проверяется по
ступенчатым изменениям, которые операторы делали сами (естественные эксперименты, те же события и окна, что при калибровке),
отдельно на обучающем и отложенном периодах и по двум анализаторам. Наблюдения получены в замкнутом контуре: операторы меняют
режим в ответ на рост серы, поэтому оценка по данным смещена к нулю по модулю и служит нижней границей отклика.
Вторая проверка: как операторы сами меняли T5 и загрузку после циклов, на которых система подавала сигнал о риске.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mas.calibrate import response as R
from mas.common import CACHE_DIR, OUTPUTS_DIR, action_threshold, load_calib, load_cfg, to_jsonable
from mas.models.kinetics import LEVELS

LEVEL_NAMES = {"weak": "слабая", "medium": "средняя", "strong": "сильная"}


def natural_experiments(t5: pd.Series, f9: pd.Series, sulfur: dict[str, pd.Series], running: pd.Series, calib: dict,
                        split: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Отклик ln(S) на ступенчатые изменения T5 (на °C) и загрузки (на единицу ln F9); в какие интервалы попадают члены ансамбля."""
    t5_h, f9_h = R._hourly(t5, running, end), R._hourly(f9, running, end)
    d_t5, d_lnf9 = t5_h - t5_h.shift(3), np.log(f9_h) - np.log(f9_h.shift(3))
    levers = (("T5, на °C", R._find_events(d_t5, d_lnf9, *R.TEMP_EVENT), d_t5, "beta_t"),
              ("загрузка F9, на ln-единицу", R._find_events(d_lnf9, d_t5, *R.LOAD_EVENT), d_lnf9, "beta_f"))
    rows = []
    for lever, events, x, key in levers:
        for source, series in sulfur.items():
            sulfur_h = R._hourly(series, running, end)
            for period, sel in (("обучение", events[events < split]), ("отложенный период", events[events >= split])):
                r = R._response_slope(sulfur_h, sel, x)
                if r["n"] == 0:
                    continue
                lo, hi = r["ci90"]
                inside = [LEVEL_NAMES[lvl] for lvl in LEVELS if lo <= calib["kinetics"][key][lvl] <= hi]
                rows.append({"lever": lever, "source": source, "period": period, "n": r["n"], "beta": r["beta"], "lo": lo, "hi": hi,
                             "ensemble": ", ".join(inside) if inside else "нет"})
    return rows


def _window_mean(series: pd.Series, t: pd.Timestamp, a: str, b: str) -> float:
    return float(series.loc[t + pd.Timedelta(a): t + pd.Timedelta(b)].mean())


def operator_concordance(cycles: pd.DataFrame, t5: pd.Series, f9: pd.Series, running: pd.Series, threshold: float) -> dict:
    """Изменение T5 и ln F9, сделанное операторами между окнами 6-3 ч до цикла и 2-8 ч после него, для циклов с сигналом
    (расчётная вероятность превышения выше `threshold`) и остальных циклов без изменений; разность и 90 %-й интервал бутстрепа."""
    t5_h, f9_h = t5.where(running).resample("1h").mean(), f9.where(running).resample("1h").mean()
    rows = []
    for t, status, p in zip(cycles["t"], cycles["status"], pd.to_numeric(cycles["P_now"], errors="coerce")):
        if status.startswith("НАБЛЮДЕНИЕ") or not np.isfinite(p):
            continue
        d5 = _window_mean(t5_h, t, "2h", "8h") - _window_mean(t5_h, t, "-6h", "-3h")
        df = np.log(_window_mean(f9_h, t, "2h", "8h") / _window_mean(f9_h, t, "-6h", "-3h"))
        if np.isfinite(d5) and np.isfinite(df):
            rows.append((p > threshold, status.startswith("БЕЗ"), d5, df))
    d = pd.DataFrame(rows, columns=["signal", "none", "d_t5", "d_lnf9"])
    sig, rest = d[d["signal"]], d[(~d["signal"]) & d["none"]]
    rng = np.random.default_rng(R.SEED)

    def ci(a: np.ndarray, b: np.ndarray) -> list[float]:
        diffs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(R.N_BOOT)]
        return [float(x) for x in np.percentile(diffs, [5, 95])]

    groups = [{"group": name, "n": int(len(g)), "d_t5_mean": float(g["d_t5"].mean()), "d_f9_mean_pct": float(100 * (np.exp(g["d_lnf9"].mean()) - 1))}
              for name, g in (("сигнал о риске (вероятность выше 1/2)", sig), ("без сигнала, статус «без изменений»", rest))]
    return {"groups": groups,
            "difference": {"d_t5": float(sig["d_t5"].mean() - rest["d_t5"].mean()), "d_t5_ci": ci(sig["d_t5"].to_numpy(), rest["d_t5"].to_numpy()),
                           "d_lnf9": float(sig["d_lnf9"].mean() - rest["d_lnf9"].mean()), "d_lnf9_ci": ci(sig["d_lnf9"].to_numpy(), rest["d_lnf9"].to_numpy())},
            "note": "Изменение между окнами 6-3 ч до цикла и 2-8 ч после него (почасовые средние в рабочем режиме); разность и 90 %-й интервал бутстрепа."}


def run_effect_check() -> dict:
    cfg, calib = load_cfg(), load_calib()
    split, end = pd.Timestamp(cfg["periods"]["train_end"]), pd.Timestamp(cfg["periods"]["test_end"]) + pd.Timedelta(days=1)
    ho = pd.read_parquet(CACHE_DIR / "ho.parquet")
    running = pd.read_parquet(CACHE_DIR / "running.parquet")["ho_running"]
    pak, q21 = pd.read_parquet(CACHE_DIR / "pak_sulfur.parquet"), pd.read_parquet(CACHE_DIR / "q21_sulfur.parquet")
    sulfur = {"ПАК": pak["raw"].where(pak["reason"] == ""), "Q21": q21["raw"].where(q21["reason"] == "")}
    rows = natural_experiments(ho["T5"], ho["F9"], sulfur, running, calib, split, end)
    cycles = pd.read_csv(OUTPUTS_DIR / "backtest_cycles.csv", parse_dates=["t"])
    concordance = operator_concordance(cycles, ho["T5"], ho["F9"], running, action_threshold(cfg, "sulfur"))
    report = {"natural_experiments": {"rows": rows, "note": "Наблюдения замкнутого контура: операторы меняют режим в ответ на рост серы, "
                                      "поэтому оценка по данным смещена к нулю по модулю и служит нижней границей отклика."},
              "operator_concordance": concordance}
    with open(OUTPUTS_DIR / "effect_check.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)
    return report
