"""Сборка `calibrated.json` (параметры, используемые агентами) и `outputs/calibration_report.json` (диагностика).

Порядок: динамика серы (T6) -> отклик/литература/b95 (T7) -> предварительный Treq -> γ -> коридоры.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mas.calibrate.bands import calibrate_bands, reference_feed
from mas.calibrate.response import calibrate_b95, calibrate_gamma, calibrate_sulfur_response, literature_beta
from mas.calibrate.sulfur_filter import fit_sulfur_filter, sulfur_diagnostics
from mas.calibrate.probability import fit_probability_map
from mas.calibrate.risk import background_exceedance, sulfur_level_sd
from mas.calibrate.startup import startup_relaxation
from mas.calibrate.t95_noise import calibrate_t95_noise
from mas.common import CACHE_DIR, OUTPUTS_DIR, load_cfg, to_jsonable, v
from mas.models.catalyst import dp_index, treq_series
from mas.models.kalman import kalman_sulfur



def _paired_offset(lims: pd.DataFrame, train_end: pd.Timestamp) -> tuple[float, float]:
    a = lims[(lims["point"] == "HT1") & (lims["parameter"] == "95%.T") & (lims["time"] < train_end)]
    b = lims[(lims["point"] == "HT2") & (lims["parameter"] == "95%.T") & (lims["time"] < train_end)]
    da = pd.Series(a["value"].to_numpy(), index=a["time"].dt.floor("D").to_numpy())
    db = pd.Series(b["value"].to_numpy(), index=b["time"].dt.floor("D").to_numpy())
    diff = (db.groupby(level=0).mean() - da.groupby(level=0).mean()).dropna()
    return float(diff.mean()), float(diff.std())


LIMS_AGE_KEYS = (("HT2", "95%.T"), ("HT2", "D15"), ("HT2", "CetaneNumber"), ("HT1", "95%.T"))
AGE_LIMIT_INTERVALS = 3  # окно информации - три медианных интервала между анализами


def _lims_age_limits(lims: pd.DataFrame, train_end: pd.Timestamp) -> dict:
    out = {}
    for point, param in LIMS_AGE_KEYS:
        t = lims[(lims["point"] == point) & (lims["parameter"] == param) & (lims["time"] < train_end)]["time"].sort_values()
        out[f"{point}:{param}"] = float(AGE_LIMIT_INTERVALS * t.diff().dt.total_seconds().median() / 3600.0)
    return out


def run_calibration() -> dict:
    cfg = load_cfg()
    train_end = pd.Timestamp(cfg["periods"]["train_end"])
    avt_vals, ho_vals = pd.read_parquet(CACHE_DIR / "avt.parquet"), pd.read_parquet(CACHE_DIR / "ho.parquet")
    running = pd.read_parquet(CACHE_DIR / "running.parquet")
    avt_running, ho_running = running["avt_running"], running["ho_running"]
    pak, q21 = pd.read_parquet(CACHE_DIR / "pak_sulfur.parquet"), pd.read_parquet(CACHE_DIR / "q21_sulfur.parquet")
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")

    sulfur_params, sulfur_fit = fit_sulfur_filter()
    sulfur_diag = sulfur_diagnostics(pak, q21, lims, ho_running, sulfur_params, train_end)
    resp = calibrate_sulfur_response(ho_vals, pak, ho_running, train_end)
    lit = literature_beta(tuple(v(cfg["kinetics"]["ea_kj_mol"])), sulfur_params["mu"],
                           v(cfg["kinetics"]["feed_sulfur_ppm"]), 643.0)
    b95 = calibrate_b95(avt_vals, avt_running, lims, train_end)
    offset, sd_diff = _paired_offset(lims, train_end)
    age_limits = _lims_age_limits(lims, train_end)
    t95_noise = calibrate_t95_noise(lims, avt_vals["F32"], b95["medium"], train_end, age_limits["HT2:95%.T"])
    startup = startup_relaxation(q21, running["ho_hours_since_start"], train_end)
    risk = background_exceedance(lims, running, cfg, train_end, startup["window_h"])
    level_sd = sulfur_level_sd(lims, running, train_end, startup["window_h"], sulfur_params["sigma_lims"])
    feed_ref = reference_feed(ho_vals, ho_running, train_end)

    beta_t = {"weak": resp["beta_t"]["beta"], "medium": lit["beta_t_medium"], "strong": lit["beta_t_strong"]}
    beta_f = {"weak": resp["beta_f"]["beta"], "strong": lit["beta_f_strong"]}
    beta_f["medium"] = (beta_f["weak"] + beta_f["strong"]) / 2
    b95_levels = {"weak": b95["weak"], "medium": b95["medium"], "strong": b95["strong"]}

    state, _ = kalman_sulfur({"pak": pak, "q21": q21}, lims, ho_running, sulfur_params)
    probability = fit_probability_map(state, lims, running, cfg, train_end, startup["window_h"])
    t5_1h = ho_vals["T5"].where(ho_running).resample("1h").mean()
    feed_1h = ho_vals["F9"].where(ho_running).resample("1h").mean()
    s_hat_1h = state["S_hat"].resample("1h").mean()
    prelim_calib = {"kinetics": {"beta_f": beta_f, "beta_t": beta_t}, "reference": {"feed_tph": feed_ref}}
    daily = treq_series(t5_1h, feed_1h, s_hat_1h, ho_running.resample("1h").max(), prelim_calib, cfg)

    ht1_daily = (lims[(lims["point"] == "HT1") & (lims["parameter"] == "95%.T") & (lims["time"] < train_end)]
                 .assign(day=lambda d: d["time"].dt.floor("D")).groupby("day")["value"].mean())
    gamma = calibrate_gamma(daily.loc[:train_end, "treq"], ht1_daily, abs(beta_t["medium"]))

    bands = calibrate_bands(avt_vals, ho_vals, avt_running, ho_running, cfg, feed_ref)
    p8_1h = ho_vals["P8"].where(ho_running).resample("1h").mean()
    dp_ratio = dp_index(p8_1h, feed_1h, ho_running.resample("1h").max(), feed_ref, bands["dp_norm_base"])["dp_ratio"]
    dp_ratio_p99 = float(np.nanpercentile(dp_ratio.loc[:train_end], 99))

    calibrated = {
        "sulfur": sulfur_params,
        "kinetics": {"beta_t": beta_t, "beta_f": beta_f, "b95": b95_levels, "gamma": gamma},
        "t95": {"sigma": t95_noise["sigma"], "drift": t95_noise["drift"], "offset": offset, "sd_diff": sd_diff},
        "bands": bands["corridors"], "dp_norm_base": bands["dp_norm_base"],
        "reference": {"feed_tph": feed_ref},
        "reliability": {"t5_roc4h_p95": bands["t5_roc4h_p95"], "t5_roc4h_p99": bands["t5_roc4h_p99"],
                        "dp_ratio_p99": dp_ratio_p99},
        "lims_age_limit_h": age_limits,
        "startup": {"window_h": startup["window_h"]},
        "uncertainty": {"sulfur_level_sd": level_sd},
        "probability": probability,
    }
    report = {"sulfur": sulfur_diag, "beta_t": resp["beta_t"], "beta_f": resp["beta_f"], "b95": b95,
              "t95": {"n_train_analyses": t95_noise["n"]}, "risk": risk, "sulfur_fit": sulfur_fit, "startup": startup}
    with open(CACHE_DIR.parent / "config" / "calibrated.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(calibrated), f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(OUTPUTS_DIR / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2, sort_keys=True)
    return calibrated
