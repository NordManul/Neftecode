"""Сборка шага `prepare`: читает источники, чистит, аудирует справочник, пишет кэш и отчёт.

Единственная точка, где io/* и prepare/* модули собираются вместе.
"""
from __future__ import annotations

import json

import pandas as pd

from mas.common import CACHE_DIR, OUTPUTS_DIR, ensure_dirs, load_cfg, to_jsonable, v
from mas.io.lims import read_lims
from mas.io.pak import read_pak
from mas.io.reference import read_tag_dictionary, read_vak_formulas
from mas.io.telemetry import read_telemetry
from mas.prepare.lims_clean import clean_lims
from mas.prepare.quality_flags import clean_analyzer, clean_telemetry
from mas.prepare.tag_audit import tag_audit
from mas.prepare.thresholds import derive_thresholds

STEP_H = 10.0 / 60.0


def running_flags(avt_vals: pd.DataFrame, ho_vals: pd.DataFrame, thresholds: dict) -> tuple[pd.Series, pd.Series]:
    """Признак работы АВТ и ГО по порогам, найденным по данным (`derive_thresholds`)."""
    ho_running = (ho_vals["F9"] > thresholds["ho_feed_min_tph"]) & (ho_vals["T5"] > thresholds["ho_temp_min_c"])
    avt_running = avt_vals["F65"] > thresholds["avt_feed_min"]
    return avt_running, ho_running


def hours_since_start(running: pd.Series) -> pd.Series:
    """Часы с начала работы, причинно; счёт начинается заново после любой остановки (признак работы был выключен)."""
    valid_start = running & ~running.shift(1, fill_value=False)
    cycle_id = valid_start.cumsum().where(running)
    t_hours = pd.Series(range(len(running)), index=running.index, dtype=float) * STEP_H
    start_hours = t_hours.where(valid_start).groupby(cycle_id).transform("first")
    return (t_hours - start_hours).where(running)


def run_prepare() -> dict:
    ensure_dirs()
    cfg = load_cfg()
    report_dir = OUTPUTS_DIR / "data_report"

    avt_raw, ho_raw = read_telemetry()
    sentinels = set(v(cfg["data_quality"]["sentinel_values"]))
    avt_plain, ho_plain = avt_raw.mask(avt_raw.isin(sentinels)), ho_raw.mask(ho_raw.isin(sentinels))
    thresholds = derive_thresholds(avt_plain, ho_plain, pd.Timestamp(cfg["periods"]["train_end"]),
                                   {"ho": ["T5", "F9"], "avt": ["F32"]})
    with open(CACHE_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(thresholds), f, ensure_ascii=False, indent=2)
    avt_vals, avt_flags = clean_telemetry(avt_raw, cfg, thresholds["telemetry_freeze_steps"])
    ho_vals, ho_flags = clean_telemetry(ho_raw, cfg, thresholds["telemetry_freeze_steps"])
    avt_running, ho_running = running_flags(avt_vals, ho_vals, thresholds)

    pak = read_pak()
    pak_sulfur = clean_analyzer(pak, tuple(v(cfg["data_quality"]["pak_sulfur_valid"])))
    q21_sulfur = clean_analyzer(ho_vals["Q21"], tuple(v(cfg["data_quality"]["q21_sulfur_valid"])))

    lims_long, units_meta, n_non_numeric = read_lims()
    lims_clean_df, lims_rejected = clean_lims(lims_long, cfg)

    avt_dict, ho_dict, dict_version = read_tag_dictionary()
    avt_audit = tag_audit(avt_raw, avt_dict, avt_running, "avt")
    ho_audit = tag_audit(ho_raw, ho_dict, ho_running, "24-2000")
    vak_formulas, vak_version = read_vak_formulas()

    avt_vals.to_parquet(CACHE_DIR / "avt.parquet")
    ho_vals.to_parquet(CACHE_DIR / "ho.parquet")
    avt_flags.to_parquet(CACHE_DIR / "avt_flags.parquet")
    ho_flags.to_parquet(CACHE_DIR / "ho_flags.parquet")
    pd.DataFrame({
        "avt_running": avt_running, "ho_running": ho_running,
        "ho_hours_since_start": hours_since_start(ho_running),
    }).to_parquet(CACHE_DIR / "running.parquet")
    pak_sulfur.to_parquet(CACHE_DIR / "pak_sulfur.parquet")
    q21_sulfur.to_parquet(CACHE_DIR / "q21_sulfur.parquet")
    lims_clean_df.to_parquet(CACHE_DIR / "lims.parquet")

    lims_rejected.to_csv(report_dir / "lims_rejected.csv", index=False)
    units_meta.to_csv(report_dir / "lims_units.csv", index=False)
    pd.concat([avt_audit, ho_audit], ignore_index=True).to_csv(report_dir / "tag_audit.csv", index=False)
    with open(CACHE_DIR / "vak_formulas.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable({"version": vak_version, "formulas": vak_formulas}), f, ensure_ascii=False, indent=2)

    summary = {
        "telemetry_rows": int(len(avt_vals)),
        "lims_records_raw": int(len(lims_long)), "lims_records_clean": int(len(lims_clean_df)),
        "lims_rejected": int(len(lims_rejected)), "lims_non_numeric": int(n_non_numeric),
        "pak_sulfur_n": int(len(pak)),
        "pak_freeze_points": int((pak_sulfur.reason == "заморозка").sum()),
        "code307_cells": int((avt_flags == 1).sum().sum() + (ho_flags == 1).sum().sum()),
        "tag_dictionary_version": dict_version, "vak_formulas_version": vak_version,
        "vak_formula_count": int(len(vak_formulas)),
        "avt_contradictions": sorted(avt_audit.loc[avt_audit.verdict == "противоречие", "tag"].tolist()),
        "ho_contradictions": sorted(ho_audit.loc[ho_audit.verdict == "противоречие", "tag"].tolist()),
    }
    with open(CACHE_DIR / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, ensure_ascii=False, indent=2)
    return summary
