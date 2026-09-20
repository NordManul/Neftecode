"""Формирование `docs/03_РЕЗУЛЬТАТЫ.md`: все числа берутся только из файлов расчёта, вручную не вводятся.

Читает `cache/data_summary.json`, `config/calibrated.json`, `outputs/validation.json`,
`outputs/backtest_summary.json`, `outputs/vak_regulation.csv`.
"""
from __future__ import annotations

import json

import pandas as pd

from mas.common import CACHE_DIR, DOCS_DIR, OUTPUTS_DIR, ROOT


def _load(path, default=None):
    if not path.exists():
        return default
    if path.suffix == ".json":
        return json.load(open(path, encoding="utf-8"))
    return pd.read_csv(path)


def _fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


_FORECAST_LABELS = {
    "n": "анализов", "horizon_h": "горизонт, ч", "mae": "MAE прогноза, мг/кг", "mae_persistence": "MAE «последняя оценка без изменений», мг/кг",
    "coverage_90": "покрытие 90%-го интервала", "auc": "AUC превышения нормы", "brier": "Brier", "brier_climatology": "Brier климатологии",
}
_T95_LABELS = {
    "n": "анализов", "mae": "MAE, °C", "coverage_90": "покрытие 90%-го интервала", "auc": "AUC превышения 360 °C", "brier": "Brier",
    "brier_climatology": "Brier климатологии", "share_above_360": "доля анализов выше 360 °C",
    "rate_when_risky": "частота превышения при P > 5 %", "rate_when_safe": "частота превышения при P ≤ 5 %",
}


def _load_json(path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _holdout_section(h: dict) -> list[str]:
    """Внутренняя проверка: подбор по началу обучающего периода, оценка на следующем отрезке (`run.py holdout`)."""
    fit_end, w = h["fit_period"]["end"], h["check_period"]
    rows = ["### Внутренняя проверка настройки фильтра серы",
            f"Параметры подобраны только по данным до {fit_end} (глобальный поиск без стартовой точки рабочей калибровки), "
            f"оценка выполнена на анализах ЛИМС с {w['start']} до {w['end']}; рабочие параметры (подбор по всему обучающему периоду) "
            "показаны для сравнения и на этом отрезке не независимы.", "",
            "| Оценка серы | MAE, мг/кг | AUC | n |", "|---|---|---|---|"]
    out, work = h["out_of_sample"], h["working_parameters_same_window"]
    for name, m in out["estimators"].items():
        label = "Фильтр, параметры до " + fit_end if name == "Фильтр по двум приборам" else name
        rows.append(f"| {label} | {_fmt(m['mae'])} | {_fmt(m['auc'])} | {m['n']} |")
    m = work["estimators"]["Фильтр по двум приборам"]
    rows.append(f"| Фильтр, рабочие параметры (для сравнения) | {_fmt(m['mae'])} | {_fmt(m['auc'])} | {m['n']} |")
    rows += ["", _dict_table({
        "Покрытие 90%-го интервала (параметры до разделения)": out["coverage_90"], "Brier": out["brier"],
        "Brier климатологии": out["brier_climatology"], "Доля анализов выше 10 мг/кг": out["share_above_10"],
        "Прогноз на 4 ч: MAE, мг/кг": out["forecast"]["mae"], "Прогноз на 4 ч: MAE «без изменений»": out["forecast"]["mae_persistence"],
        "Прогноз на 4 ч: покрытие 90%-го интервала": out["forecast"]["coverage_90"],
        "Прогноз на 4 ч: AUC превышения": out["forecast"]["auc"]}), ""]
    return rows


def _rename(d: dict, labels: dict) -> dict:
    return {labels.get(k, k): x for k, x in d.items()}


def _dict_table(d: dict, header: tuple[str, str] = ("Показатель", "Значение")) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {_fmt(v)} |")
    return "\n".join(lines)


def build_results_doc() -> str:
    prep = _load(CACHE_DIR / "data_summary.json", {})
    calib = _load(ROOT / "config" / "calibrated.json", {})
    report = _load(OUTPUTS_DIR / "calibration_report.json", {})
    val = _load(OUTPUTS_DIR / "validation.json", {})
    bt = _load(OUTPUTS_DIR / "backtest_summary.json", {})
    vak = _load(OUTPUTS_DIR / "vak_regulation.csv")

    parts = ["# Результаты", "", "Показатели получены командой `python run.py docs` по файлам в `cache/` и `outputs/`.", ""]

    parts += ["## 1. Данные (`prepare`)", _dict_table({
        "Строк телеметрии": prep.get("telemetry_rows"), "ЛИМС до/после очистки":
            f"{prep.get('lims_records_raw')} / {prep.get('lims_records_clean')}",
        "Отбраковано ЛИМС": prep.get("lims_rejected"), "Точек ПАК (сера)": prep.get("pak_sulfur_n"),
        "Заморозка ПАК, точек": prep.get("pak_freeze_points"), "Ячеек с кодом 307": prep.get("code307_cells"),
        "Версия справочника": prep.get("tag_dictionary_version"), "Версия формул ВАК": prep.get("vak_formulas_version"),
        "Противоречия 24-2000": prep.get("ho_contradictions"), "Противоречия АВТ": prep.get("avt_contradictions"),
    }), ""]

    sulfur_c, t95_c, risk_c = calib.get("sulfur", {}), calib.get("t95", {}), calib.get("risk", {})
    parts += ["## 2. Калибровка", "### Фильтр серы", _dict_table({
        "mu, мг/кг": sulfur_c.get("mu"), "phi_s (медленная составляющая)": sulfur_c.get("phi_s"),
        "постоянная времени медленной составляющей, ч": report.get("sulfur", {}).get("tau_slow_h"),
        "phi_fast (быстрая составляющая)": sulfur_c.get("phi_fast"),
        "sigma_ЛИМС, мг/кг": sulfur_c.get("sigma_lims"),
        "сдвиг показаний ПАК, шагов": sulfur_c.get("pak", {}).get("shift"),
        "запаздывание Q21, шагов": sulfur_c.get("q21", {}).get("delay"),
        "median NIS (train)": report.get("sulfur", {}).get("median_nis_train"),
        "корреляция ПАК/Q21": report.get("sulfur", {}).get("corr_pak_q21"),
        "анализов ЛИМС в критерии подбора": report.get("sulfur_fit", {}).get("n_targets"),
    }), "", "### T95 и фоновый риск", _dict_table({
        "sigma лабораторного анализа T95, °C": t95_c.get("sigma"),
        "рост дисперсии старого анализа, °C²/сут": t95_c.get("drift"),
        "анализов T95 в обучающем периоде": report.get("t95", {}).get("n_train_analyses"),
        "фон превышения нормы по сере (обучение)": risk_c.get("sulfur_background"),
        "фон превышения нормы по T95 (обучение)": risk_c.get("t95_background"),
    }), ""]

    parts += ["## 3. Проверка на отложенном периоде", ""]
    if val:
        parts += ["### Сера: оценка перед анализом", "| Оценщик | MAE | AUC | n |", "|---|---|---|---|"]
        for name, m in val.get("sulfur", {}).get("estimators", {}).items():
            if not isinstance(m, dict):
                continue
            parts.append(f"| {name} | {_fmt(m['mae'])} | {_fmt(m['auc'])} | {m['n']} |")
        s = val["sulfur"]
        parts += ["", _dict_table({"Покрытие 90%-го интервала": s.get("coverage_90"), "Brier": s.get("brier"),
                                    "Brier климатологии": s.get("brier_climatology")}), ""]
        if val.get("sulfur_forecast"):
            parts += ["### Прогноз серы на горизонт решения", _dict_table(_rename(val["sulfur_forecast"], _FORECAST_LABELS)), ""]
        parts += ["### Риск по T95", _dict_table(_rename(val.get("t95_risk", {}), _T95_LABELS)), ""]
    holdout = _load_json(OUTPUTS_DIR / "internal_holdout.json")
    if holdout:
        parts += _holdout_section(holdout)
    parts += ["### Регламент ВАК", f"Сверок: {len(vak) if vak is not None else 0}; пригодных: "
              f"{int((vak['verdict'] == 'пригодна').sum()) if vak is not None else 0}.", ""]

    parts += ["## 4. Бэктест (2025-01-01 … 2026-08-08)", ""]
    if bt:
        parts += [f"Циклов: {bt.get('n_cycles')}. Эскалаций: {_fmt(bt.get('escalation_share'))}.", ""]
        parts += ["### Распределение статусов", _dict_table(bt.get("status_distribution", {})), ""]
        parts += ["### Драйверы действий (доля рабочих циклов, где риск без действия выше порога запуска)", _dict_table(bt.get("driver_share_of_working", {})), ""]
        parts += [f"Эпизодов действий в месяц: {_fmt(bt.get('episodes_per_month'))}, "
                  f"медианная длина: {_fmt(bt.get('episode_median_length'))} цикл. (шаг цикла 6 ч).", ""]
        parts += ["### Причины отказа", _dict_table(bt.get("failure_reasons", {})), ""]

    text = "\n".join(parts)
    (DOCS_DIR / "03_РЕЗУЛЬТАТЫ.md").write_text(text, encoding="utf-8")
    return text
