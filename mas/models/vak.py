"""Регламент «ВАК или собственная модель»: сверка модель × точка отбора по критериям пригодности.

Формулы ВАК записаны в двух нотациях (запятая-децималь и «x» умножения, либо обычный Python) -
нормализуются перед `eval`. LIMS-автопризнак («последний доступный анализ ГО т.2») - причинный
поиск по available_at, а не по времени отбора. Входные значения вне коридора p1-p99 обучающего периода
в рабочем режиме отбрасываются: единичные выбросы датчиков иначе разрушают среднечасовые значения формул и
линейную калибровку.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mas.common import CACHE_DIR, load_cfg, v

PARAM_MAP = {"D15": "D15", "T50": "50%.T", "T90": "90%.T", "T95": "95%.T", "EBP": "EBP.T", "IBP": "IBP.T",
             "CFPP": "CFPP", "I250": "I250", "I350": "I350", "CloudPoint": "CloudPoint", "ViscosityK": None}
GROUP_POINTS = {"240-350": ["AVT2", "AVT2.1", "AVT3"], "350": ["AVT1"], "350-500": ["AVT1"], "GODT": ["HT2"]}
LIMS_REFS = {"LIMS:24-2000.Pipeline.D15": "D15", "LIMS:24-2000.Pipeline.95%.T": "95%.T",
             "LIMS.D15": "D15", "LIMS.95%.T": "95%.T"}  # исходный лист «ВАК» / уточнённый файл
TAG_RE = re.compile(r"\b[TPFWLQ]\d{1,2}\b")


def _normalize(formula: str) -> str:
    f = formula.replace("×", "*").replace("−", "-").replace(" ", "")
    if re.search(r"\d,\d", f):
        f = f.replace(",", ".")
    f = re.sub(r"(?<=\d)x", "*", f)
    for lit, param in LIMS_REFS.items():
        f = f.replace(lit, f"__lims_{param.replace('%', 'pct').replace('.', '_')}__")
    return f


INPUT_CORRIDOR = (0.01, 0.99)  # коридор p1-p99 обучающего периода, как у контролируемых тегов (ТЗ, «Правило границ»)


def _mask_inputs(telemetry: pd.DataFrame, running: pd.Series, tags: list[str], train_end: pd.Timestamp) -> pd.DataFrame:
    """Копия телеметрии, где значения тегов формулы вне коридора p1-p99 обучающего периода (рабочий режим) заменены на NaN."""
    out = telemetry.copy()
    base = telemetry[running.reindex(telemetry.index).fillna(False) & (telemetry.index < train_end)]
    for tag in tags:
        lo, hi = base[tag].quantile(INPUT_CORRIDOR[0]), base[tag].quantile(INPUT_CORRIDOR[1])
        out[tag] = telemetry[tag].where((telemetry[tag] >= lo) & (telemetry[tag] <= hi))
    return out


def _parse_name(name: str) -> tuple[str, str, str]:
    unit, group, param = name.split(":")
    return unit, group, param


def _eval_series(formula: str, telemetry: pd.DataFrame, running: pd.Series, lims_ht2: pd.DataFrame) -> pd.Series | None:
    f = _normalize(formula)
    tags = sorted(set(TAG_RE.findall(f)))
    if any(t not in telemetry.columns for t in tags):
        return None
    ns: dict[str, pd.Series] = {t: telemetry[t] for t in tags}
    for lit, param in LIMS_REFS.items():
        key = f"__lims_{param.replace('%', 'pct').replace('.', '_')}__"
        if key in f:
            sub = lims_ht2[lims_ht2["parameter"] == param].sort_values("available_at")
            asof = pd.merge_asof(pd.DataFrame({"t": telemetry.index}), sub[["available_at", "value"]],
                                  left_on="t", right_on="available_at", direction="backward", allow_exact_matches=False)
            ns[key] = pd.Series(asof["value"].to_numpy(), index=telemetry.index)
    try:
        result = eval(f, {"__builtins__": {}}, ns)  # noqa: S307 - формулы из проверенного файла пакета
    except Exception:
        return None
    return result.where(running) if isinstance(result, pd.Series) else None


# Формула пригодна, если на отложенном периоде превосходит тривиальные оценки, которые заменила бы: среднее значение
# (R² > 0) и последний анализ ЛИМС (абсолютные ошибки после линейной калибровки по обучающему периоду значимо меньше, чем у
# последнего анализа: парный критерий Вилкоксона, односторонний, на уровне допустимой вероятности `spec.risk_alpha`).
# Минимальные объёмы - то, что требуется для подбора прямой по обучающему периоду (3 точки) и для того, чтобы парный критерий
# мог дать значимый результат на уровне alpha (наименьшее n, при котором 0.5**n < alpha).
MIN_TRAIN = 3


def _metrics(pred: pd.Series, actual: pd.Series, train_end: pd.Timestamp, alpha: float) -> dict:
    df = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    train, test = df[df.index < train_end], df[df.index >= train_end]
    if len(train) < MIN_TRAIN or len(test) < int(np.ceil(np.log2(1 / alpha))):
        return {"verdict": "недостаточно данных", "r2_train": None, "r2_test": None, "mae_raw": None,
                "mae_calibrated": None, "mae_persistence": None, "n_train": len(train), "n_test": len(test)}

    def r2(sub):
        ss_res = ((sub["actual"] - sub["pred"]) ** 2).sum()
        ss_tot = ((sub["actual"] - sub["actual"].mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    a, b = np.polyfit(train["pred"], train["actual"], 1)
    test_calib = a * test["pred"] + b
    mae_raw = float((test["pred"] - test["actual"]).abs().mean())
    mae_cal = float((test_calib - test["actual"]).abs().mean())
    persistence = test["actual"].shift(1).ffill()
    mae_pers = float((persistence - test["actual"]).abs().mean())
    r2_test = float(r2(test))
    errors = pd.DataFrame({"cal": (test_calib - test["actual"]).abs(), "pers": (persistence - test["actual"]).abs()}).dropna()
    better = len(errors) >= 1 and bool((errors["cal"] != errors["pers"]).any()) and         wilcoxon(errors["cal"], errors["pers"], alternative="less").pvalue < alpha
    verdict = "пригодна" if (r2_test > 0 and better) else "не пригодна"
    return {"verdict": verdict, "r2_train": float(r2(train)), "r2_test": r2_test, "mae_raw": mae_raw,
            "mae_calibrated": mae_cal, "mae_persistence": mae_pers, "n_train": len(train), "n_test": len(test)}


def evaluate_vak() -> pd.DataFrame:
    cfg = load_cfg()
    train_end = pd.Timestamp(cfg["periods"]["train_end"])
    with open(CACHE_DIR / "vak_formulas.json", "r", encoding="utf-8") as f:
        formulas = [tuple(p) for p in json.load(f)["formulas"]]
    avt, ho = pd.read_parquet(CACHE_DIR / "avt.parquet"), pd.read_parquet(CACHE_DIR / "ho.parquet")
    running = pd.read_parquet(CACHE_DIR / "running.parquet")
    lims = pd.read_parquet(CACHE_DIR / "lims.parquet")
    lims_ht2 = lims[lims["point"] == "HT2"]

    rows = []
    for name, formula in formulas:
        unit, group, param = _parse_name(name)
        telemetry, run_mask = (ho, running["ho_running"]) if unit == "24-2000" else (avt, running["avt_running"])
        tags = sorted(set(TAG_RE.findall(_normalize(formula))))
        if any(t not in telemetry.columns for t in tags):
            rows.append({"model": name, "point": None, "verdict": "ошибка разбора формулы"})
            continue
        pred = _eval_series(formula, _mask_inputs(telemetry, run_mask, tags, train_end), run_mask, lims_ht2)
        lims_param = PARAM_MAP.get(param)
        points = GROUP_POINTS.get(group, [])
        if pred is None:
            rows.append({"model": name, "point": None, "verdict": "ошибка разбора формулы"})
            continue
        if lims_param is None or not points:
            rows.append({"model": name, "point": None, "verdict": "нет сопоставимого показателя ЛИМС"})
            continue
        for point in points:
            actual = lims[(lims["point"] == point) & (lims["parameter"] == lims_param)]
            actual_s = pd.Series(actual["value"].to_numpy(), index=actual["time"].to_numpy())
            pred_daily, actual_daily = pred.resample("1h").mean(), actual_s.resample("1h").mean()
            m = _metrics(pred_daily, actual_daily, train_end, v(cfg["spec"]["risk_alpha"]))
            rows.append({"model": name, "point": point, **m})
    return pd.DataFrame(rows)
