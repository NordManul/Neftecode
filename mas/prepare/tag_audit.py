"""Аудит справочника тегов: код 307, знак расхода, согласованность пар масса/объём.

Категория берётся из колонки «Физическая величина» уточнённого справочника («расход, т/ч», «перепад, МПа» и т.д.).
Если уточнённого справочника нет, для 24-2000 используется реконструкция по проверенной таблице
(mas.io.reference.HO_TAGS_RECONSTRUCTED), а для АВТ - префикс кода КИП плюс ключевые слова описания. Числовых
диапазонов правдоподобия по категориям нет: их не подтверждают материалы проекта. Проверки: тег целиком состоит из кода
ошибки 307; медиана расхода отрицательна (физически невозможна); пара «массовый и объёмный расход» одного потока
(по справочнику либо по данным) не связана линейно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = {"перепад", "мм_рт_ст", "давление", "температура", "сера", "уровень", "плотность", "нм3ч", "м3ч", "тч"}
FLOW_CATEGORIES = {"нм3ч", "м3ч", "тч"}
PREFIX_CATEGORY = {"T": "температура", "P": "давление", "D": "плотность", "L": "уровень", "Q": "сера", "W": "тч"}
PAIR_MIN_CORRELATION = 0.9  # сильная линейная связь массового и объёмного расхода одного потока
# dict - заявленные справочником пары (масса, объём), hyp - подсказанные данными.
PAIRS_24_2000 = {
    "dict": [("F9", "F15"), ("F17", "F26"), ("W10", "F1")],
    "hyp": [("F9", "F26"), ("F19", "F17")],
}


def _category_from_quantity(phys: str) -> str | None:
    """Категория диапазона по строке «Физическая величина» справочника; None, если строка не распознана."""
    text = phys.lower().replace("³", "3").replace(" ", "")
    unit = text.split(",")[-1]
    if "нм3/ч" in unit:
        return "нм3ч"
    if "м3/ч" in unit:
        return "м3ч"
    if "т/ч" in unit:
        return "тч"
    if "ммрт" in unit:
        return "мм_рт_ст"
    if text.startswith("перепад"):
        return "перепад"
    if "мпа" in unit:
        return "давление"
    if "кг/м3" in unit:
        return "плотность"
    if "ppm" in unit:
        return "сера"
    if unit == "%":
        return "уровень"
    if "°c" in unit:
        return "температура"
    return None


def _category(tag: str, desc: str | None, phys: str | None, unit: str) -> str:
    if phys:
        if phys in CATEGORIES:  # реконструированный справочник несёт категорию напрямую
            return phys
        parsed = _category_from_quantity(phys)
        if parsed:
            return parsed
    d = (desc or "").lower() if unit == "avt" else ""  # раздел 24-2000: свободный текст ненадёжен
    if "перепад" in d:
        return "перепад"
    if "рт.ст" in d:
        return "мм_рт_ст"
    if tag[0] == "F":
        return "м3ч" if ("объём" in d or "объем" in d) else "тч"
    return PREFIX_CATEGORY.get(tag[0], "тч")


def _median_running(s: pd.Series, running: pd.Series) -> float:
    ok = s[running]
    return float(ok.median()) if len(ok) else float("nan")


def _basic_checks(s_raw: pd.Series, running: pd.Series, category: str) -> tuple[list[str], float]:
    s = s_raw.replace(307.0, np.nan)
    med = _median_running(s, running)
    problems: list[str] = []
    if running.any() and s[running].notna().sum() == 0:
        problems.append("все значения = 307 (код ошибки, тег неисправен)")
    if category in FLOW_CATEGORIES and not np.isnan(med) and med < 0:
        problems.append("отрицательный расход")
    return problems, med


def _pair_result(df: pd.DataFrame, running: pd.Series, mass_tag: str, vol_tag: str) -> tuple[bool, float, float]:
    m, o = df.loc[running, mass_tag].replace(307.0, np.nan), df.loc[running, vol_tag].replace(307.0, np.nan)
    valid = m.notna() & o.notna()
    if int(valid.sum()) < 2:
        return False, float("nan"), float("nan")
    corr = float(m[valid].corr(o[valid]))
    return corr > PAIR_MIN_CORRELATION, corr, float((m[valid] / o[valid]).median())


def _cross_pair_notes(df: pd.DataFrame, running: pd.Series) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Возвращает (заметки по тегу, причина противоречия) для 24-2000."""
    notes: dict[str, list[str]] = {}
    hyp_role_ok: dict[str, tuple[str, bool]] = {}  # tag -> (role, подтверждена ли пара)
    for mass_tag, vol_tag in PAIRS_24_2000["hyp"]:
        ok, corr, ratio = _pair_result(df, running, mass_tag, vol_tag)
        notes.setdefault(mass_tag, []).append(f"hyp {mass_tag}/{vol_tag}: r={corr:.2f} отн={ratio:.3f} -> {'ок' if ok else 'нет'}")
        notes.setdefault(vol_tag, []).append(notes[mass_tag][-1])
        hyp_role_ok[mass_tag] = ("mass", ok)
        hyp_role_ok[vol_tag] = ("vol", ok)

    contradictory: dict[str, str] = {}
    dict_role_of: dict[str, str] = {}
    for mass_tag, vol_tag in PAIRS_24_2000["dict"]:
        dict_role_of[mass_tag], dict_role_of[vol_tag] = "mass", "vol"
        ok, corr, ratio = _pair_result(df, running, mass_tag, vol_tag)
        notes.setdefault(mass_tag, []).append(f"dict {mass_tag}/{vol_tag}: r={corr:.2f} отн={ratio:.3f} -> {'ок' if ok else 'нет'}")
        notes.setdefault(vol_tag, []).append(notes[mass_tag][-1])
        if ok:
            continue
        for tag, claimed_role in ((mass_tag, "mass"), (vol_tag, "vol")):
            role, hyp_ok = hyp_role_ok.get(tag, (None, False))
            if hyp_ok and role == claimed_role:
                continue  # подтверждён прошедшей hyp-парой в той же роли
            contradictory[tag] = f"пара {mass_tag}/{vol_tag} не подтверждена (r={corr:.2f}, отн={ratio:.3f})"
    for tag, (role, hyp_ok) in hyp_role_ok.items():
        dict_role = dict_role_of.get(tag)
        if hyp_ok and dict_role and dict_role != role:
            contradictory[tag] = f"данные указывают на {'объёмный' if role == 'vol' else 'массовый'} расход, справочник - противоположный"
    return notes, contradictory


def tag_audit(df: pd.DataFrame, dictionary: dict, running: pd.Series, unit: str) -> pd.DataFrame:
    """Один тег - одна строка: описание, категория, медиана, проблемы, вердикт."""
    cross_notes, cross_bad = ({}, {}) if unit != "24-2000" else _cross_pair_notes(df, running)

    rows = []
    for tag in df.columns:
        desc, phys = dictionary.get(tag, (None, None))
        category = _category(tag, desc, phys, unit)
        problems, med = _basic_checks(df[tag], running, category)
        if tag in cross_bad:
            problems.append(cross_bad[tag])
        verdict = "противоречие" if problems else "согласован"
        rows.append({
            "tag": tag, "unit": unit, "description": desc, "physical_quantity": phys,
            "category": category, "median_running": med, "problems": "; ".join(problems),
            "pair_notes": "; ".join(cross_notes.get(tag, [])), "verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)
