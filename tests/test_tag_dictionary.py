"""Справочник тегов: разбор уточнённого файла и категории проверки диапазонов."""
from __future__ import annotations

import openpyxl
import pytest

from mas.io.reference import HO_TAGS_RECONSTRUCTED, _read_v2_dict_sheet
from mas.prepare.tag_audit import CATEGORIES, _category, _category_from_quantity


@pytest.mark.parametrize("quantity, category", [
    ("расход, м³/ч", "м3ч"), ("расход газа, нм³/ч", "нм3ч"), ("расход ВСГ, нм³/ч", "нм3ч"), ("масс.расход, т/ч", "тч"),
    ("расход, т/ч", "тч"), ("давление, МПа", "давление"), ("перепад, МПа", "перепад"), ("давление, мм рт.ст.", "мм_рт_ст"),
    ("температура, °C", "температура"), ("°C", "температура"), ("плотность, кг/м3", "плотность"), ("уровень, %", "уровень"),
    ("сера, ppm", "сера"),
])
def test_quantity_string_maps_to_range_category(quantity, category):
    assert _category_from_quantity(quantity) == category
    assert category in CATEGORIES


def test_unknown_quantity_falls_back_to_tag_prefix():
    assert _category_from_quantity("вязкость, сСт") is None
    assert _category("T7", "Температура", "вязкость, сСт", "avt") == "температура"


def test_reconstructed_categories_are_still_accepted():
    assert all(_category(tag, desc, phys, "24-2000") == phys for tag, (desc, phys) in HO_TAGS_RECONSTRUCTED.items())


@pytest.mark.parametrize("header", [("Колонка", "Описание (КИП)", "Физическая величина"),
                                     ("Колонка CSV", "Описание", "Физическая величина")])
def test_both_header_variants_of_refined_dictionary_are_read(header):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(header)
    ws.append(("T5", "Температура ГСС на выходе", "температура, °C"))
    assert _read_v2_dict_sheet(ws) == {"T5": ("Температура ГСС на выходе", "температура, °C")}
