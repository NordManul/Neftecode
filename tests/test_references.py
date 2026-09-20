"""Источники допущений: у каждого допущения есть ссылка, нормативные значения совпадают с документами."""
from __future__ import annotations

import pytest

from mas.common import load_cfg, load_refs, node_refs, v


def _nodes(node, path=""):
    if isinstance(node, dict) and "src" in node:
        yield path, node
    elif isinstance(node, dict):
        for k, x in node.items():
            yield from _nodes(x, f"{path}.{k}" if path else k)


def test_every_assumption_has_known_source():
    refs = load_refs()
    for path, node in _nodes(load_cfg()):
        ids = node.get("ref")
        ids = ids if isinstance(ids, list) else [ids]
        assert ids != [None], f"нет источника: {path}"
        assert all(i in refs for i in ids), f"неизвестный источник: {path}"


def test_official_and_literature_sources_have_links_and_missing_sources_are_explained():
    for rid, r in load_refs().items():
        assert r["kind"] in ("official", "literature", "data", "none")
        assert r.get("short") and r.get("title")
        if r["kind"] == "official":
            assert r["url"].startswith("http"), rid
        if r["kind"] == "literature":  # публичная ссылка либо имя предоставленного файла
            assert r.get("url", "").startswith("http") or r.get("file"), rid
        if r["kind"] == "none":
            assert r["note"], rid
            assert "url" not in r, rid


def test_value_types_do_not_claim_literature_without_a_literature_source():
    for path, node in _nodes(load_cfg()):
        kinds = {r["kind"] for r in node_refs(node)}
        if node["src"] in ("GOST", "REQUIREMENT"):
            assert "official" in kinds, path
        if node["src"] == "LIT":
            assert kinds & {"literature", "official"}, path
        if node["src"] == "TASK":
            assert "data" in kinds, path


def test_norms_match_official_documents():
    spec = load_cfg()["spec"]
    assert v(spec["sulfur_max_mgkg"]) == 10.0          # ТР ТС 013/2011, ГОСТ 32511-2013 (класс K5)
    assert v(spec["t95_max_c"]) == 360.0               # ГОСТ 32511-2013, ISO 3405
    assert v(spec["flash_min_c"]) == 55.0              # ГОСТ 32511-2013, ISO 2719
    assert v(spec["d15_range"]) == [820.0, 845.0]      # ГОСТ 32511-2013
    assert v(spec["cetane_min"]) == 51.0               # ГОСТ 32511-2013, ГОСТ 32508
    assert v(spec["winter"]["d15_range"]) == [800.0, 845.0]   # ГОСТ 32511-2013, таблица 3, классы 0 и 1
    assert v(spec["winter"]["cetane_min"]) == 49.0            # ГОСТ 32511-2013, таблица 3, классы 0 и 1


def test_no_value_rests_on_a_missing_source():
    refs = load_refs()
    for path, node in _nodes(load_cfg()):
        ids = node.get("ref")
        ids = ids if isinstance(ids, list) else [ids]
        assert all(refs[i]["kind"] != "none" for i in ids), f"значение без источника: {path}"
        assert node["src"] in ("GOST", "REQUIREMENT", "TASK", "DATA", "LIT"), path      # допущений без источника нет
