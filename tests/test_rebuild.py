"""Пересчёт производных рядов в памяти: воспроизводит кэш при исходных допущениях и реагирует на пороги приборов."""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from mas.common import CACHE_DIR, load_calib, load_cfg
from mas.state.derived import load_derived, rebuild_derived

pytestmark = pytest.mark.skipif(not (CACHE_DIR / "state_sulfur.parquet").exists(), reason="нужен кэш: python run.py all")


@pytest.fixture(scope="module")
def base():
    return load_derived()


def test_rebuild_with_default_settings_matches_cache(base):
    new = rebuild_derived(load_cfg(), load_calib())
    for key in base:
        pd.testing.assert_frame_equal(base[key], new[key], check_dtype=False, check_freq=False, obj=key)


def test_narrow_analyzer_range_flags_more_points_and_moves_estimate(base):
    cfg = copy.deepcopy(load_cfg())
    cfg["data_quality"]["pak_sulfur_valid"]["value"] = [0.05, 8.0]
    new = rebuild_derived(cfg, load_calib())
    assert (new["pak_sulfur"]["reason"] != "").sum() > (base["pak_sulfur"]["reason"] != "").sum()
    assert not new["state_sulfur"]["S_hat"].equals(base["state_sulfur"]["S_hat"])
    assert new["q21_sulfur"].equals(base["q21_sulfur"]) or (new["q21_sulfur"]["reason"] == base["q21_sulfur"]["reason"]).all()


def test_lims_range_change_rejects_more_records_and_moves_estimate(base):
    cfg = copy.deepcopy(load_cfg())
    cfg["data_quality"]["lims_ranges"]["value"]["Mg.Sulfur"] = [0, 12]
    new = rebuild_derived(cfg, load_calib())
    assert len(new["lims"]) < len(base["lims"])
    assert not new["state_sulfur"]["S_hat"].equals(base["state_sulfur"]["S_hat"])
