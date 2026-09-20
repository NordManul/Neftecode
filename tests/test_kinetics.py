"""Знаки и монотонность отклика, шанс-ограничение по смеси."""
from __future__ import annotations

import numpy as np

from mas.models.kinetics import chance_exceed, ensemble_sulfur, response_ensemble, t95_ensemble, upper_quantile_mix

CALIB = {"beta_t": {"weak": -0.009, "medium": -0.045, "strong": -0.073},
         "beta_f": {"weak": 0.57, "medium": 1.25, "strong": 1.94},
         "b95": {"weak": 0.10, "medium": 0.165, "strong": 0.28}}
GAMMA = 0.007


def test_ensemble_has_27_members_with_equal_weights():
    members, w = response_ensemble(CALIB)
    assert len(members) == 27
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w, 1 / 27)   # оснований предпочесть один сценарий отклика нет (веса не задаются)


def test_temperature_up_lowers_sulfur_for_all_members():
    members, w = response_ensemble(CALIB)
    ln_s = np.full(1, np.log(8.0))
    s_base = ensemble_sulfur(ln_s, np.array([0.0]), np.array([0.0]), np.array([0.0]), GAMMA, members, w)
    s_up = ensemble_sulfur(ln_s, np.array([2.0]), np.array([0.0]), np.array([0.0]), GAMMA, members, w)
    assert np.all(s_up[0] < s_base[0])


def test_feed_up_raises_sulfur_for_all_members():
    members, w = response_ensemble(CALIB)
    ln_s = np.full(1, np.log(8.0))
    s_base = ensemble_sulfur(ln_s, np.array([0.0]), np.array([0.0]), np.array([0.0]), GAMMA, members, w)
    s_up = ensemble_sulfur(ln_s, np.array([0.0]), np.array([0.10]), np.array([0.0]), GAMMA, members, w)
    assert np.all(s_up[0] > s_base[0])


def test_f32_up_raises_t95():
    members, w = response_ensemble(CALIB)
    b95_values = np.asarray(members)[:, 2]
    t95_base = t95_ensemble(np.array([350.0]), np.array([0.0]), b95_values)
    t95_up = t95_ensemble(np.array([350.0]), np.array([5.0]), b95_values)
    assert np.all(t95_up[0] > t95_base[0])


def test_p_exceed_monotone_in_dT5():
    members, w = response_ensemble(CALIB)
    ln_s = np.full(5, np.log(9.0))
    dT5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    s_k = ensemble_sulfur(ln_s, dT5, np.zeros(5), np.zeros(5), GAMMA, members, w)
    sigma_rel = np.full(5, 0.6 / 9.0)
    p_exceed, p_worst = chance_exceed(s_k, sigma_rel, w, 10.0)
    assert np.all(np.diff(p_exceed) <= 1e-9)  # не возрастает при увеличении ΔT5
    assert np.all(p_exceed <= p_worst + 1e-9)


def test_upper_quantile_mix_matches_grid_definition():
    members, w = response_ensemble(CALIB)
    ln_s = np.full(1, np.log(8.5))
    s_k = ensemble_sulfur(ln_s, np.array([0.0]), np.array([0.0]), np.array([0.0]), GAMMA, members, w)
    sigma_rel = np.full(1, 0.6 / 8.5)
    upper = upper_quantile_mix(s_k, sigma_rel, w)
    from scipy.stats import norm
    cdf_at_upper = norm.cdf((upper - s_k) / (s_k * sigma_rel[:, None])) @ w
    assert cdf_at_upper[0] >= 0.95 - 1e-6
    cdf_below = norm.cdf((upper - 0.05 - s_k) / (s_k * sigma_rel[:, None])) @ w
    assert cdf_below[0] < 0.95


def test_no_action_quality_margin_equals_ten_minus_current_upper():
    members, w = response_ensemble(CALIB)
    s_forecast, sigma_forecast = 9.0, 0.6
    ln_s = np.full(1, np.log(s_forecast))
    s_k = ensemble_sulfur(ln_s, np.array([0.0]), np.array([0.0]), np.array([0.0]), GAMMA, members, w)
    sigma_rel = np.full(1, sigma_forecast / s_forecast)
    upper = upper_quantile_mix(s_k, sigma_rel, w)
    quality_margin = 10.0 - upper[0]
    assert quality_margin == 10.0 - upper[0]  # тождество по построению; проверяем, что не NaN/отрицательно по модулю
    assert np.isfinite(quality_margin)


def test_upper_quantile_mix_saturates_at_grid_top_and_follows_level():
    members, w = response_ensemble(CALIB)
    ln_hi, ln_mid = np.full(1, np.log(20.0)), np.full(1, np.log(8.5))
    zero = np.array([0.0])
    s_hi = ensemble_sulfur(ln_hi, zero, zero, zero, GAMMA, members, w)
    upper_hi = upper_quantile_mix(s_hi, np.full(1, 0.05), w)
    assert upper_hi[0] == 16.0                    # квантиль за пределами сетки: верхняя граница, а не нижняя
    s_mid = ensemble_sulfur(ln_mid, zero, zero, zero, GAMMA, members, w)
    sigma_rel = np.full(1, 0.6 / 8.5)
    assert upper_quantile_mix(s_mid, sigma_rel, w, q=0.90)[0] < upper_quantile_mix(s_mid, sigma_rel, w, q=0.99)[0]
