"""Фильтр Калмана: сходимость, согласованность, реакция на брак."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mas.models.kalman import kalman_sulfur

MU, PHI_S, Q_S = 8.5, 0.995, 0.006


def _empty_lims() -> pd.DataFrame:
    return pd.DataFrame({"point": pd.array([], dtype=object), "parameter": pd.array([], dtype=object),
                          "time": pd.array([], dtype="datetime64[ns]"), "value": pd.array([], dtype=float),
                          "available_at": pd.array([], dtype="datetime64[ns]")})


def _params(**overrides) -> dict:
    base = {"mu": MU, "phi_s": PHI_S, "q_s": Q_S, "q_b": 0.001736, "sigma_lims": 1.0,
            "pak": {"phi_f": 0.5, "q_f": 0.1, "r": 0.5, "delay": 0}}
    base.update(overrides)
    return base


def _pak_df(idx, raw, reason=""):
    return pd.DataFrame({"raw": raw, "reason": reason, "run_len": 1}, index=idx)


@pytest.fixture()
def synthetic_series():
    rng = np.random.default_rng(0)
    n = 3000
    idx = pd.date_range("2023-01-01", periods=n, freq="10min")
    true_s = np.empty(n)
    true_s[0] = MU
    for t in range(1, n):
        true_s[t] = MU + PHI_S * (true_s[t - 1] - MU) + rng.normal(0, np.sqrt(Q_S))
    raw = true_s + rng.normal(0, 1.0, n)
    return idx, true_s, raw


def test_converges_to_true_level(synthetic_series):
    idx, true_s, raw = synthetic_series
    running = pd.Series(True, index=idx)
    state, priors = kalman_sulfur({"pak": _pak_df(idx, raw)}, _empty_lims(), running, _params())
    err = state["S_hat"].to_numpy()[500:] - true_s[500:]
    assert np.mean(np.abs(err)) < 1.0


def test_median_nis_in_expected_range(synthetic_series):
    idx, true_s, raw = synthetic_series
    running = pd.Series(True, index=idx)
    lims_times = idx[::200]
    lims = pd.DataFrame({"point": "HT2", "parameter": "Mg.Sulfur", "time": lims_times,
                          "value": true_s[::200] + np.random.default_rng(1).normal(0, 1.0, len(lims_times)),
                          "available_at": lims_times})
    state, priors = kalman_sulfur({"pak": _pak_df(idx, raw)}, lims, running, _params(sigma_lims=1.0))
    assert len(priors) > 5
    nis = (priors["S_prior"].to_numpy() - lims.sort_values("available_at")["value"].to_numpy()[:len(priors)]) ** 2 / \
        (priors["sigma_prior"].to_numpy() ** 2 + 1.0 ** 2)
    assert 0.2 <= np.median(nis) <= 1.2  # мягкий диапазон для короткой синтетики


def test_sigma_grows_without_updates_from_converged_state(synthetic_series):
    idx, true_s, raw = synthetic_series
    running = pd.Series(True, index=idx)
    # точный анализатор (малый r) - сходится к малой sigma с обновлениями, ниже стационарного
    # потолка без обновлений, поэтому после заморозки sigma растёт (а не падает к потолку сверху).
    reason = pd.Series([""] * 800 + ["заморозка"] * (len(idx) - 800), index=idx)
    state, _ = kalman_sulfur({"pak": _pak_df(idx, raw, reason)}, _empty_lims(), running, _params(pak={
        "phi_f": 0.5, "q_f": 0.1, "r": 0.05, "delay": 0}))
    seg = state["S_sigma"].iloc[799:900]
    assert seg.iloc[-1] > seg.iloc[0]
    assert (seg.diff().dropna() >= -1e-9).all()


def test_outlier_ignored(synthetic_series):
    idx, true_s, raw = synthetic_series
    running = pd.Series(True, index=idx)
    raw2 = raw.copy()
    raw2[500] = raw2[500] + 100.0  # выброс > 5 сигма
    state, _ = kalman_sulfur({"pak": _pak_df(idx, raw2)}, _empty_lims(), running, _params())
    state_clean, _ = kalman_sulfur({"pak": _pak_df(idx, raw)}, _empty_lims(), running, _params())
    assert abs(state["S_hat"].iloc[501] - state_clean["S_hat"].iloc[501]) < 0.5


def test_sigma_after_startup_reset(synthetic_series):
    idx, true_s, raw = synthetic_series
    running = pd.Series([False] * 100 + [True] * (len(idx) - 100), index=idx)
    reason = pd.Series([""] * len(idx), index=idx)
    reason.iloc[100] = "заморозка"  # анализатор ещё не даёт свежего чтения точно в момент пуска
    state, _ = kalman_sulfur({"pak": _pak_df(idx, raw, reason)}, _empty_lims(), running, _params())
    assert state["S_sigma"].iloc[100] >= 2.9


def test_forecast_of_single_component_is_closed_form_ou():
    from mas.models.kalman import forecast_sulfur
    params = {"mu": MU, "phi_s": PHI_S, "q_s": Q_S}
    mean, sd = forecast_sulfur(10.0, 0.0, 0.25, 0.0, 0.0, params, 24)
    ph = PHI_S ** 24
    assert mean == pytest.approx(MU + ph * (10.0 - MU))
    var_stationary = Q_S / (1 - PHI_S ** 2)
    assert sd == pytest.approx(np.sqrt(ph ** 2 * 0.25 + var_stationary * (1 - ph ** 2)))


def test_fast_component_decays_faster_than_slow_in_forecast():
    from mas.models.kalman import forecast_sulfur
    params = {"mu": MU, "phi_s": 0.995, "q_s": Q_S, "phi_fast": 0.9, "q_fast": 0.3}
    slow_only, _ = forecast_sulfur(MU + 1.0, 0.0, 0.1, 0.1, 0.0, params, 24)
    fast_only, _ = forecast_sulfur(MU, 1.0, 0.1, 0.1, 0.0, params, 24)
    assert slow_only - MU > 10 * (fast_only - MU) > 0


def test_two_component_state_is_consistent(synthetic_series):
    idx, _, raw = synthetic_series
    params = _params(phi_fast=0.9, q_fast=0.3)
    state, _ = kalman_sulfur({"pak": _pak_df(idx, raw)}, _empty_lims(), pd.Series(True, index=idx), params)
    assert np.allclose(state["S_hat"], state["S_slow"] + state["S_fast"])
    var_total = state["P_ss"] + state["P_ff"] + 2 * state["P_sf"]
    assert np.allclose(state["S_sigma"] ** 2, var_total)
    assert (state["S_sigma"] > 0).all() and state["S_fast"].abs().max() > 0


def _lims_at(times, values):
    return pd.DataFrame({"point": "HT2", "parameter": "Mg.Sulfur", "time": times, "value": values, "available_at": times})


def test_lab_result_does_not_change_state_before_its_step(synthetic_series):
    idx, true_s, raw = synthetic_series
    running = pd.Series(True, index=idx)
    t_lab = idx[1000]
    base, _ = kalman_sulfur({"pak": _pak_df(idx, raw)}, _lims_at(pd.DatetimeIndex([t_lab]), [MU]), running, _params())
    other, _ = kalman_sulfur({"pak": _pak_df(idx, raw)}, _lims_at(pd.DatetimeIndex([t_lab]), [MU + 2.0]), running, _params())
    assert np.allclose(base["S_hat"].iloc[:1000], other["S_hat"].iloc[:1000])          # до шага анализа состояние то же
    assert other["S_hat"].iloc[1000] > base["S_hat"].iloc[1000]                          # с шага анализа - учитывает его
    assert other["S_hat"].iloc[1050] > base["S_hat"].iloc[1050]


def test_lab_outlier_is_ignored(synthetic_series):
    idx, true_s, raw = synthetic_series
    running = pd.Series(True, index=idx)
    t_lab = idx[1000]
    clean, _ = kalman_sulfur({"pak": _pak_df(idx, raw)}, _empty_lims(), running, _params())
    outlier, _ = kalman_sulfur({"pak": _pak_df(idx, raw)}, _lims_at(pd.DatetimeIndex([t_lab]), [true_s[1000] + 80.0]), running, _params())
    assert np.allclose(clean["S_hat"], outlier["S_hat"])  # результат, далёкий от прогноза фильтра, не принимается
