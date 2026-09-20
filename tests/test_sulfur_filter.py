"""Настройка фильтра серы: параметры, целевая функция, локальная оптимизация на синтетических данных."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mas.calibrate import sulfur_filter as F
from mas.models.kalman import kalman_sulfur


def _synthetic(n: int = 3000, seed: int = 0) -> F.Dataset:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="10min")
    s = np.empty(n)
    s[0] = 0.0
    for i in range(1, n):
        s[i] = 0.99 * s[i - 1] + rng.normal(0, 0.15)
    truth = 8.0 + s
    pak = pd.DataFrame({"raw": truth + rng.normal(0, 1.5, n), "reason": ""}, index=idx)
    q21 = pd.DataFrame({"raw": truth + rng.normal(0, 0.3, n), "reason": ""}, index=idx)
    times = idx[60::36]
    lims = pd.DataFrame({"point": "HT2", "parameter": "Mg.Sulfur", "time": times, "available_at": times,
                         "value": truth[60::36] + rng.normal(0, 1.0, len(times))})
    return F.make_dataset(pak, q21, lims, pd.Series(True, index=idx), idx[-1], mu=8.0)


@pytest.fixture()
def dataset():
    data = _synthetic()
    F.set_dataset(data)
    yield data
    F.set_dataset(None)


def test_start_point_is_within_bounds_and_maps_to_filter_parameters():
    z, shift = F.load_start()
    assert all(lo <= x <= hi for x, (lo, hi) in zip(z, F.BOUNDS)) and F.SHIFT_BOUNDS[0] <= shift <= F.SHIFT_BOUNDS[1]
    params = F.to_params(z, shift, mu=8.0)
    assert params["pak"]["shift"] == shift and params["q21"]["shift"] == 0
    assert 0 < params["phi_s"] < 1 and params["sigma_lims"] > 0 and isinstance(params["q21"]["delay"], int)


def test_objective_is_finite_for_start_and_boundary_parameters(dataset):
    z, shift = F.load_start()
    good = np.array(z)
    bad = good.copy()
    bad[F.NAMES.index("log_sigma_lims")] = F.BOUNDS[10][0]   # σ_ЛИМС ≈ 0.6 вместо настоящего около 1.0-1.6
    assert np.isfinite(F.objective(good, shift)) and np.isfinite(F.objective(bad, shift))


def test_local_fit_does_not_worsen_objective(dataset):
    params, diag = F.fit_sulfur_filter(max_iter=2, workers=1)
    assert diag["objective"] <= diag["objective_start"] + 1e-12
    assert diag["n_targets"] > 30 and set(params) >= {"mu", "phi_s", "q_s", "q_b", "sigma_lims", "pak", "q21"}


def test_pak_shift_moves_measurements_later_in_time():
    idx = pd.date_range("2024-01-01", periods=50, freq="10min")
    raw = np.arange(50, dtype=float)
    pak = pd.DataFrame({"raw": raw, "reason": ""}, index=idx)
    params = F.to_params(F.load_start()[0], 5, mu=8.0)
    params["pak"]["shift"] = 5
    from mas.models.kalman import _delayed
    shifted = _delayed(raw, 5)
    assert np.isnan(shifted[:5]).all() and shifted[5] == raw[0] and shifted[-1] == raw[-6]
    state, _ = kalman_sulfur({"pak": pak, "q21": pak.copy()}, pd.DataFrame(columns=["point", "parameter", "available_at", "value"]).astype(
        {"available_at": "datetime64[ns]"}), pd.Series(True, index=idx), params)
    assert len(state) == 50
