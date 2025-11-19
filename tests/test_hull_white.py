import numpy as np
import pytest

from benchmarks.models.hull_white import (
    HullWhiteParams,
    get_hull_white_preset,
    hull_white_bond_price_mc,
    hull_white_mean_short_rate,
    hull_white_price_curve,
    hull_white_variance_short_rate,
    hull_white_zero_coupon_price,
    simulate_hull_white_paths,
)
from benchmarks.models.vasicek import (
    vasicek_mean_short_rate,
    vasicek_variance_short_rate,
    vasicek_zero_coupon_price,
)


def _zero_shift_params(name: str = "baseline") -> HullWhiteParams:
    return get_hull_white_preset(name, shift_times=[0.0, 30.0], shift_values=[0.0, 0.0])


def test_zero_shift_matches_vasicek_price_curve():
    hw_params = _zero_shift_params()
    mats = [0.5, 1.0, 5.0]
    hw_prices = hull_white_price_curve(hw_params, mats)
    vas_prices = np.array([vasicek_zero_coupon_price(hw_params.base_vasicek(), m) for m in mats])
    assert np.allclose(hw_prices, vas_prices)


def test_mean_and_variance_include_shift():
    shift_times = [0.0, 5.0, 10.0]
    shift_values = [0.01, 0.015, 0.02]
    hw_params = get_hull_white_preset("baseline", shift_times, shift_values)
    T = 2.0
    mean = hull_white_mean_short_rate(hw_params, T)
    var = hull_white_variance_short_rate(hw_params, T)
    base_params = hw_params.base_vasicek()
    base_mean = vasicek_mean_short_rate(base_params, T)
    base_var = vasicek_variance_short_rate(base_params, T)
    assert mean == pytest.approx(base_mean + hw_params.shift_at(T))
    assert var == pytest.approx(base_var)


def test_mc_price_matches_analytic_with_constant_shift():
    shift_times = [0.0, 10.0]
    shift_values = [0.01, 0.01]
    hw_params = get_hull_white_preset("baseline", shift_times, shift_values)
    analytic = hull_white_zero_coupon_price(hw_params, 3.0)
    price, stderr = hull_white_bond_price_mc(
        params=hw_params,
        T=3.0,
        n_paths=6000,
        n_steps=3 * 252,
        seed=321,
        scheme="exact",
    )
    assert price == pytest.approx(analytic, rel=0.02)
    assert stderr >= 0


def test_simulation_returns_shifted_rates():
    shift_times = [0.0, 2.0]
    shift_values = [0.02, 0.02]
    hw_params = get_hull_white_preset("baseline", shift_times, shift_values)
    grid, paths = simulate_hull_white_paths(
        scheme="em",
        params=hw_params,
        T=0.5,
        n_steps=10,
        n_paths=5,
        seed=0,
    )
    assert grid.shape[0] == paths.shape[1]
    assert np.allclose(paths[:, 0], hw_params.r0)
