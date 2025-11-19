import numpy as np
import pytest

from benchmarks.models.vasicek import (
    VasicekParams,
    calibrate_vasicek_curve,
    simulate_vasicek_paths,
    vasicek_bond_price_mc,
    vasicek_mean_short_rate,
    vasicek_price_curve,
    vasicek_zero_coupon_price,
)


def test_zero_coupon_price_is_between_zero_and_one():
    params = VasicekParams(kappa=0.9, theta=0.04, sigma=0.15, r0=0.03)
    price = vasicek_zero_coupon_price(params, maturity=2.5)
    assert 0 < price < 1


def test_exact_simulation_matches_analytic_mean_within_tolerance():
    params = VasicekParams(kappa=1.1, theta=0.05, sigma=0.12, r0=0.02)
    T = 1.0
    _, paths = simulate_vasicek_paths(
        scheme="exact",
        params=params,
        T=T,
        n_steps=252,
        n_paths=50_000,
        seed=123,
    )
    sample_mean = paths[:, -1].mean()
    analytic_mean = vasicek_mean_short_rate(params, T=T)
    assert sample_mean == pytest.approx(analytic_mean, rel=2e-2)


def test_calibration_recovers_synthetic_curve():
    true_params = VasicekParams(kappa=0.8, theta=0.03, sigma=0.10, r0=0.025)
    maturities = np.linspace(0.25, 5.0, 10)
    market = vasicek_price_curve(true_params, maturities)
    initial = VasicekParams(kappa=0.5, theta=0.02, sigma=0.07, r0=0.01)
    result = calibrate_vasicek_curve(maturities, market, initial)
    assert result.success
    assert result.fun < 1e-4
    fitted = vasicek_price_curve(result.params, maturities)
    assert np.allclose(fitted, market, atol=3e-4)


def test_mc_bond_price_is_close_to_analytic():
    params = VasicekParams(kappa=1.0, theta=0.04, sigma=0.15, r0=0.03)
    analytic = vasicek_zero_coupon_price(params, 2.0)
    price, stderr = vasicek_bond_price_mc(
        params=params,
        T=2.0,
        n_paths=5000,
        n_steps=2 * 252,
        seed=7,
        scheme="exact",
    )
    assert price == pytest.approx(analytic, rel=0.02)
    assert stderr >= 0
