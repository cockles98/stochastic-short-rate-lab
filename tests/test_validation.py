import pandas as pd

from cir.params import get_params_preset
from cir.validation import (
    compare_moments,
    compare_zero_coupon_prices,
    zero_coupon_error_by_steps,
)


def test_compare_zero_coupon_prices_columns():
    params = get_params_preset("baseline")
    df = compare_zero_coupon_prices(params, maturities=[1.0, 2.0], n_paths=2000, steps_per_year=64, seed=123)
    assert list(df.columns) == [
        "T",
        "mc_price",
        "stderr",
        "analytic_price",
        "abs_error",
        "rel_error",
        "n_steps",
    ]
    assert len(df) == 2
    assert (df["abs_error"] >= 0).all()


def test_compare_moments_returns_small_errors():
    params = get_params_preset("baseline")
    result = compare_moments(params, T=1.0, n_paths=5000, n_steps=252, seed=42)
    assert result["mean_abs_error"] < 0.01
    assert result["var_abs_error"] >= 0


def test_zero_coupon_error_by_steps_sorted():
    params = get_params_preset("baseline")
    df = zero_coupon_error_by_steps(
        params=params,
        maturity=1.5,
        n_paths=3000,
        steps_list=[50, 100, 200],
        seed=99,
    )
    assert list(df.columns) == ["n_steps", "dt", "abs_error", "rel_error"]
    assert df["dt"].is_monotonic_increasing
