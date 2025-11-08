"""Tests for CIR bond pricing and term-structure utilities."""

import math

from cir.bonds import bond_price_mc
from cir.params import get_params_preset


def _price(params, T, n_paths, steps_per_year, seed, scheme):
    n_steps = max(1, int(T * steps_per_year))
    price, _ = bond_price_mc(
        params=params,
        T=T,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        scheme=scheme,
    )
    return price


def test_bond_price_monotonicity() -> None:
    params = get_params_preset("baseline")
    scheme = "milstein"
    price_short = _price(params, T=1.0, n_paths=4000, steps_per_year=128, seed=111, scheme=scheme)
    price_long = _price(params, T=5.0, n_paths=4000, steps_per_year=128, seed=222, scheme=scheme)
    assert price_long <= price_short


def test_zero_rates_non_negative_for_reasonable_params() -> None:
    params = get_params_preset("baseline")
    scheme = "em"
    maturities = [0.5, 1.0, 2.5, 5.0]
    zero_rates = []
    for idx, T in enumerate(maturities):
        price = _price(
            params,
            T=T,
            n_paths=5000,
            steps_per_year=128,
            seed=1000 + idx,
            scheme=scheme,
        )
        price = max(price, 1e-12)
        zero_rates.append(-math.log(price) / T)
    assert all(rate >= -1e-4 for rate in zero_rates)
