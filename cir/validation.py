"""Validation helpers comparing Monte Carlo outputs to CIR analytical formulas."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .analytics import mean_short_rate, variance_short_rate, zero_coupon_price
from .bonds import bond_price_mc
from .params import CIRParams
from .simulate import simulate_paths

__all__ = [
    "compare_zero_coupon_prices",
    "compare_moments",
    "zero_coupon_error_by_steps",
]


def _ensure_iterable(maturities: Iterable[float]) -> list[float]:
    values = [float(m) for m in maturities]
    if not values:
        raise ValueError("maturities must contain at least one value.")
    return values


def compare_zero_coupon_prices(
    params: CIRParams,
    maturities: Sequence[float],
    n_paths: int,
    steps_per_year: int,
    seed: int | None,
    scheme: str = "milstein",
) -> pd.DataFrame:
    """Return Monte Carlo vs analytical zero-coupon prices for each maturity."""

    mats = _ensure_iterable(maturities)
    rows = []
    base_seed = seed or 0

    for idx, maturity in enumerate(mats):
        n_steps = max(1, int(math.ceil(maturity * steps_per_year)))
        price_mc, stderr = bond_price_mc(
            params=params,
            T=maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=base_seed + idx,
            scheme=scheme,
        )
        price_ana = zero_coupon_price(params, maturity)
        abs_err = abs(price_mc - price_ana)
        rel_err = abs_err / price_ana if price_ana not in (0.0, math.inf) else math.nan
        rows.append(
            {
                "T": maturity,
                "mc_price": price_mc,
                "stderr": stderr,
                "analytic_price": price_ana,
                "abs_error": abs_err,
                "rel_error": rel_err,
                "n_steps": n_steps,
            }
        )

    return pd.DataFrame(rows)


def compare_moments(
    params: CIRParams,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    scheme: str = "milstein",
) -> dict[str, float]:
    """Compare Monte Carlo mean/variance against analytical moments at time T."""

    if T < 0:
        raise ValueError("T must be non-negative.")
    _, paths = simulate_paths(
        scheme=scheme,
        params=params,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    terminals = paths[:, -1]
    mean_mc = float(np.mean(terminals))
    var_mc = float(np.var(terminals, ddof=1))

    mean_ana = mean_short_rate(params, T)
    var_ana = variance_short_rate(params, T)

    return {
        "mean_mc": mean_mc,
        "mean_analytic": mean_ana,
        "mean_abs_error": abs(mean_mc - mean_ana),
        "var_mc": var_mc,
        "var_analytic": var_ana,
        "var_abs_error": abs(var_mc - var_ana),
    }


def zero_coupon_error_by_steps(
    params: CIRParams,
    maturity: float,
    n_paths: int,
    steps_list: Sequence[int],
    seed: int | None,
    scheme: str = "milstein",
) -> pd.DataFrame:
    """Return absolute price error as a function of the number of time steps."""

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if not steps_list:
        raise ValueError("steps_list must contain at least one element.")

    price_ana = zero_coupon_price(params, maturity)
    rows = []

    for idx, n_steps in enumerate(sorted(set(int(s) for s in steps_list))):
        if n_steps <= 0:
            raise ValueError("entries in steps_list must be positive integers.")
        price_mc, _ = bond_price_mc(
            params=params,
            T=maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=None if seed is None else seed + idx,
            scheme=scheme,
        )
        abs_err = abs(price_mc - price_ana)
        rows.append(
            {
                "n_steps": n_steps,
                "dt": maturity / n_steps,
                "abs_error": abs_err,
                "rel_error": abs_err / price_ana if price_ana else math.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)
