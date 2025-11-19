"""Hull-White (shifted Vasicek) helpers for benchmarking.

We model the short rate as

    r(t) = x(t) + s(t)

where ``x`` follows a Vasicek process with parameters ``(kappa, theta, sigma)``
while ``s(t)`` is a deterministic shift specified by a time/value schedule.
When the shift is identically zero the model collapses to Vasicek. This
representation keeps simulation simple and allows us to reuse the analytical
formulas from the OU component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

from cir.bonds import discount_factors_from_paths
from cir.rng import make_rng

from .vasicek import (
    VasicekParams,
    get_vasicek_preset,
    simulate_vasicek_paths,
    vasicek_mean_short_rate,
    vasicek_variance_short_rate,
    vasicek_zero_coupon_price,
)

Array = np.ndarray
Scheme = Literal["em", "exact"]

__all__ = [
    "HullWhiteParams",
    "get_hull_white_preset",
    "simulate_hull_white_paths",
    "hull_white_zero_coupon_price",
    "hull_white_price_curve",
    "hull_white_mean_short_rate",
    "hull_white_variance_short_rate",
    "hull_white_bond_price_mc",
]


def _prepare_schedule(times: Sequence[float], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(times, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    if t_arr.ndim != 1 or v_arr.ndim != 1:
        raise ValueError("shift schedule arrays must be 1-D.")
    if t_arr.size != v_arr.size:
        raise ValueError("shift_times and shift_values must have the same length.")
    if t_arr.size < 2:
        raise ValueError("Provide at least two (time, value) pairs for the shift schedule.")
    order = np.argsort(t_arr)
    t_sorted = t_arr[order]
    v_sorted = v_arr[order]
    if t_sorted[0] != 0.0:
        raise ValueError("Shift schedule must start at t=0.")
    return t_sorted, v_sorted


@dataclass(frozen=True)
class HullWhiteParams:
    """Shifted Vasicek parameters with deterministic schedule."""

    kappa: float
    theta: float
    sigma: float
    r0: float
    shift_times: Sequence[float]
    shift_values: Sequence[float]

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive for Hull-White.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive for Hull-White.")
        times, values = _prepare_schedule(self.shift_times, self.shift_values)
        object.__setattr__(self, "shift_times", times)
        object.__setattr__(self, "shift_values", values)

    def shift_at(self, t: Array | float) -> Array | float:
        """Deterministic shift evaluated at t."""

        return np.interp(t, self.shift_times, self.shift_values)

    def base_vasicek(self) -> VasicekParams:
        """Return the Vasicek component parameters."""

        shift0 = float(self.shift_at(0.0))
        return VasicekParams(
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            r0=self.r0 - shift0,
        )

    def integrate_shift(self, T: float) -> float:
        """Numerically integrate the deterministic shift between 0 and T."""

        if T <= 0:
            return 0.0
        key_points = self.shift_times[(self.shift_times > 0.0) & (self.shift_times < T)]
        dense_points = np.linspace(0.0, T, max(2, min(2000, int(50 * T) + 1)))
        grid = np.unique(np.concatenate(([0.0, T], key_points, dense_points)))
        values = np.interp(grid, self.shift_times, self.shift_values)
        return float(np.trapz(values, grid))


def get_hull_white_preset(
    name: str,
    shift_times: Sequence[float] | None = None,
    shift_values: Sequence[float] | None = None,
) -> HullWhiteParams:
    """Seed Hull-White parameters reusing the Vasicek presets."""

    base = get_vasicek_preset(name)
    if shift_times is None or shift_values is None:
        shift_times = [0.0, 30.0]
        shift_values = [0.0, 0.0]
    return HullWhiteParams(
        kappa=base.kappa,
        theta=base.theta,
        sigma=base.sigma,
        r0=base.r0,
        shift_times=shift_times,
        shift_values=shift_values,
    )


def simulate_hull_white_paths(
    scheme: Scheme,
    params: HullWhiteParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Hull-White paths by shifting Vasicek trajectories."""

    if rng is not None and seed is not None:
        raise ValueError("Provide either rng or seed, not both.")
    base_params = params.base_vasicek()
    grid, base_paths = simulate_vasicek_paths(
        scheme=scheme,
        params=base_params,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        rng=rng,
    )
    shift_grid = params.shift_at(grid)
    rates = base_paths + shift_grid[np.newaxis, :]
    return grid, rates


def hull_white_zero_coupon_price(params: HullWhiteParams, maturity: float) -> float:
    """Closed-form price via deterministic shift decomposition."""

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if maturity == 0:
        return 1.0
    base = params.base_vasicek()
    base_price = vasicek_zero_coupon_price(base, maturity)
    shift_accum = params.integrate_shift(maturity)
    return math.exp(-shift_accum) * base_price


def hull_white_price_curve(params: HullWhiteParams, maturities: Iterable[float]) -> np.ndarray:
    mats = np.asarray(list(maturities), dtype=float)
    if mats.size == 0:
        raise ValueError("maturities must not be empty.")
    if np.any(mats < 0):
        raise ValueError("maturities must be non-negative.")
    return np.asarray([hull_white_zero_coupon_price(params, float(m)) for m in mats])


def hull_white_mean_short_rate(params: HullWhiteParams, T: float) -> float:
    if T < 0:
        raise ValueError("T must be non-negative.")
    base = params.base_vasicek()
    return vasicek_mean_short_rate(base, T) + float(params.shift_at(T))


def hull_white_variance_short_rate(params: HullWhiteParams, T: float) -> float:
    if T < 0:
        raise ValueError("T must be non-negative.")
    base = params.base_vasicek()
    return vasicek_variance_short_rate(base, T)


def hull_white_bond_price_mc(
    params: HullWhiteParams,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    scheme: Scheme = "exact",
    chunk_size: int | None = None,
) -> tuple[float, float]:
    """Monte Carlo zero-coupon price for Hull-White."""

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided.")
    effective_chunk = (
        min(n_paths, 5000) if chunk_size is None else min(chunk_size, n_paths)
    )
    rng = make_rng(seed)
    remaining = n_paths
    count = 0
    mean = 0.0
    m2 = 0.0

    while remaining > 0:
        batch = min(effective_chunk, remaining)
        grid, rates = simulate_hull_white_paths(
            scheme=scheme,
            params=params,
            T=T,
            n_steps=n_steps,
            n_paths=batch,
            rng=rng,
        )
        dfs = discount_factors_from_paths(grid, rates)
        batch_mean = float(dfs.mean())
        batch_var = float(dfs.var(ddof=0))
        batch_count = dfs.size
        delta = batch_mean - mean
        total = count + batch_count
        mean += delta * batch_count / total
        m2 += batch_var * batch_count + delta**2 * count * batch_count / total
        count = total
        remaining -= batch

    price = float(mean)
    if count > 1:
        variance = m2 / (count - 1)
        std_err = float(math.sqrt(variance / count))
    else:
        std_err = 0.0
    return price, std_err
