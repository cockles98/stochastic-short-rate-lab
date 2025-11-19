"""Vasicek short-rate model helpers for benchmarking against CIR.

The Vasicek (Ornstein-Uhlenbeck) dynamics

    dr_t = kappa (theta - r_t) dt + sigma dW_t

shares the same mean-reversion structure as CIR but features additive noise,
allowing negative rates. This module mirrors the public surface of the CIR
utilities (presets, path simulation, analytical pricing, calibration) so the
benchmark layer can plug alternate models without rewriting the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import minimize

from cir.bonds import discount_factors_from_paths
from cir.calibration import CalibrationResult
from cir.rng import make_rng, normal_increments

Array = np.ndarray
Scheme = Literal["em", "exact"]

__all__ = [
    "VasicekParams",
    "calibrate_vasicek_curve",
    "get_vasicek_preset",
    "simulate_vasicek_paths",
    "vasicek_bond_price_mc",
    "vasicek_mean_short_rate",
    "vasicek_price_curve",
    "vasicek_variance_short_rate",
    "vasicek_zero_coupon_price",
]


@dataclass(frozen=True)
class VasicekParams:
    """Parameter container enforcing positivity of kappa/sigma."""

    kappa: float
    theta: float
    sigma: float
    r0: float

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive for Vasicek dynamics.")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive for Vasicek dynamics.")


_PRESETS: dict[str, VasicekParams] = {
    "baseline": VasicekParams(kappa=1.2, theta=0.05, sigma=0.20, r0=0.03),
    "slow-revert": VasicekParams(kappa=0.5, theta=0.08, sigma=0.25, r0=0.04),
    "fast-revert": VasicekParams(kappa=3.0, theta=0.02, sigma=0.10, r0=0.015),
}


def get_vasicek_preset(name: str) -> VasicekParams:
    """Return one of the preset parameter configurations."""

    try:
        return _PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Unknown preset '{name}'. Available: {', '.join(sorted(_PRESETS))}."
        ) from exc


def _validate_inputs(T: float, n_steps: int, n_paths: int) -> float:
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    return T / n_steps


def _simulate_euler(
    params: VasicekParams,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Array:
    dt = _validate_inputs(T, n_steps, n_paths)
    rates = np.empty((n_paths, n_steps + 1), dtype=float)
    rates[:, 0] = params.r0
    dW = normal_increments(rng, n_paths=n_paths, n_steps=n_steps, dt=dt)

    for idx in range(n_steps):
        r_t = rates[:, idx]
        drift = params.kappa * (params.theta - r_t) * dt
        diffusion = params.sigma * dW[:, idx]
        rates[:, idx + 1] = r_t + drift + diffusion

    return rates


def _simulate_exact(
    params: VasicekParams,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Array:
    dt = _validate_inputs(T, n_steps, n_paths)
    rates = np.empty((n_paths, n_steps + 1), dtype=float)
    rates[:, 0] = params.r0
    exp_decay = math.exp(-params.kappa * dt)
    variance = (params.sigma ** 2) * (1 - math.exp(-2 * params.kappa * dt)) / (2 * params.kappa)
    std = math.sqrt(max(variance, 0.0))

    for idx in range(n_steps):
        mean = params.theta + (rates[:, idx] - params.theta) * exp_decay
        shocks = rng.normal(loc=0.0, scale=std, size=n_paths)
        rates[:, idx + 1] = mean + shocks

    return rates


def simulate_vasicek_paths(
    scheme: Scheme,
    params: VasicekParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Vasicek sample paths using Euler or the exact propagation."""

    if rng is not None and seed is not None:
        raise ValueError("Provide either rng or seed, not both.")
    steppers: dict[str, callable] = {
        "em": _simulate_euler,
        "exact": _simulate_exact,
    }
    try:
        stepper = steppers[scheme.lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError("scheme must be 'em' or 'exact'.") from exc

    local_rng = rng if rng is not None else make_rng(seed)
    rates = stepper(params=params, T=T, n_steps=n_steps, n_paths=n_paths, rng=local_rng)
    grid = np.linspace(0.0, T, n_steps + 1)
    return grid, rates


def vasicek_zero_coupon_price(params: VasicekParams, maturity: float) -> float:
    """Closed-form zero-coupon price for the Vasicek model."""

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if maturity == 0:
        return 1.0
    exp_term = math.exp(-params.kappa * maturity)
    B = (1 - exp_term) / params.kappa
    A = math.exp(
        (params.theta - (params.sigma ** 2) / (2 * params.kappa ** 2)) * (B - maturity)
        - (params.sigma ** 2) * (B ** 2) / (4 * params.kappa)
    )
    return A * math.exp(-B * params.r0)


def vasicek_price_curve(params: VasicekParams, maturities: Sequence[float]) -> np.ndarray:
    """Vectorized zero-coupon prices for the requested maturities."""

    mats = np.asarray(maturities, dtype=float)
    if mats.size == 0:
        raise ValueError("maturities must not be empty.")
    if np.any(mats < 0):
        raise ValueError("maturities must be non-negative.")
    return np.asarray([vasicek_zero_coupon_price(params, float(m)) for m in mats])


def vasicek_mean_short_rate(params: VasicekParams, T: float) -> float:
    """Analytical mean of r_T."""

    if T < 0:
        raise ValueError("T must be non-negative.")
    return params.theta + (params.r0 - params.theta) * math.exp(-params.kappa * T)


def vasicek_variance_short_rate(params: VasicekParams, T: float) -> float:
    """Analytical variance of r_T."""

    if T < 0:
        raise ValueError("T must be non-negative.")
    return (params.sigma ** 2) * (1 - math.exp(-2 * params.kappa * T)) / (2 * params.kappa)


def _sanitize_params(x: Sequence[float]) -> VasicekParams:
    kappa = max(float(x[0]), 1e-6)
    theta = float(x[1])
    sigma = max(float(x[2]), 1e-6)
    r0 = float(x[3])
    return VasicekParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)


def vasicek_bond_price_mc(
    params: VasicekParams,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    scheme: str = "exact",
    chunk_size: int | None = None,
) -> tuple[float, float]:
    """Monte Carlo zero-coupon price via Vasicek simulations."""

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided.")
    effective_chunk = (
        min(n_paths, 5000) if chunk_size is None else min(chunk_size, n_paths)
    )
    rng = make_rng(seed)
    remaining = n_paths
    grid: np.ndarray | None = None
    count = 0
    mean = 0.0
    m2 = 0.0

    while remaining > 0:
        batch = min(effective_chunk, remaining)
        grid, rates = simulate_vasicek_paths(
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


def calibrate_vasicek_curve(
    maturities: Iterable[float],
    market_prices: Iterable[float],
    initial: VasicekParams,
    weights: Iterable[float] | None = None,
    method: str = "L-BFGS-B",
) -> CalibrationResult[VasicekParams]:
    """Least-squares calibration of Vasicek parameters to a zero-coupon curve."""

    mats = np.asarray(list(maturities), dtype=float)
    market = np.asarray(list(market_prices), dtype=float)
    if mats.size == 0 or mats.size != market.size:
        raise ValueError("maturities and market_prices must have the same non-zero length.")
    if np.any(mats < 0) or np.any(market <= 0):
        raise ValueError("maturities must be >=0 and prices >0.")

    if weights is None:
        weights_arr = 1.0 / np.maximum(market, 1e-8)
    else:
        weights_arr = np.asarray(list(weights), dtype=float)
        if weights_arr.size != mats.size:
            raise ValueError("weights must match maturities length.")

    x0 = np.array([initial.kappa, initial.theta, initial.sigma, initial.r0], dtype=float)
    bounds = (
        (1e-6, None),
        (None, None),
        (1e-6, None),
        (None, None),
    )

    def objective(x: np.ndarray) -> float:
        params = _sanitize_params(x)
        model_prices = vasicek_price_curve(params, mats)
        residuals = (model_prices - market) * weights_arr
        return float(np.sum(residuals**2))

    result = minimize(
        objective,
        x0=x0,
        method=method,
        bounds=bounds,
    )
    opt_params = _sanitize_params(result.x)
    return CalibrationResult(
        params=opt_params,
        success=bool(result.success),
        message=result.message,
        fun=float(result.fun),
        nit=result.nit,
        raw_solution=result.x,
    )
