"""Calibration utilities for fitting CIR parameters to market curves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Literal, Sequence, TypeVar

import numpy as np
from scipy.optimize import minimize

from .analytics import zero_coupon_price
from .params import CIRParams

try:  # pragma: no cover - optional dependency
    from data_loaders.curves import load_curve_components
except ImportError:  # pragma: no cover - optional dependency
    load_curve_components = None

__all__ = ["calibrate_zero_coupon_curve", "price_curve", "market_curve_from_file"]

ParamsT = TypeVar("ParamsT")


def price_curve(params: CIRParams, maturities: Sequence[float]) -> np.ndarray:
    """Return analytical zero-coupon prices for the given maturities."""

    mats = np.asarray(maturities, dtype=float)
    if mats.size == 0:
        raise ValueError("maturities must not be empty.")
    if np.any(mats < 0):
        raise ValueError("maturities must be non-negative.")
    return np.array([zero_coupon_price(params, float(m)) for m in mats])


@dataclass
class CalibrationResult(Generic[ParamsT]):
    params: ParamsT
    success: bool
    message: str
    fun: float
    nit: int | None
    raw_solution: np.ndarray | None = None


def _objective_function(
    x: np.ndarray,
    maturities: np.ndarray,
    market: np.ndarray,
    weights: np.ndarray,
    penalty: float,
) -> float:
    kappa, theta, sigma, r0 = x
    if np.any(x <= 0):
        return 1e9 + float(np.sum(np.clip(-x, 0, None) ** 2))

    penalty_loss = 0.0
    try:
        params = CIRParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)
    except ValueError:
        params = _sanitize_parameters(x)
        penalty_loss = penalty * 10.0

    try:
        model_prices = price_curve(params, maturities)
    except (OverflowError, ValueError):
        return 1e12
    residuals = (model_prices - market) * weights
    loss = float(np.sum(residuals**2)) + penalty_loss

    feller_gap = sigma**2 - 2 * kappa * theta
    if feller_gap >= 0:
        loss += penalty * (feller_gap + 1e-6) ** 2

    return loss


def _sanitize_parameters(x: np.ndarray) -> CIRParams:
    """Coerce optimizer output into a valid CIRParams, enforcing the Feller condition."""

    kappa, theta, sigma, r0 = [max(float(val), 1e-6) for val in x]
    if 2 * kappa * theta <= sigma**2:
        theta = sigma**2 / (2 * kappa) + 1e-6
    return CIRParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)


def calibrate_zero_coupon_curve(
    maturities: Iterable[float],
    market_prices: Iterable[float],
    initial: CIRParams,
    weights: Iterable[float] | None = None,
    penalty: float = 1e4,
    method: str = "L-BFGS-B",
) -> CalibrationResult:
    """Calibrate CIR parameters to a set of market zero-coupon prices."""

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
        (1e-6, None),
        (1e-6, None),
        (1e-6, None),
    )

    result = minimize(
        _objective_function,
        x0=x0,
        args=(mats, market, weights_arr, penalty),
        method=method,
        bounds=bounds,
    )

    try:
        opt_params = CIRParams(*result.x)
    except ValueError:
        opt_params = _sanitize_parameters(result.x)

    return CalibrationResult(
        params=opt_params,
        success=bool(result.success),
        message=result.message,
        fun=float(result.fun),
        nit=result.nit,
        raw_solution=result.x,
    )


def market_curve_from_file(
    curve_file: str | Path,
    curve_kind: Literal["prefixados", "ipca"] = "prefixados",
    column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return maturities (years), prices and yields from a curve CSV."""

    if load_curve_components is None:  # pragma: no cover
        raise ImportError("data_loaders.curves não está disponível.")
    components = load_curve_components(curve_file)
    if curve_kind == "prefixados":
        df = components.prefixados.dropna()
        rates = df["Taxa (%a.a.)"].astype(float) / 100.0
    elif curve_kind == "ipca":
        col = column or "ETTJ PREF"
        df = components.ettj.dropna(subset=[col])
        rates = df[col].astype(float) / 100.0
    else:
        raise ValueError("curve_kind deve ser 'prefixados' ou 'ipca'.")
    maturities = df["Vertices"].astype(float) / 252.0
    prices = np.exp(-rates.to_numpy() * maturities.to_numpy())
    return maturities.to_numpy(), prices, rates.to_numpy(), components.ref_date
