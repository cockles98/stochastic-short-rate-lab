"""Analytical formulas for the Cox-Ingersoll-Ross (CIR) model."""

from __future__ import annotations

import math

from cir.params import CIRParams


def zero_coupon_price(params: CIRParams, maturity: float) -> float:
    """Return the closed-form price of a zero-coupon bond under CIR."""

    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    if maturity == 0:
        return 1.0
    if params.sigma <= 0 or params.kappa <= 0:
        raise ValueError("kappa and sigma must be positive for analytical price.")

    try:
        gamma = math.sqrt(params.kappa ** 2 + 2 * params.sigma ** 2)
        numerator = 2 * gamma * math.exp((params.kappa + gamma) * maturity / 2)
        denominator = (params.kappa + gamma) * (math.exp(gamma * maturity) - 1) + 2 * gamma
        A = (numerator / denominator) ** (2 * params.kappa * params.theta / params.sigma ** 2)
        B = 2 * (math.exp(gamma * maturity) - 1) / denominator
        return A * math.exp(-B * params.r0)
    except OverflowError:
        return math.inf


def mean_short_rate(params: CIRParams, T: float) -> float:
    """Analytical expectation of r_T."""

    if T < 0:
        raise ValueError("T must be non-negative.")
    return params.theta + (params.r0 - params.theta) * math.exp(-params.kappa * T)


def variance_short_rate(params: CIRParams, T: float) -> float:
    """Analytical variance of r_T."""

    if T < 0:
        raise ValueError("T must be non-negative.")
    if params.kappa <= 0:
        raise ValueError("kappa must be positive for variance formula.")
    exp_term = math.exp(-params.kappa * T)
    term1 = (
        params.r0 * params.sigma ** 2 * exp_term * (1 - exp_term) / params.kappa
    )
    term2 = (
        params.theta * params.sigma ** 2 / (2 * params.kappa)
        * (1 - exp_term) ** 2
    )
    return term1 + term2
