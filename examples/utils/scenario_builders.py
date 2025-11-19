"""Scenario builders for simple ALM stress tests."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def parallel_shift(amount: float) -> Callable[[np.ndarray], np.ndarray]:
    """Add a constant shift to the entire curve."""

    def shock(curve: np.ndarray) -> np.ndarray:
        return curve + amount

    return shock


def steepener(amount: float) -> Callable[[np.ndarray], np.ndarray]:
    """Increase long-term rates and decrease short-term ones."""

    def shock(curve: np.ndarray) -> np.ndarray:
        factors = np.linspace(-0.5, 1.0, curve.size)
        return curve + amount * factors

    return shock


def flattener(amount: float) -> Callable[[np.ndarray], np.ndarray]:
    """Flatten the curve by moving short end up and long end down."""

    def shock(curve: np.ndarray) -> np.ndarray:
        factors = np.linspace(1.0, -1.0, curve.size)
        return curve + amount * factors

    return shock


def ramp(amount: float) -> Callable[[np.ndarray], np.ndarray]:
    """Apply a linear upward ramp along maturities."""

    def shock(curve: np.ndarray) -> np.ndarray:
        factors = np.linspace(0.0, 1.0, curve.size)
        return curve + amount * factors

    return shock


DEFAULT_SCENARIOS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "base": lambda curve: curve,
    "parallel_up_100bps": parallel_shift(0.01),
    "parallel_down_100bps": parallel_shift(-0.01),
    "steepener_50bps": steepener(0.005),
    "flattener_50bps": flattener(0.005),
    "ramp_up_30bps": ramp(0.003),
}
