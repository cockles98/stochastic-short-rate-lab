"""Shared helpers for swap and swaption valuation examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SwapSchedule:
    """Simple tenor representation for fixed -> floating swaps."""

    payment_times: np.ndarray  # shape (N,)
    accruals: np.ndarray  # shape (N,)

    @classmethod
    def from_tenor(cls, exercise: float, tenor: float, freq: int) -> "SwapSchedule":
        """Build a schedule starting right after exercise."""

        if freq <= 0:
            raise ValueError("freq must be positive.")
        accrual = 1.0 / freq
        n_payments = int(round(tenor * freq))
        times = exercise + np.arange(1, n_payments + 1, dtype=float) * accrual
        accruals = np.full_like(times, fill_value=accrual)
        return cls(payment_times=times, accruals=accruals)


def discount_path(rates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute discount factors for a single path at specific times."""

    grid = np.linspace(0.0, times[-1], rates.shape[0])
    discounts = []
    for T in times:
        mask = grid <= T
        integral = np.trapz(rates[mask], grid[mask])
        discounts.append(np.exp(-integral))
    return np.array(discounts)


def pv_fixed_leg(discounts: np.ndarray, accruals: np.ndarray, strike: float) -> float:
    """Present value of the fixed leg given discount factors."""

    return float(np.sum(discounts * accruals * strike))


def pv_float_leg(discounts: np.ndarray) -> float:
    """Assuming par swap with continuous compounding."""

    # In the single-curve setup, PV float = 1 - D(T_N)
    return float(1.0 - discounts[-1])


def swaption_payoff(discounts: np.ndarray, schedule: SwapSchedule, strike: float, kind: str) -> float:
    """Return the payoff (not discounted to t=0) for payer/receiver swaptions."""

    pv_fix = pv_fixed_leg(discounts, schedule.accruals, strike)
    pv_float = pv_float_leg(discounts)
    if kind == "payer":
        return max(pv_float - pv_fix, 0.0)
    elif kind == "receiver":
        return max(pv_fix - pv_float, 0.0)
    raise ValueError("kind must be 'payer' or 'receiver'.")
