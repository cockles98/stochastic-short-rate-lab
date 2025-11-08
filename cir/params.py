"""Parameter utilities for the Cox-Ingersoll-Ross (CIR) short-rate model.

The CIR stochastic differential equation

    dr_t = kappa * (theta - r_t) dt + sigma * sqrt(r_t) dW_t

admits strictly positive solutions only when the Feller condition holds,
which requires 2 * kappa * theta > sigma**2. This module centralizes the
parameter dataclass and preset configurations, ensuring every instance
validates the inequality before use in simulations or pricing tasks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CIRParams:
    """Typed container for CIR parameters with Feller validation."""

    kappa: float
    theta: float
    sigma: float
    r0: float

    def __post_init__(self) -> None:
        # Condition implies non-attainability of zero and well-behaved paths.
        if 2.0 * self.kappa * self.theta <= self.sigma**2:
            msg = (
                "Feller condition violated: require 2 * kappa * theta > sigma**2 "
                f"(received kappa={self.kappa}, theta={self.theta}, sigma={self.sigma})"
            )
            raise ValueError(msg)


_PRESETS: dict[str, CIRParams] = {
    "baseline": CIRParams(kappa=1.2, theta=0.05, sigma=0.20, r0=0.03),
    "slow-revert": CIRParams(kappa=0.5, theta=0.08, sigma=0.25, r0=0.04),
    "fast-revert": CIRParams(kappa=3.0, theta=0.02, sigma=0.10, r0=0.015),
}


def get_params_preset(name: str) -> CIRParams:
    """Return a predefined CIR parameter set that obeys the Feller condition."""

    try:
        return _PRESETS[name]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise ValueError(
            f"Unknown preset '{name}'. Valid options: {', '.join(sorted(_PRESETS))}"
        ) from exc
