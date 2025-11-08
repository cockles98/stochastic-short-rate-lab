"""Time-stepping schemes for the CIR short-rate stochastic differential equation.

The Cox-Ingersoll-Ross (CIR) process

    dr_t = kappa (theta - r_t) dt + sigma sqrt(r_t) dW_t

is mean-reverting and strictly positive under the Feller condition imposed in
``cir.params``. Numerical discretizations still need extra care close to zero:
Euler–Maruyama (EM) is easy to implement but tends to bias the distribution when
rates flirt with the boundary, while the Milstein correction elevates the
strong order to 1.0 and generally stabilizes behavior near r_t = 0.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .params import CIRParams
from .rng import make_rng, normal_increments

Array = np.ndarray
Scheme = Literal["em", "milstein"]

__all__ = ["euler_maruyama", "milstein", "simulate_terminal", "simulate_terminal_from_increments"]


def _validate_inputs(T: float, n_steps: int, n_paths: int) -> float:
    if T <= 0:
        raise ValueError("Total horizon T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")
    return T / n_steps


def _setup(params: CIRParams, T: float, n_steps: int, n_paths: int) -> tuple[float, Array]:
    dt = _validate_inputs(T, n_steps, n_paths)
    rates = np.empty((n_paths, n_steps + 1), dtype=float)
    rates[:, 0] = params.r0
    return dt, rates


def euler_maruyama(
    params: CIRParams,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Array:
    """Simulate CIR paths with the Euler–Maruyama discretization.

    Examples
    --------
    >>> from cir.params import get_params_preset
    >>> from cir.rng import make_rng
    >>> rng = make_rng(0)
    >>> params = get_params_preset("baseline")
    >>> euler_maruyama(params, T=1.0, n_steps=4, n_paths=2, rng=rng).shape
    (2, 5)
    """

    dt, rates = _setup(params, T, n_steps, n_paths)
    dW = normal_increments(rng, n_paths=n_paths, n_steps=n_steps, dt=dt)

    for t in range(n_steps):
        r_t = rates[:, t]
        sqrt_rt = np.sqrt(np.maximum(r_t, 0.0))
        drift = params.kappa * (params.theta - r_t) * dt
        diffusion = params.sigma * sqrt_rt * dW[:, t]
        rates[:, t + 1] = r_t + drift + diffusion

    return np.maximum(rates, 0.0)


def milstein(
    params: CIRParams,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Array:
    """Simulate CIR paths with the Milstein discretization.

    The Milstein scheme augments Euler–Maruyama with a diffusion derivative
    term, significantly reducing bias for square-root processes.

    Examples
    --------
    >>> from cir.params import get_params_preset
    >>> from cir.rng import make_rng
    >>> rng = make_rng(0)
    >>> params = get_params_preset("baseline")
    >>> milstein(params, T=1.0, n_steps=4, n_paths=2, rng=rng).shape
    (2, 5)
    """

    dt, rates = _setup(params, T, n_steps, n_paths)
    dW = normal_increments(rng, n_paths=n_paths, n_steps=n_steps, dt=dt)
    sqrt_dt = np.sqrt(dt)
    xi = dW / sqrt_dt
    correction_scale = 0.25 * params.sigma**2 * dt

    for t in range(n_steps):
        r_t = rates[:, t]
        sqrt_rt = np.sqrt(np.maximum(r_t, 0.0))
        drift = params.kappa * (params.theta - r_t) * dt
        diffusion = params.sigma * sqrt_rt * dW[:, t]
        correction = correction_scale * (xi[:, t] ** 2 - 1.0)
        rates[:, t + 1] = r_t + drift + diffusion + correction

    return np.maximum(rates, 0.0)


def simulate_terminal_from_increments(
    scheme: Scheme,
    params: CIRParams,
    dt: float,
    dW: np.ndarray,
) -> np.ndarray:
    """Propagate only terminal rates given Brownian increments."""

    if scheme not in {"em", "milstein"}:
        raise ValueError("scheme must be 'em' or 'milstein'.")

    n_paths, n_steps = dW.shape
    r = np.full(n_paths, params.r0, dtype=float)
    sqrt_dt = np.sqrt(dt)

    for t in range(n_steps):
        sqrt_rt = np.sqrt(np.maximum(r, 0.0))
        drift = params.kappa * (params.theta - r) * dt
        diffusion = params.sigma * sqrt_rt * dW[:, t]
        update = drift + diffusion
        if scheme == "milstein":
            xi = dW[:, t] / sqrt_dt
            update += 0.25 * params.sigma**2 * (xi**2 - 1.0) * dt
        r = r + update

    return np.maximum(r, 0.0)


def simulate_terminal(
    scheme: Scheme,
    params: CIRParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate only the terminal short rate to avoid storing full paths."""

    if rng is not None and seed is not None:
        raise ValueError("Provide either rng or seed, not both.")
    dt = _validate_inputs(T, n_steps, n_paths)
    local_rng = rng if rng is not None else make_rng(seed)
    dW = normal_increments(local_rng, n_paths=n_paths, n_steps=n_steps, dt=dt)
    return simulate_terminal_from_increments(
        scheme=scheme,
        params=params,
        dt=dt,
        dW=dW,
    )
