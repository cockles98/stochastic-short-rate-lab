"""Random number helpers for CIR simulations."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Return a reproducible NumPy Generator using the PCG64 bit generator."""

    return np.random.Generator(np.random.PCG64(seed))


def normal_increments(
    rng: np.random.Generator, n_paths: int, n_steps: int, dt: float
) -> np.ndarray:
    """Draw Brownian increments scaled to variance ``dt`` for CIR schemes."""

    if dt <= 0:
        raise ValueError("dt must be positive for diffusion increments.")
    std = np.sqrt(dt)
    return rng.normal(loc=0.0, scale=std, size=(n_paths, n_steps))
