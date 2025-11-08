"""Bond pricing utilities under the CIR short-rate model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .params import CIRParams, get_params_preset
from .plots import plot_yield_curve
from .rng import make_rng
from .simulate import simulate_paths

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

__all__ = [
    "discount_factors_from_paths",
    "bond_price_mc",
    "term_structure",
]


def discount_factors_from_paths(t: Sequence[float], R: np.ndarray) -> np.ndarray:
    """Compute Monte Carlo discount factors via trapezoidal integration."""

    t = np.asarray(t, dtype=float)
    rates = np.asarray(R, dtype=float)
    if rates.ndim != 2:
        raise ValueError("R must be a 2-D array with shape (n_paths, n_steps+1).")
    if t.ndim != 1 or t.size != rates.shape[1]:
        raise ValueError("t must match the time dimension of R.")

    integrals = np.trapz(rates, t, axis=1)
    return np.exp(-integrals)


def bond_price_mc(
    params: CIRParams,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int | None,
    scheme: str = "em",
    chunk_size: int | None = None,
) -> tuple[float, float]:
    """Monte Carlo zero-coupon bond price and standard error."""

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided.")

    effective_chunk = (
        min(n_paths, 5000) if chunk_size is None else min(chunk_size, n_paths)
    )
    rng = make_rng(seed)
    time_grid = np.linspace(0.0, T, n_steps + 1)

    count = 0
    mean = 0.0
    m2 = 0.0

    remaining = n_paths
    while remaining > 0:
        batch = min(effective_chunk, remaining)
        _, rates = simulate_paths(
            scheme=scheme,
            params=params,
            T=T,
            n_steps=n_steps,
            n_paths=batch,
            rng=rng,
        )
        dfs = discount_factors_from_paths(time_grid, rates)
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


def _infer_preset_name(params: CIRParams) -> str:
    for name in ("baseline", "slow-revert", "fast-revert"):
        preset = get_params_preset(name)
        if (
            math.isclose(params.kappa, preset.kappa)
            and math.isclose(params.theta, preset.theta)
            and math.isclose(params.sigma, preset.sigma)
            and math.isclose(params.r0, preset.r0)
        ):
            return name
    return "custom"


def term_structure(
    params: CIRParams,
    maturities: Iterable[float],
    n_paths: int,
    steps_per_year: int,
    seed: int | None,
    scheme: str = "em",
) -> tuple[Path, Path]:
    """Compute Monte Carlo term structure and save CSV + figure."""

    mats = np.asarray(list(maturities), dtype=float)
    if mats.size == 0:
        raise ValueError("maturities must contain at least one element.")
    if np.any(mats <= 0):
        raise ValueError("maturities must be positive.")
    if steps_per_year <= 0:
        raise ValueError("steps_per_year must be positive.")

    results = []
    base_seed = seed

    for idx, T in enumerate(mats):
        n_steps = max(1, int(math.ceil(T * steps_per_year)))
        scenario_seed = None if base_seed is None else base_seed + idx
        price, stderr = bond_price_mc(
            params=params,
            T=T,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=scenario_seed,
            scheme=scheme,
        )
        safe_price = max(price, 1e-12)
        zero_rate = 0.0 if T == 0 else -math.log(safe_price) / T
        results.append(
            {
                "T": T,
                "price": price,
                "stderr": stderr,
                "zero_rate": zero_rate,
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    preset_name = _infer_preset_name(params)
    csv_path = DATA_DIR / f"term_structure_{scheme.lower()}_{preset_name}.csv"
    df.to_csv(csv_path, index=False)

    fig_path = plot_yield_curve(
        maturities=df["T"].to_numpy(),
        prices=df["price"].to_numpy(),
        yields=df["zero_rate"].to_numpy(),
        path_png=f"term_structure_{scheme.lower()}_{preset_name}.png",
    )

    return csv_path, fig_path
