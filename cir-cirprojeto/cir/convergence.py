"""Strong convergence diagnostics for CIR discretization schemes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .params import CIRParams
from .plots import plot_loglog_convergence
from .rng import make_rng, normal_increments
from .sde import simulate_terminal_from_increments

Scheme = str

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class StrongConvergenceResult:
    """Container for strong convergence diagnostics."""

    scheme: str
    steps: np.ndarray
    dts: np.ndarray
    errors: np.ndarray
    slope: float
    intercept: float
    dts_fit: np.ndarray
    errors_fit: np.ndarray


def strong_order_convergence(
    scheme: Scheme,
    params: CIRParams,
    T: float,
    n_paths: int,
    base_steps_list: Sequence[int],
    seed: int | None = None,
) -> StrongConvergenceResult:
    """Estimate strong convergence order using coupled Brownian increments."""

    if T <= 0:
        raise ValueError("T must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    if len(base_steps_list) < 2:
        raise ValueError("base_steps_list must contain at least two values.")

    scheme_key = scheme.lower()
    steps = sorted({int(s) for s in base_steps_list})
    if steps[-1] <= 0:
        raise ValueError("n_steps entries must be positive.")

    fine_steps = steps[-1]
    dt_fine = T / fine_steps
    rng = make_rng(seed)
    dW_fine = normal_increments(rng, n_paths=n_paths, n_steps=fine_steps, dt=dt_fine)

    terminal_values: dict[int, np.ndarray] = {}
    for n_steps in steps:
        if fine_steps % n_steps != 0:
            raise ValueError(
                "All entries in base_steps_list must divide the finest grid."
            )
        block = fine_steps // n_steps
        if n_steps == fine_steps:
            dW = dW_fine
        else:
            dW = dW_fine.reshape(n_paths, n_steps, block).sum(axis=2)

        dt = T / n_steps
        terminal_values[n_steps] = simulate_terminal_from_increments(
            scheme=scheme_key, params=params, dt=dt, dW=dW
        )

    ref = terminal_values[fine_steps]
    dts = []
    errors = []
    for n_steps in steps:
        dt = T / n_steps
        dts.append(dt)
        if n_steps == fine_steps:
            errors.append(0.0)
            continue
        rmse = np.sqrt(np.mean((terminal_values[n_steps] - ref) ** 2))
        errors.append(rmse)

    dts_arr = np.asarray(dts)
    errors_arr = np.asarray(errors)

    mask = errors_arr > 0
    if mask.sum() < 2:
        raise ValueError(
            "Need at least two strictly positive error levels to fit a slope."
        )

    log_dt = np.log(dts_arr[mask])
    log_err = np.log(errors_arr[mask])
    slope, intercept = np.polyfit(log_dt, log_err, 1)

    return StrongConvergenceResult(
        scheme=scheme_key,
        steps=np.asarray(steps, dtype=int),
        dts=dts_arr,
        errors=errors_arr,
        slope=slope,
        intercept=intercept,
        dts_fit=dts_arr[mask],
        errors_fit=errors_arr[mask],
    )


def run_convergence_report(
    scheme: Scheme,
    params: CIRParams,
    T: float,
    n_paths: int,
    base_steps_list: Sequence[int],
    seed: int | None = None,
) -> tuple[Path, Path]:
    """Compute strong order diagnostics, save CSV and figure."""

    result = strong_order_convergence(
        scheme=scheme,
        params=params,
        T=T,
        n_paths=n_paths,
        base_steps_list=base_steps_list,
        seed=seed,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / f"convergence_{result.scheme}.csv"
    df = pd.DataFrame(
        {
            "scheme": result.scheme,
            "n_steps": result.steps,
            "dt": result.dts,
            "rmse": result.errors,
        }
    )
    df.to_csv(csv_path, index=False)

    fig_path = plot_loglog_convergence(
        dts=result.dts_fit,
        errors=result.errors_fit,
        slope=result.slope,
        intercept=result.intercept,
        path_png=f"convergence_{result.scheme}.png",
    )

    return csv_path, fig_path
