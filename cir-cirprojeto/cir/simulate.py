"""High-level simulation helpers for CIR path generation and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .params import CIRParams, get_params_preset
from .plots import plot_hist_terminal
from .rng import make_rng
from .sde import euler_maruyama, milstein, simulate_terminal

Scheme = Literal["em", "milstein"]

__all__ = ["simulate_paths", "run_scenarios"]
__all__.append("plot_terminal_distribution")


def simulate_paths(
    scheme: Scheme,
    params: CIRParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate CIR sample paths with the selected discretization scheme.

    Parameters
    ----------
    scheme
        Either ``"em"`` (Euler–Maruyama) or ``"milstein"``.
    params
        Instance of :class:`cir.params.CIRParams` which already satisfies
        the Feller condition.
    T
        Horizon in years (or consistent time unit).
    n_steps
        Number of discretization steps.
    n_paths
        Number of Monte Carlo trajectories to generate.
    seed
        Optional random seed for reproducibility.
    rng
        Optional ``np.random.Generator`` to enable chunked simulations without
        re-seeding between batches.

    Returns
    -------
    t, R
        ``t`` is a 1-D array with the time grid, ``R`` is an array with shape
        ``(n_paths, n_steps + 1)`` that includes the initial rate ``r0``.
    """

    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer.")

    if rng is not None and seed is not None:
        raise ValueError("Provide either rng or seed, not both.")

    scheme_key = scheme.lower()
    steppers = {
        "em": euler_maruyama,
        "milstein": milstein,
    }
    try:
        stepper = steppers[scheme_key]
    except KeyError as exc:
        raise ValueError("scheme must be 'em' or 'milstein'.") from exc

    local_rng = rng if rng is not None else make_rng(seed)
    rates = stepper(params=params, T=T, n_steps=n_steps, n_paths=n_paths, rng=local_rng)
    t = np.linspace(0.0, T, n_steps + 1)
    return t, rates


def run_scenarios(
    scheme: Scheme = "em",
    T: float = 2.0,
    n_steps: int = 500,
    n_paths: int = 8,
    seed: int | None = 42,
) -> list[Path]:
    """Generate and save illustrative CIR trajectories for the three presets.

    The function simulates between five and ten paths per preset (default 8) and
    stores PNG figures under ``figures/``. ``n_paths`` can be increased to
    create denser spaghetti plots when needed.
    """

    if not (5 <= n_paths <= 10):
        raise ValueError("n_paths should be between 5 and 10 for visualization clarity.")

    figure_dir = Path(__file__).resolve().parent.parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    presets = ("baseline", "slow-revert", "fast-revert")
    base_seed = seed

    for idx, preset_name in enumerate(presets):
        params = get_params_preset(preset_name)
        scenario_seed = None if base_seed is None else base_seed + idx
        t, rates = simulate_paths(
            scheme=scheme,
            params=params,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=scenario_seed,
        )

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for path in rates:
            ax.plot(t, path, alpha=0.9, linewidth=1.0)
        ax.set(
            title=f"CIR paths — {preset_name} ({scheme.lower()})",
            xlabel="Time",
            ylabel="Short rate",
        )
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        fig.tight_layout()

        out_path = figure_dir / f"paths_{scheme.lower()}_{preset_name}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def plot_terminal_distribution(
    scheme: Scheme,
    preset: str,
    T: float = 5.0,
    n_steps: int = 5 * 252,
    n_paths: int = 50_000,
    seed: int | None = 123,
    bins: int = 100,
    chunk_size: int | None = None,
) -> Path:
    """Simulate many paths and save the histogram of terminal rates."""

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided.")

    params = get_params_preset(preset)
    figure_chunks = []
    remaining = n_paths
    effective_chunk = (
        min(n_paths, 10_000) if chunk_size is None else min(chunk_size, n_paths)
    )
    rng = make_rng(seed)

    while remaining > 0:
        batch = min(effective_chunk, remaining)
        terminals = simulate_terminal(
            scheme=scheme,
            params=params,
            T=T,
            n_steps=n_steps,
            n_paths=batch,
            rng=rng,
        )
        figure_chunks.append(terminals)
        remaining -= batch

    terminal = np.concatenate(figure_chunks, axis=0)
    title = f"Terminal distribution — {preset} ({scheme.lower()})"
    filename = f"hist_terminal_{scheme.lower()}_{preset}.png"
    return plot_hist_terminal(terminal, bins=bins, title=title, path_png=filename)
