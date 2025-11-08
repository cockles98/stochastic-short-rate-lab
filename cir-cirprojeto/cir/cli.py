"""Command-line interface for CIR workflows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
import pandas as pd
import typer

from .bonds import bond_price_mc, term_structure as compute_term_structure
from .convergence import run_convergence_report
from .params import CIRParams, get_params_preset
from .plots import plot_paths
from .simulate import (
    plot_terminal_distribution,
    simulate_paths as simulate_paths_func,
)

app = typer.Typer(help="CIR model utilities", no_args_is_help=True)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SchemeLiteral = Literal["em", "milstein"]
PresetLiteral = Literal["baseline", "slow-revert", "fast-revert"]


def _steps_from_years(T: float, steps_per_year: int) -> int:
    return max(1, int(math.ceil(T * steps_per_year)))


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _parse_steps_list(base_steps: str) -> List[int]:
    try:
        steps = [int(item.strip()) for item in base_steps.split(",") if item.strip()]
    except ValueError as exc:
        raise typer.BadParameter("base-steps must be a comma-separated list of ints") from exc
    if not steps:
        raise typer.BadParameter("Provide at least one integer for base-steps.")
    return steps


@app.command("simulate-paths")
def simulate_paths(
    scheme: SchemeLiteral = typer.Option(
        "em", "--scheme", "-s", help="Discretization scheme to use."
    ),
    preset: PresetLiteral = typer.Option(
        "baseline",
        "--preset",
        "-p",
        help="Parameter preset adhering to the Feller condition.",
    ),
    T: float = typer.Option(5.0, "--T", help="Time horizon."),
    steps_per_year: int = typer.Option(
        252, "--steps-per-year", help="Discretization granularity per year."
    ),
    n_paths: int = typer.Option(10, "--n-paths", help="Number of sample paths."),
    seed: int | None = typer.Option(42, "--seed", help="Random seed."),
) -> None:
    """Simulate CIR paths and store a PNG figure under figures/."""

    params = get_params_preset(preset)
    n_steps = _steps_from_years(T, steps_per_year)
    t, paths = simulate_paths_func(
        scheme=scheme,
        params=params,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    title = f"CIR paths — {preset} ({scheme})"
    filename = f"paths_{scheme}_{preset}.png"
    out_path = plot_paths(t, paths, title=title, path_png=filename)
    typer.echo(f"Saved {out_path}")


@app.command("convergence")
def convergence(
    scheme: SchemeLiteral = typer.Option("em", "--scheme", "-s"),
    preset: PresetLiteral = typer.Option("baseline", "--preset", "-p"),
    T: float = typer.Option(1.0, "--T"),
    paths: int = typer.Option(50_000, "--paths"),
    base_steps: str = typer.Option(
        "52,104,208,416,832", "--base-steps", help="Comma-separated time steps."
    ),
    seed: int | None = typer.Option(123, "--seed"),
) -> None:
    """Estimate strong convergence order on a coupled Brownian grid."""

    params = get_params_preset(preset)
    steps_list = _parse_steps_list(base_steps)
    csv_path, fig_path = run_convergence_report(
        scheme=scheme,
        params=params,
        T=T,
        n_paths=paths,
        base_steps_list=steps_list,
        seed=seed,
    )
    typer.echo(f"Saved convergence data to {csv_path}")
    typer.echo(f"Saved log-log figure to {fig_path}")


@app.command("terminal-hist")
def terminal_hist(
    scheme: SchemeLiteral = typer.Option("em", "--scheme", "-s"),
    preset: PresetLiteral = typer.Option("baseline", "--preset", "-p"),
    T: float = typer.Option(5.0, "--T"),
    paths: int = typer.Option(50_000, "--paths"),
    steps_per_year: int = typer.Option(252, "--steps-per-year"),
    seed: int | None = typer.Option(123, "--seed"),
    bins: int = typer.Option(100, "--bins"),
) -> None:
    """Plot the terminal short-rate distribution."""

    n_steps = _steps_from_years(T, steps_per_year)
    fig_path = plot_terminal_distribution(
        scheme=scheme,
        preset=preset,
        T=T,
        n_steps=n_steps,
        n_paths=paths,
        seed=seed,
        bins=bins,
    )
    typer.echo(f"Saved histogram to {fig_path}")


@app.command("bond-price")
def bond_price(
    scheme: SchemeLiteral = typer.Option("em", "--scheme", "-s"),
    preset: PresetLiteral = typer.Option("baseline", "--preset", "-p"),
    T: float = typer.Option(5.0, "--T"),
    paths: int = typer.Option(5_000, "--paths"),
    steps_per_year: int = typer.Option(252, "--steps-per-year"),
    seed: int | None = typer.Option(321, "--seed"),
) -> None:
    """Monte Carlo zero-coupon pricing with tabular output."""

    params = get_params_preset(preset)
    n_steps = _steps_from_years(T, steps_per_year)
    price, stderr = bond_price_mc(
        params=params,
        T=T,
        n_paths=paths,
        n_steps=n_steps,
        seed=seed,
        scheme=scheme,
    )

    data_dir = _ensure_data_dir()
    out_path = data_dir / f"bond_price_{scheme}_{preset}_T{T:.2f}.csv"
    df = pd.DataFrame(
        [
            {
                "scheme": scheme,
                "preset": preset,
                "T": T,
                "price": price,
                "stderr": stderr,
                "n_paths": paths,
                "n_steps": n_steps,
            }
        ]
    )
    df.to_csv(out_path, index=False)

    typer.echo(f"Bond price: {price:.6f} ± {stderr:.6f}")
    typer.echo(f"Saved tabular output to {out_path}")


@app.command("term-structure")
def term_structure(
    scheme: SchemeLiteral = typer.Option("em", "--scheme", "-s"),
    preset: PresetLiteral = typer.Option("baseline", "--preset", "-p"),
    Tmax: float = typer.Option(10.0, "--Tmax", help="Maximum maturity."),
    grid: int = typer.Option(40, "--grid", help="Number of grid points."),
    paths: int = typer.Option(5_000, "--paths"),
    steps_per_year: int = typer.Option(252, "--steps-per-year"),
    seed: int | None = typer.Option(777, "--seed"),
) -> None:
    """Compute and store the zero-coupon term structure."""

    if Tmax <= 0:
        raise typer.BadParameter("Tmax must be positive.")
    if grid <= 0:
        raise typer.BadParameter("grid must be a positive integer.")

    params = get_params_preset(preset)
    maturities = np.linspace(0.25, Tmax, grid)
    csv_path, fig_path = compute_term_structure(
        params=params,
        maturities=maturities,
        n_paths=paths,
        steps_per_year=steps_per_year,
        seed=seed,
        scheme=scheme,
    )
    typer.echo(f"Saved term-structure data to {csv_path}")
    typer.echo(f"Saved figure to {fig_path}")


def app_main() -> None:
    app()


if __name__ == "__main__":
    app_main()
