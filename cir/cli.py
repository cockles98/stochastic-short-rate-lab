"""Command-line interface for CIR workflows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Literal

import numpy as np
import pandas as pd
import typer

from .bonds import bond_price_mc, term_structure as compute_term_structure
from .calibration import calibrate_zero_coupon_curve, price_curve
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


def _parse_float_list(value: str) -> List[float]:
    try:
        floats = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise typer.BadParameter("Provide a comma-separated list of numbers.") from exc
    if not floats:
        raise typer.BadParameter("Provide at least one numeric value.")
    return floats


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


@app.command("calibrate-market")
def calibrate_market(
    data_path: Path = typer.Option(
        Path("data/raw_di_curve.csv"), "--data", help="Arquivo CSV com colunas 'date' e 'rate'."
    ),
    maturities: str = typer.Option(
        "0.25,0.5,1.0,2.0,3.0,5.0", "--maturities", help="Lista de maturidades alvo em anos."
    ),
    initial_preset: PresetLiteral = typer.Option(
        "baseline", "--initial-preset", help="Preset usado como chute inicial."
    ),
    curve_out: Path = typer.Option(
        Path("data/calibration_curve.csv"), "--curve-out", help="CSV com preços mercado vs ajustado."
    ),
    params_out: Path = typer.Option(
        Path("data/calibration_params.json"), "--params-out", help="JSON com parâmetros calibrados."
    ),
    last_n: int = typer.Option(
        1, "--last-n", min=1, help="Usar média das últimas N observações do arquivo."
    ),
    penalty: float = typer.Option(1e4, "--penalty", help="Penalidade para violação da Feller."),
) -> None:
    """Calibrate CIR parameters to a simple DI market curve."""

    if not data_path.exists():
        raise typer.BadParameter(f"Arquivo {data_path} não encontrado.")
    df = pd.read_csv(data_path)
    if "rate" not in df.columns:
        raise typer.BadParameter("CSV deve conter a coluna 'rate'.")
    if last_n > len(df):
        raise typer.BadParameter("last-n excede o tamanho do arquivo.")

    rate = float(df["rate"].tail(last_n).astype(float).mean())
    mats = _parse_float_list(maturities)
    market_prices = np.exp(-rate * np.asarray(mats))

    initial = get_params_preset(initial_preset)
    result = calibrate_zero_coupon_curve(
        maturities=mats,
        market_prices=market_prices,
        initial=initial,
        penalty=penalty,
    )

    fitted_prices = price_curve(result.params, mats)
    compare_df = pd.DataFrame(
        {
            "T": mats,
            "market_price": market_prices,
            "fitted_price": fitted_prices,
            "abs_error": np.abs(fitted_prices - market_prices),
        }
    )

    curve_out.parent.mkdir(parents=True, exist_ok=True)
    compare_df.to_csv(curve_out, index=False)

    params_out.parent.mkdir(parents=True, exist_ok=True)
    params_data = {
        "kappa": result.params.kappa,
        "theta": result.params.theta,
        "sigma": result.params.sigma,
        "r0": result.params.r0,
        "success": result.success,
        "message": result.message,
        "loss": result.fun,
    }
    params_out.write_text(json.dumps(params_data, indent=2))

    typer.echo(f"Param calib: {result.params}")
    typer.echo(f"Saved comparison curve to {curve_out}")
    typer.echo(f"Saved calibrated params to {params_out}")


def app_main() -> None:
    app()


if __name__ == "__main__":
    app_main()
