"""Generate side-by-side outputs for CIR, Vasicek and Hull-White presets.

Usage
-----

    python benchmarks/scripts/run_benchmark.py --preset baseline --maturities 0.5,1,2,5

Creates ``benchmarks/data/benchmark_<preset>.csv`` consolidating analytical and
Monte Carlo zero-coupon prices / terminal statistics para os três modelos.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - CLI path fix
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.models import (  # noqa: E402
    get_hull_white_preset,
    get_vasicek_preset,
    hull_white_bond_price_mc,
    hull_white_mean_short_rate,
    hull_white_variance_short_rate,
    hull_white_zero_coupon_price,
    vasicek_bond_price_mc,
    vasicek_mean_short_rate,
    vasicek_variance_short_rate,
    vasicek_zero_coupon_price,
)
from cir.analytics import mean_short_rate, variance_short_rate, zero_coupon_price
from cir.bonds import bond_price_mc
from cir.params import get_params_preset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _parse_maturities(raw: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Invalid maturity list '{raw}'.") from exc
    if not values:
        raise SystemExit("Provide at least one maturity.")
    if any(val <= 0 for val in values):
        raise SystemExit("Maturities must be positive.")
    return values


def _parse_shift_schedule(raw: str) -> tuple[list[float], list[float]]:
    if not raw:
        return [0.0, 30.0], [0.0, 0.0]
    times: list[float] = []
    values: list[float] = []
    chunks = [chunk.strip() for chunk in raw.split(";") if chunk.strip()]
    if not chunks:
        raise SystemExit("Shift schedule must contain at least one 'time:value' pair.")
    for item in chunks:
        try:
            t_str, v_str = item.split(":")
            times.append(float(t_str))
            values.append(float(v_str))
        except ValueError as exc:  # pragma: no cover - CLI guard
            raise SystemExit(f"Invalid shift pair '{item}'. Use the form time:value;...") from exc
    if times[0] != 0.0:
        times.insert(0, 0.0)
        values.insert(0, values[0])
    return times, values


def generate_report(
    preset: str,
    maturities: Sequence[float],
    n_paths: int,
    steps_per_year: int,
    cir_scheme: str,
    vasicek_scheme: str,
    hw_scheme: str,
    hw_shift: tuple[list[float], list[float]],
    seed: int,
) -> pd.DataFrame:
    """Return a tidy DataFrame with CIR, Vasicek and Hull-White comparatives."""

    results: list[dict[str, float | str]] = []
    cir_params = get_params_preset(preset)
    vas_params = get_vasicek_preset(preset)
    hw_params = get_hull_white_preset(preset, *hw_shift)

    for maturity in maturities:
        n_steps = max(1, int(math.ceil(maturity * steps_per_year)))

        cir_price, cir_std = bond_price_mc(
            params=cir_params,
            T=float(maturity),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            scheme=cir_scheme,
        )
        results.append(
            {
                "model": "CIR",
                "maturity": maturity,
                "analytic_price": zero_coupon_price(cir_params, maturity),
                "mc_price": cir_price,
                "mc_std": cir_std,
                "mean_rT": mean_short_rate(cir_params, maturity),
                "var_rT": variance_short_rate(cir_params, maturity),
            }
        )

        vas_price, vas_std = vasicek_bond_price_mc(
            params=vas_params,
            T=float(maturity),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            scheme=vasicek_scheme,
        )
        results.append(
            {
                "model": "Vasicek",
                "maturity": maturity,
                "analytic_price": vasicek_zero_coupon_price(vas_params, maturity),
                "mc_price": vas_price,
                "mc_std": vas_std,
                "mean_rT": vasicek_mean_short_rate(vas_params, maturity),
                "var_rT": vasicek_variance_short_rate(vas_params, maturity),
            }
        )

        hw_price, hw_std = hull_white_bond_price_mc(
            params=hw_params,
            T=float(maturity),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            scheme=hw_scheme,
        )
        results.append(
            {
                "model": "Hull-White",
                "maturity": maturity,
                "analytic_price": hull_white_zero_coupon_price(hw_params, maturity),
                "mc_price": hw_price,
                "mc_std": hw_std,
                "mean_rT": hull_white_mean_short_rate(hw_params, maturity),
                "var_rT": hull_white_variance_short_rate(hw_params, maturity),
            }
        )

    return pd.DataFrame(results)


def main() -> None:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description="Benchmark CIR, Vasicek e Hull-White.")
    parser.add_argument("--preset", default="baseline", help="Preset name shared by both models.")
    parser.add_argument(
        "--maturities",
        default="0.5,1,2,5",
        help="Comma-separated list of maturities (years).",
    )
    parser.add_argument("--n-paths", type=int, default=5000)
    parser.add_argument("--steps-per-year", type=int, default=252)
    parser.add_argument("--cir-scheme", default="milstein", choices=["em", "milstein"])
    parser.add_argument("--vasicek-scheme", default="exact", choices=["em", "exact"])
    parser.add_argument("--hw-scheme", default="exact", choices=["em", "exact"])
    parser.add_argument(
        "--hw-shift",
        default="0:0.0;30:0.0",
        help="Semicolon-separated shift schedule 't:value'. Example: 0:0.0;5:0.01;10:0.015",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional custom CSV path (defaults to benchmarks/data/benchmark_<preset>.csv).",
    )
    args = parser.parse_args()

    maturities = _parse_maturities(args.maturities)
    shift_schedule = _parse_shift_schedule(args.hw_shift)
    df = generate_report(
        preset=args.preset,
        maturities=maturities,
        n_paths=int(args.n_paths),
        steps_per_year=int(args.steps_per_year),
        cir_scheme=args.cir_scheme,
        vasicek_scheme=args.vasicek_scheme,
        hw_scheme=args.hw_scheme,
        hw_shift=shift_schedule,
        seed=int(args.seed),
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or DATA_DIR / f"benchmark_{args.preset}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved benchmark comparison to {out_path}")


if __name__ == "__main__":
    main()
