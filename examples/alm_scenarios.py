"""ALM scenario generator applying deterministic shocks to cash-flow ladders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.models import get_hull_white_preset, simulate_hull_white_paths
from cir.simulate import simulate_paths
from cir.params import get_params_preset
from cir.bonds import discount_factors_from_paths
from examples.utils.scenario_builders import DEFAULT_SCENARIOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construção de cenários ALM simplificados.")
    parser.add_argument("--cashflows", type=Path, required=True, help="Arquivo JSON com ativos/passivos.")
    parser.add_argument("--model", default="CIR", choices=["CIR", "Hull-White"], help="Modelo para gerar curvas base.")
    parser.add_argument("--preset", default="baseline")
    parser.add_argument("--paths", type=int, default=1000, help="Caminhos MC para gerar curva média.")
    parser.add_argument("--steps-per-year", type=int, default=252)
    parser.add_argument("--horizon", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=Path, default=Path("examples/output/alm_report.csv"))
    return parser.parse_args()


def load_cashflows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expect JSON with fields assets/passives containing list of {time, amount}."""

    data = json.loads(path.read_text())
    assets = pd.DataFrame(data.get("assets", []))
    passives = pd.DataFrame(data.get("passives", []))
    for df, label in [(assets, "assets"), (passives, "passives")]:
        if df.empty or not {"time", "amount"} <= set(df.columns):
            raise ValueError(f"{label} precisa ter campos 'time' e 'amount'.")
        df.sort_values("time", inplace=True)
    return assets, passives


def compute_discount_curve(model: str, preset: str, horizon: float, n_paths: int, steps_per_year: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    params = get_params_preset(preset)
    n_steps = int(horizon * steps_per_year)
    t, paths = simulate_paths(
        scheme="milstein",
        params=params,
        T=horizon,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    discounts = discount_factors_from_paths(t, paths)
    mean_curve = paths.mean(axis=0)
    return t, mean_curve


def pv_cashflows(df: pd.DataFrame, curve_times: np.ndarray, curve_values: np.ndarray) -> float:
    times = df["time"].to_numpy(dtype=float)
    amounts = df["amount"].to_numpy(dtype=float)
    # Integral acumulada da curva via regra do trapézio para obter D(T) = exp(-∫ r(u) du).
    dt = np.diff(curve_times)
    trap = 0.5 * dt * (curve_values[1:] + curve_values[:-1])
    integral_curve = np.concatenate([[0.0], np.cumsum(trap)])
    integral_at_T = np.interp(times, curve_times, integral_curve)
    discounts = np.exp(-integral_at_T)
    return float(np.sum(amounts * discounts))


def duration(df: pd.DataFrame, curve_times: np.ndarray, curve_values: np.ndarray) -> float:
    times = df["time"].to_numpy(dtype=float)
    amounts = df["amount"].to_numpy(dtype=float)
    dt = np.diff(curve_times)
    trap = 0.5 * dt * (curve_values[1:] + curve_values[:-1])
    integral_curve = np.concatenate([[0.0], np.cumsum(trap)])
    integral_at_T = np.interp(times, curve_times, integral_curve)
    discounts = np.exp(-integral_at_T)
    pv = np.sum(amounts * discounts)
    if pv == 0:
        return 0.0
    weighted = np.sum(times * amounts * discounts)
    return float(weighted / pv)


def apply_scenarios(base_curve: np.ndarray, scenarios) -> dict[str, np.ndarray]:
    result = {}
    for name, fn in scenarios.items():
        result[name] = fn(base_curve)
    return result


def main() -> None:
    args = parse_args()
    assets, passives = load_cashflows(args.cashflows)
    curve_times = np.linspace(0.0, args.horizon, int(args.horizon * args.steps_per_year) + 1)
    _, base_curve = compute_discount_curve(
        args.model,
        args.preset,
        args.horizon,
        args.paths,
        args.steps_per_year,
        args.seed,
    )
    scenarios = apply_scenarios(base_curve, DEFAULT_SCENARIOS)
    records = []
    for name, curve in scenarios.items():
        pv_assets = pv_cashflows(assets, curve_times, curve)
        pv_passives = pv_cashflows(passives, curve_times, curve)
        duration_assets = duration(assets, curve_times, curve)
        duration_passives = duration(passives, curve_times, curve)
        records.append(
            {
                "scenario": name,
                "pv_assets": pv_assets,
                "pv_passives": pv_passives,
                "pv_net": pv_assets - pv_passives,
                "duration_assets": duration_assets,
                "duration_passives": duration_passives,
                "duration_gap": duration_assets - duration_passives,
            }
        )
    out_df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(out_df)
    print(f"Resultado salvo em {args.out}")


if __name__ == "__main__":
    main()
