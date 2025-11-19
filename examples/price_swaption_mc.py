"""Monte Carlo pricing for payer/receiver swaptions using CIR/Vasicek/Hull-White."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.models import (
    HullWhiteParams,
    VasicekParams,
    get_hull_white_preset,
    get_vasicek_preset,
    hull_white_bond_price_mc,
    simulate_hull_white_paths,
    simulate_vasicek_paths,
    vasicek_bond_price_mc,
)
from cir.bonds import discount_factors_from_paths
from cir.params import CIRParams, get_params_preset
from cir.simulate import simulate_paths
from examples.utils.swap_helpers import SwapSchedule, discount_path, swaption_payoff

ModelName = Literal["CIR", "Vasicek", "Hull-White"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo swaption pricing.")
    parser.add_argument("--model", default="CIR", choices=["CIR", "Vasicek", "Hull-White"])
    parser.add_argument("--preset", default="baseline", help="Preset (baseline/slow-revert/fast-revert).")
    parser.add_argument("--kind", default="payer", choices=["payer", "receiver"], help="Tipo da swaption.")
    parser.add_argument("--exercise", type=float, default=2.0, help="Tempo do exercício (anos).")
    parser.add_argument("--tenor", type=float, default=3.0, help="Tenor do swap subjacente.")
    parser.add_argument("--freq", type=int, default=2, help="Pagamentos fixos por ano (2=semi).")
    parser.add_argument("--strike", type=float, default=0.04, help="Taxa fixa do swap.")
    parser.add_argument("--paths", type=int, default=20000, help="Número de trajetórias para o MC.")
    parser.add_argument("--steps-per-year", type=int, default=252, help="Passos por ano na simulação.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="Arquivo opcional para salvar resultados em JSON.")
    return parser.parse_args()


def load_params(model: ModelName, preset: str) -> CIRParams | VasicekParams | HullWhiteParams:
    if model == "CIR":
        return get_params_preset(preset)
    if model == "Vasicek":
        return get_vasicek_preset(preset)
    if model == "Hull-White":
        return get_hull_white_preset(preset)
    raise ValueError(f"Modelo {model} não suportado.")


def simulate(model: ModelName, params, T: float, n_steps: int, n_paths: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    simulators = {
        "CIR": simulate_paths,
        "Vasicek": simulate_vasicek_paths,
        "Hull-White": simulate_hull_white_paths,
    }
    try:
        t, paths = simulators[model](
            scheme="milstein" if model == "CIR" else "exact",
            params=params,
            T=T,
            n_steps=n_steps,
            n_paths=n_paths,
            seed=seed,
        )
        return t, paths
    except Exception as exc:
        raise RuntimeError(f"Erro ao simular caminhos para {model}: {exc}") from exc


def price_swaption(
    model: ModelName,
    params,
    schedule: SwapSchedule,
    kind: str,
    strike: float,
    n_paths: int,
    n_steps: int,
    exercise: float,
    seed: int,
) -> tuple[float, float]:
    total_T = schedule.payment_times[-1]
    t, paths = simulate(model, params, total_T, n_steps, n_paths, seed)
    time_idx = t <= exercise
    exercise_grid = t[time_idx]
    payoffs = []
    for path in paths:
        grid = exercise_grid
        r_segment = path[time_idx]
        integral = np.trapz(r_segment, grid)
        discount_ex = np.exp(-integral)
        schedule_discounts = []
        for ti in schedule.payment_times:
            mask = t <= ti
            integral_t = np.trapz(path[mask], t[mask])
            schedule_discounts.append(np.exp(-integral_t))
        payoff = swaption_payoff(np.array(schedule_discounts), schedule, strike, kind)
        payoffs.append(discount_ex * payoff)
    payoffs_arr = np.array(payoffs)
    return float(payoffs_arr.mean()), float(payoffs_arr.std(ddof=1) / np.sqrt(n_paths))


def main() -> None:
    args = parse_args()
    params = load_params(args.model, args.preset)
    schedule = SwapSchedule.from_tenor(args.exercise, args.tenor, args.freq)
    n_steps = int((args.exercise + args.tenor) * args.steps_per_year)
    price, stderr = price_swaption(
        model=args.model,
        params=params,
        schedule=schedule,
        kind=args.kind,
        strike=args.strike,
        n_paths=args.paths,
        n_steps=n_steps,
        exercise=args.exercise,
        seed=args.seed,
    )
    result = {
        "model": args.model,
        "preset": args.preset,
        "kind": args.kind,
        "exercise": args.exercise,
        "tenor": args.tenor,
        "freq": args.freq,
        "strike": args.strike,
        "paths": args.paths,
        "steps_per_year": args.steps_per_year,
        "price": price,
        "stderr": stderr,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        print(f"Resultado salvo em {args.out}")


if __name__ == "__main__":
    main()
