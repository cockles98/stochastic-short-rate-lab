"""Quick Monte Carlo volatility sanity check against calibrated parameters.

Loads calibration metadata, simulates a modest number of paths for CIR,
Vasicek and Hull-White, and compares terminal standard deviation to the
analytical expectation. If simulated dispersion collapses below a tolerance
threshold, the script exits with an error to flag near-deterministic fits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cir.analytics import variance_short_rate
from cir.params import CIRParams
from cir.simulate import simulate_paths
from benchmarks.models import (
    HullWhiteParams,
    VasicekParams,
    hull_white_variance_short_rate,
    simulate_hull_white_paths,
    simulate_vasicek_paths,
    vasicek_variance_short_rate,
)

DEFAULT_META = Path("benchmarks/data/calibration_meta_17-11-2025.json")


def _load_params(meta_path: Path) -> dict[str, object]:
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models", {})
    cir_raw = models.get("CIR")
    vas_raw = models.get("Vasicek")
    hw_raw = models.get("Hull-White")
    if not (cir_raw and vas_raw and hw_raw):
        raise ValueError("Meta JSON deve conter blocos 'CIR', 'Vasicek' e 'Hull-White'.")
    cir_params = CIRParams(**cir_raw)
    vas_params = VasicekParams(**vas_raw)
    hull_params = HullWhiteParams(
        kappa=hw_raw["kappa"],
        theta=hw_raw["theta"],
        sigma=hw_raw["sigma"],
        r0=hw_raw["r0"],
        shift_times=hw_raw["shift_times"],
        shift_values=hw_raw["shift_values"],
    )
    return {"CIR": cir_params, "Vasicek": vas_params, "Hull-White": hull_params}


def _simulate_terminal_std(
    model: str,
    params,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    if model == "CIR":
        _, paths = simulate_paths("milstein", params, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
        var_th = variance_short_rate(params, T)
    elif model == "Vasicek":
        _, paths = simulate_vasicek_paths("exact", params, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
        var_th = vasicek_variance_short_rate(params, T)
    elif model == "Hull-White":
        _, paths = simulate_hull_white_paths("exact", params, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
        var_th = hull_white_variance_short_rate(params, T)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Modelo desconhecido: {model}")

    terminals = paths[:, -1]
    std_sim = float(np.std(terminals, ddof=1))
    std_th = float(math.sqrt(max(var_th, 0.0)))
    return std_sim, std_th


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida desvio padrao MC para parametros calibrados.")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Arquivo calibration_meta_*.json.")
    parser.add_argument("--paths", type=int, default=10_000, help="Numero de caminhos simulados por modelo.")
    parser.add_argument("--steps", type=int, default=252, help="Passos discretos no intervalo [0,T].")
    parser.add_argument("--T", type=float, default=1.0, help="Horizonte da simulacao (anos).")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Tolerancia minima para o desvio padrao simulado antes de acusar calibracao degenerada.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed base para as simulacoes.")
    args = parser.parse_args()

    params_by_model = _load_params(args.meta)
    for idx, (name, params) in enumerate(params_by_model.items()):
        std_sim, std_th = _simulate_terminal_std(
            model=name,
            params=params,
            T=args.T,
            n_steps=args.steps,
            n_paths=args.paths,
            seed=args.seed + idx,
        )
        print(f"{name}: std_sim={std_sim:.6f} | std_theory={std_th:.6f}")
        if std_sim < args.tol:
            raise SystemExit("Calibracao Deterministica Detetada - Sigma muito baixo")


if __name__ == "__main__":
    main()
