"""Tests for CIR SDE discretizations and simulation helpers."""

import numpy as np
import pytest

from cir.params import CIRParams
from cir.rng import make_rng
from cir.sde import euler_maruyama, milstein
from cir.simulate import run_scenarios, simulate_paths


def test_euler_maruyama_mean_reversion(baseline_params) -> None:
    rng = make_rng(123)
    n_paths, n_steps, T = 200, 200, 5.0
    paths = euler_maruyama(
        params=baseline_params, T=T, n_steps=n_steps, n_paths=n_paths, rng=rng
    )
    assert paths.shape == (n_paths, n_steps + 1)
    assert np.all(paths >= 0)
    final_mean = paths[:, -1].mean()
    assert final_mean == pytest.approx(baseline_params.theta, abs=0.02, rel=0.2)


def test_milstein_matches_em_when_sigma_zero() -> None:
    params = CIRParams(kappa=0.9, theta=0.04, sigma=0.0, r0=0.03)
    args = dict(params=params, T=1.0, n_steps=10, n_paths=4)
    em_paths = euler_maruyama(rng=make_rng(0), **args)
    mil_paths = milstein(rng=make_rng(0), **args)
    assert np.allclose(em_paths, mil_paths)


def test_simulate_paths_returns_time_grid(baseline_params) -> None:
    t, paths = simulate_paths(
        scheme="milstein",
        params=baseline_params,
        T=1.0,
        n_steps=10,
        n_paths=3,
        seed=7,
    )
    assert np.allclose(t, np.linspace(0.0, 1.0, 11))
    assert paths.shape == (3, 11)
    assert np.all(paths >= 0)


@pytest.mark.parametrize("scheme", ["em", "milstein"])
def test_short_horizon_paths_are_non_negative_and_finite(baseline_params, scheme) -> None:
    T = 0.25
    n_steps = 64
    n_paths = 256
    _, paths = simulate_paths(
        scheme=scheme,
        params=baseline_params,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=999,
    )
    assert paths.shape == (n_paths, n_steps + 1)
    assert np.isfinite(paths).all()
    assert (paths >= 0).all()


def test_run_scenarios_creates_pngs_and_cleans() -> None:
    outputs = run_scenarios(scheme="em", T=0.2, n_steps=20, n_paths=5, seed=1)
    assert len(outputs) == 3
    for path in outputs:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
        path.unlink()
