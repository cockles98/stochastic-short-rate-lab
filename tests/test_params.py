"""Tests for CIR parameter utilities."""

import pytest

from cir.params import CIRParams, get_params_preset


@pytest.mark.parametrize("name", ["baseline", "slow-revert", "fast-revert"])
def test_presets_respect_feller_condition(name: str) -> None:
    params = get_params_preset(name)
    assert 2 * params.kappa * params.theta > params.sigma**2


def test_cirparams_enforces_feller_condition() -> None:
    with pytest.raises(ValueError):
        CIRParams(kappa=0.1, theta=0.01, sigma=1.0, r0=0.03)


def test_get_params_preset_returns_expected_values() -> None:
    preset = get_params_preset("baseline")
    assert preset.kappa == pytest.approx(1.2)
    assert preset.theta == pytest.approx(0.05)
    assert preset.sigma == pytest.approx(0.20)
    assert preset.r0 == pytest.approx(0.03)


def test_get_params_preset_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_params_preset("unknown")
