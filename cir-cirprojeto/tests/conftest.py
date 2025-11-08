"""Pytest fixtures and configuration for CIR project tests."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cir.params import CIRParams, get_params_preset

matplotlib.use("Agg", force=True)


@pytest.fixture
def baseline_params() -> CIRParams:
    """Convenience fixture for the baseline preset."""

    return get_params_preset("baseline")
