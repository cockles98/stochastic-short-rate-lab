import numpy as np
import pytest

from cir.analytics import zero_coupon_price
from cir.calibration import calibrate_zero_coupon_curve, price_curve
from cir.params import CIRParams, get_params_preset


def test_price_curve_matches_zero_coupon():
    params = get_params_preset("baseline")
    mats = [0.5, 1.0]
    prices = price_curve(params, mats)
    assert np.allclose(prices, [zero_coupon_price(params, m) for m in mats])


def test_calibrate_zero_coupon_recovers_curve_shape():
    true_params = CIRParams(kappa=1.2, theta=0.05, sigma=0.20, r0=0.03)
    maturities = np.linspace(0.25, 3.0, 12)
    market = price_curve(true_params, maturities)

    initial = CIRParams(kappa=0.8, theta=0.04, sigma=0.15, r0=0.02)
    result = calibrate_zero_coupon_curve(maturities, market, initial)

    assert result.success
    fitted = price_curve(result.params, maturities)
    mae = float(np.mean(np.abs(fitted - market)))
    assert mae < 2e-4
