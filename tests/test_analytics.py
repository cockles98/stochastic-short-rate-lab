import math

import pytest

from cir.analytics import mean_short_rate, variance_short_rate, zero_coupon_price
from cir.params import get_params_preset


def test_zero_coupon_closed_form_matches_expectation_baseline():
    params = get_params_preset("baseline")
    price = zero_coupon_price(params, maturity=2.0)
    assert 0 < price < 1


def test_zero_coupon_handles_zero_maturity():
    params = get_params_preset("baseline")
    assert zero_coupon_price(params, 0.0) == pytest.approx(1.0)


def test_mean_and_variance_match_known_limits():
    params = get_params_preset("slow-revert")
    assert mean_short_rate(params, T=0.0) == pytest.approx(params.r0)
    assert variance_short_rate(params, T=0.0) == pytest.approx(0.0)
    limit = mean_short_rate(params, T=1e6)
    assert limit == pytest.approx(params.theta, rel=1e-6)
