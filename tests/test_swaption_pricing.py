import math

import numpy as np

from benchmarks.models import VasicekParams, vasicek_zero_coupon_price
from examples.utils.swap_helpers import SwapSchedule, swaption_payoff
from streamlit_app.app import MODEL_REGISTRY, price_swaption_for_model


def _intrinsic_swaption_price(params: VasicekParams, schedule: SwapSchedule, strike: float, exercise: float) -> float:
    """Valor intrínseco da swaption no tempo 0 usando descontos analíticos."""

    disc_0_tex = vasicek_zero_coupon_price(params, exercise)
    disc_0_Ti = np.array([vasicek_zero_coupon_price(params, float(Ti)) for Ti in schedule.payment_times])
    disc_fwd = disc_0_Ti / disc_0_tex
    payoff_ex = swaption_payoff(disc_fwd, schedule, strike, kind="payer")
    return float(payoff_ex * disc_0_tex)


def test_swaption_mc_no_arbitrage_extremes():
    # Setup Vasicek quase determinístico para reduzir ruído.
    params = VasicekParams(kappa=0.1, theta=0.05, sigma=0.01, r0=0.05)
    exercise = 1.0
    schedule = SwapSchedule.from_tenor(exercise=exercise, tenor=2.0, freq=2)
    cfg = MODEL_REGISTRY["Vasicek"]

    base_kwargs = dict(
        model_name="Vasicek",
        model_cfg=cfg,
        params=params,
        schedule=schedule,
        n_paths=8000,
        steps_per_year=128,
        exercise=exercise,
        seed=123,
        preferred_scheme="exact",
    )

    # Deep OTM: strike muito acima da curva -> preço ~ 0
    price_otm, stderr_otm = price_swaption_for_model(strike=0.20, kind="payer", **base_kwargs)
    assert price_otm < 1e-3, f"Deep OTM swaption deveria valer ~0, veio {price_otm}"
    assert price_otm < 5 * stderr_otm + 1e-3

    # Deep ITM: strike bem abaixo -> preço ~ valor intrínseco (pouca optionalidade)
    strike_itm = 0.005
    intrinsic = _intrinsic_swaption_price(params, schedule, strike_itm, exercise)
    price_itm, stderr_itm = price_swaption_for_model(strike=strike_itm, kind="payer", **base_kwargs)
    # tolerância baseada no erro padrão do MC + pequeno buffer absoluto
    tol = 5 * stderr_itm + 5e-4
    assert math.isclose(price_itm, intrinsic, rel_tol=0, abs_tol=tol), (
        f"Deep ITM deveria convergir para o valor intrínseco. "
        f"MC={price_itm:.6f}, intrínseco={intrinsic:.6f}, tol={tol:.6f}"
    )

test_swaption_mc_no_arbitrage_extremes()