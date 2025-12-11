import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.models import (
    HullWhiteParams,
    VasicekParams,
    calibrate_vasicek_curve,
    hull_white_bond_price_mc,
    hull_white_mean_short_rate,
    hull_white_price_curve,
    hull_white_variance_short_rate,
    hull_white_zero_coupon_price,
    simulate_hull_white_paths,
    simulate_vasicek_paths,
    vasicek_bond_price_mc,
    vasicek_mean_short_rate,
    vasicek_price_curve,
    vasicek_variance_short_rate,
    vasicek_zero_coupon_price,
)
from cir.analytics import mean_short_rate, variance_short_rate, zero_coupon_price
from cir.bonds import bond_price_mc
from cir.calibration import calibrate_zero_coupon_curve, price_curve, market_curve_from_file
from cir.convergence import strong_order_convergence
from cir.params import CIRParams
from cir.simulate import simulate_paths
from cir.validation import (
    compare_moments,
    compare_zero_coupon_prices,
    zero_coupon_error_by_steps,
)
from examples.utils.scenario_builders import DEFAULT_SCENARIOS
from examples.utils.swap_helpers import SwapSchedule, swaption_payoff
from data_loaders.curves import load_curve_components
from data_loaders.selic import get_latest_rate, load_selic_csv

alt.data_transformers.disable_max_rows()


def _integrated_discount(path: np.ndarray, t: np.ndarray, T: float) -> float:
    mask = t <= T
    if not np.any(mask):
        return 1.0
    integral = np.trapz(path[mask], t[mask])
    return float(math.exp(-integral))


def _replace_r0(params, new_r0: float):
    if isinstance(params, CIRParams):
        return CIRParams(kappa=params.kappa, theta=params.theta, sigma=params.sigma, r0=new_r0)
    if isinstance(params, VasicekParams):
        return VasicekParams(kappa=params.kappa, theta=params.theta, sigma=params.sigma, r0=new_r0)
    if isinstance(params, HullWhiteParams):
        return HullWhiteParams(
            kappa=params.kappa,
            theta=params.theta,
            sigma=params.sigma,
            r0=new_r0,
            shift_times=params.shift_times,
            shift_values=params.shift_values,
        )
    return params


def price_swaption_for_model(
    model_name: str,
    model_cfg: dict,
    params,
    schedule: SwapSchedule,
    strike: float,
    kind: str,
    n_paths: int,
    steps_per_year: int,
    exercise: float,
    seed: int,
    preferred_scheme: str,
) -> tuple[float, float]:
    # Simula apenas até o exercício para evitar viés de predição perfeita.
    total_T = float(schedule.payment_times[-1])
    n_steps = max(1, int(exercise * steps_per_year))
    scheme = preferred_scheme if preferred_scheme in model_cfg["schemes"] else model_cfg["schemes"][model_cfg["default_scheme_index"]]
    t, paths = model_cfg["simulate"](
        scheme=scheme,
        params=params,
        T=exercise,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    zero_coupon_fn = model_cfg["zero_coupon"]
    payoffs = []
    for path in paths:
        # desconto 0->T_ex a partir da simulação
        disc_ex = _integrated_discount(path, t, exercise)
        # valor do swap no exercício usando preço analítico com r(T_ex) como novo r0
        r_tex = float(path[-1])
        params_tex = _replace_r0(params, r_tex)
        discounts_fwd = np.array(
            [zero_coupon_fn(params_tex, float(Ti - exercise)) for Ti in schedule.payment_times]
        )
        payoff_ex = swaption_payoff(discounts_fwd, schedule, strike, kind)
        payoffs.append(payoff_ex * disc_ex)
    payoffs = np.asarray(payoffs)
    price = float(payoffs.mean())
    stderr = float(payoffs.std(ddof=1) / math.sqrt(len(payoffs))) if len(payoffs) > 1 else 0.0
    return price, stderr


def calibrate_models_from_curve(
    curve_file: str | Path, curve_kind: str, sync_volatility: bool = False
) -> tuple[dict[str, object], str]:
    mats, prices, market_rates, ref_date = market_curve_from_file(curve_file, curve_kind=curve_kind)
    cir_initial = CIRParams(kappa=1.0, theta=0.05, sigma=0.2, r0=0.05)
    cir_result = calibrate_zero_coupon_curve(mats, prices, cir_initial)
    vas_initial = VasicekParams(kappa=1.0, theta=0.05, sigma=0.15, r0=cir_initial.r0)
    vas_result = calibrate_vasicek_curve(mats, prices, vas_initial)
    cir_sigma = cir_result.params.sigma
    vas_sigma_raw = vas_result.params.sigma
    if vas_sigma_raw < 0.1 * cir_sigma:
        st.warning(
            f"Volatilidade Vasicek ({vas_sigma_raw:.4f}) ficou abaixo de 10% da vol CIR ({cir_sigma:.4f}). "
            "Isso pode subestimar o risco; considere ativar 'Sincronizar Volatilidade'."
        )
    vas_params = vas_result.params
    if sync_volatility:
        vas_params = VasicekParams(
            kappa=vas_params.kappa,
            theta=vas_params.theta,
            sigma=cir_sigma,
            r0=vas_params.r0,
        )
    vas_prices = vasicek_price_curve(vas_params, mats)
    vas_yields = -np.log(np.maximum(vas_prices, 1e-12)) / np.maximum(mats, 1e-6)
    shift_times = np.insert(mats, 0, 0.0)
    shift_values = np.insert(market_rates - vas_yields, 0, 0.0)
    hull_params = HullWhiteParams(
        kappa=vas_params.kappa,
        theta=vas_params.theta,
        sigma=vas_params.sigma,
        r0=vas_params.r0,
        shift_times=shift_times,
        shift_values=shift_values,
    )
    params = {
        "CIR": cir_result.params,
        "Vasicek": vas_params,
        "Hull-White": hull_params,
    }
    return params, ref_date


@st.cache_data(show_spinner=False)
def load_curve_components_cached(curve_file: str | Path):
    return load_curve_components(curve_file)


@st.cache_data(show_spinner=False)
def calibrate_models_cached(curve_file: str | Path, curve_kind: str, sync_volatility: bool = False):
    return calibrate_models_from_curve(curve_file, curve_kind, sync_volatility=sync_volatility)


@st.cache_data(show_spinner=False)
def load_selic_cached(real_selic_file: str | Path):
    df = load_selic_csv(real_selic_file)
    return df, get_latest_rate(df)


@st.cache_data(show_spinner="Calculando term structure...")
def _term_structure_cached(
    maturities: np.ndarray,
    model_name: str,
    params,
    steps_per_year: int,
    scheme: str,
    n_paths: int,
    base_seed: int,
):
    bond_price_fn = MODEL_REGISTRY[model_name]["bond_price"]
    rows = []
    for idx, maturity in enumerate(maturities):
        steps = max(1, int(maturity * steps_per_year))
        price, stderr = bond_price_fn(
            params=params,
            T=float(maturity),
            n_paths=n_paths,
            n_steps=steps,
            seed=base_seed + idx,
            scheme=scheme,
        )
        yld = 0.0 if maturity == 0 else -math.log(max(price, 1e-12)) / maturity
        rows.append({"T": maturity, "price": price, "stderr": stderr, "zero_rate": yld})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Rodando validação analítica...")
def _validation_cached(params, val_maturities, val_paths, val_steps, seed, scheme: str):
    price_df = compare_zero_coupon_prices(
        params=params,
        maturities=val_maturities,
        n_paths=int(val_paths),
        steps_per_year=int(val_steps),
        seed=int(seed) * 3,
    )
    err_df = zero_coupon_error_by_steps(
        params=params,
        maturity=1.0,
        n_paths=int(val_paths),
        steps_list=[32, 64, 128, 256],
        seed=int(seed) * 5,
        scheme=scheme,
    )
    moment_result = compare_moments(
        params=params,
        T=1.0,
        n_paths=int(val_paths),
        n_steps=int(val_steps),
        seed=int(seed) * 7,
        scheme=scheme,
    )
    return price_df, err_df, moment_result


@st.cache_data(show_spinner="Calculando swaption...")
def _swaption_cached(
    compare_models_ordered: list[str],
    params_by_model: dict[str, object],
    swaption_settings: dict,
    seed: int,
    scheme: str,
):
    rows = []
    schedule = SwapSchedule.from_tenor(
        exercise=swaption_settings["exercise"],
        tenor=swaption_settings["tenor"],
        freq=swaption_settings["freq"],
    )
    for idx, name in enumerate(compare_models_ordered):
        cfg = MODEL_REGISTRY[name]
        params_cmp = params_by_model.get(name)
        if params_cmp is None:
            continue
        price_mc, stderr_mc = price_swaption_for_model(
            model_name=name,
            model_cfg=cfg,
            params=params_cmp,
            schedule=schedule,
            strike=swaption_settings["strike"],
            kind=swaption_settings["kind"],
            n_paths=swaption_settings["paths"],
            steps_per_year=swaption_settings["steps_per_year"],
            exercise=swaption_settings["exercise"],
            seed=int(seed) + idx * 17,
            preferred_scheme=scheme,
        )
        rows.append({"Modelo": name, "Preco": price_mc, "Erro padrao": stderr_mc})
    return pd.DataFrame(rows).set_index("Modelo") if rows else pd.DataFrame()


@st.cache_data(show_spinner="Calculando cenários ALM...")
def _alm_cached(
    model_name: str,
    params,
    alm_settings: dict,
    scheme: str,
    assets_df: pd.DataFrame,
    passives_df: pd.DataFrame,
):
    cfg = MODEL_REGISTRY[model_name]
    scheme_alm = scheme if scheme in cfg["schemes"] else cfg["schemes"][cfg["default_scheme_index"]]
    max_time = float(max(assets_df["time"].max(), passives_df["time"].max()))
    horizon = max(1.0, max_time + 0.5)
    n_steps = int(horizon * alm_settings["steps_per_year"])
    t_curve, paths_curve = cfg["simulate"](
        scheme=scheme_alm,
        params=params,
        T=horizon,
        n_steps=n_steps,
        n_paths=alm_settings["paths"],
        seed=alm_settings["seed"],
    )
    mean_curve = paths_curve.mean(axis=0)
    scenario_curves = {name: fn(mean_curve) for name, fn in DEFAULT_SCENARIOS.items()}
    records = []
    for sc_name, curve in scenario_curves.items():
        pv_assets = _pv_cashflows(assets_df, t_curve, curve)
        pv_passives = _pv_cashflows(passives_df, t_curve, curve)
        records.append(
            {
                "Cenario": sc_name,
                "PV ativos": pv_assets,
                "PV passivos": pv_passives,
                "PV liquido": pv_assets - pv_passives,
                "Duration ativos": _duration(assets_df, t_curve, curve),
                "Duration passivos": _duration(passives_df, t_curve, curve),
            }
        )
    df_alm = pd.DataFrame(records).set_index("Cenario")
    net_base = (
        assets_df.groupby("time")["amount"].sum()
        - passives_df.groupby("time")["amount"].sum()
    ).reset_index(name="net_amount")
    net_base = net_base.set_index("time")
    return t_curve, scenario_curves, df_alm, net_base


def _build_cashflow_df(data: list[dict], label: str) -> pd.DataFrame:
    df = pd.DataFrame(data)
    if df.empty or not {"time", "amount"} <= set(df.columns):
        raise ValueError(f"{label} precisa conter 'time' e 'amount'.")
    df = df.astype({"time": float, "amount": float}).sort_values("time")
    return df


def _pv_cashflows(df: pd.DataFrame, times: np.ndarray, rates: np.ndarray) -> float:
    interp_rates = np.interp(df["time"].to_numpy(), times, rates, left=rates[0], right=rates[-1])
    discounts = np.exp(-interp_rates * df["time"].to_numpy())
    return float(np.sum(df["amount"].to_numpy() * discounts))


def _duration(df: pd.DataFrame, times: np.ndarray, rates: np.ndarray) -> float:
    interp_rates = np.interp(df["time"].to_numpy(), times, rates, left=rates[0], right=rates[-1])
    discounts = np.exp(-interp_rates * df["time"].to_numpy())
    pv = np.sum(df["amount"].to_numpy() * discounts)
    if pv == 0:
        return 0.0
    weighted = np.sum(df["time"].to_numpy() * df["amount"].to_numpy() * discounts)
    return float(weighted / pv)


def _read_csv_fallback(path: Path) -> pd.DataFrame:
    """Load CSV tolerating encodings and separators (UTF-8/Latin-1/ISO + ;)."""

    attempts = [
        {},
        {"sep": ";"},
        {"encoding": "latin-1"},
        {"encoding": "latin-1", "sep": ";"},
        {"encoding": "ISO-8859-1", "engine": "python"},
        {"encoding": "ISO-8859-1", "engine": "python", "sep": ";"},
    ]
    last_exc: Exception | None = None
    for opts in attempts:
        try:
            return pd.read_csv(path, **opts)
        except Exception as exc:  # keep last error for context
            last_exc = exc
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Falha ao ler CSV.")


def _parse_rate_fraction(series: pd.Series) -> pd.Series:
    """Parse rates to fraction handling . or , as decimal and auto-percent."""

    def to_float(val: object) -> float:
        s = str(val).strip()
        if not s:
            return float("nan")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return float("nan")

    vals = series.map(to_float).to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanmedian(finite) > 1:
        vals = vals / 100.0
    return pd.Series(vals)


def _parse_rate_fraction(series: pd.Series) -> pd.Series:
    """Parse rates to fraction handling '.', ',' and auto-percent."""

    def to_float(val: object) -> float:
        s = str(val).strip()
        if not s:
            return float("nan")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return float("nan")

    vals = series.map(to_float).to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanmedian(finite) > 1:
        vals = vals / 100.0
    return pd.Series(vals)


def _extract_rate_column(df: pd.DataFrame) -> pd.Series:
    """Return a numeric rate column from common headers."""

    norm_map = {col.lower().strip(): col for col in df.columns}
    for key in ["rate", "rate_annual", "taxa", "taxa (% a.a.)", "valor"]:
        if key in norm_map:
            return _parse_rate_fraction(df[norm_map[key]])
    raise ValueError(f"CSV precisa conter a coluna 'rate'. Colunas encontradas: {list(df.columns)}")


def _get_rate_series(df: pd.DataFrame, path: Path) -> pd.Series:
    """Try to obtain a rate series; fallback to CurvaZero parser when applicable."""

    try:
        return _extract_rate_column(df)
    except ValueError:
        pass
    return pd.Series(dtype=float)


def _curve_from_curva_zero(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract maturities (years) and prices from CurvaZero_*.csv."""

    try:
        components = load_curve_components(path)
    except Exception:
        return None
    pref = components.prefixados
    if pref.empty or "Vertices" not in pref.columns or "Taxa (%a.a.)" not in pref.columns:
        return None
    mats = pref["Vertices"].astype(float) / 252.0
    rates = _parse_rate_fraction(pref["Taxa (%a.a.)"])
    exponent = -np.clip(rates.to_numpy() * mats.to_numpy(), -50, 50)
    prices = np.exp(exponent)
    mask = np.isfinite(prices) & np.isfinite(mats) & (prices > 0) & (mats > 0)
    if not mask.any():
        return None
    return mats.to_numpy()[mask], prices[mask]


def _build_market_curve(data_path: Path, calib_maturities: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return maturities and market prices using either CurvaZero or flat DI curve."""

    curve = _curve_from_curva_zero(data_path)
    if curve is not None:
        mats, prices = curve
        mask = np.isfinite(mats) & np.isfinite(prices) & (mats > 0) & (prices > 0)
        mats, prices = mats[mask], prices[mask]
        if mats.size == 0:
            raise ValueError("Curva real nao possui maturidades/precos positivos.")
        return mats, prices

    rate_series = _get_rate_series(df, data_path).dropna()
    if rate_series.empty:
        raise ValueError(f"CSV precisa conter a coluna 'rate'. Colunas encontradas: {list(df.columns)}")

    mats = np.array([float(x.strip()) for x in calib_maturities.split(",") if x.strip()])
    if mats.size == 0 or np.any(mats <= 0):
        raise ValueError("Forneca ao menos uma maturidade.")
    last_rate = float(rate_series.tail(1))
    prices = np.exp(-np.clip(last_rate * mats, -50, 50))
    mask = np.isfinite(prices) & (prices > 0)
    if not mask.any():
        raise ValueError("Precos de mercado invalidos apos leitura do CSV.")
    return mats[mask], prices[mask]
    raise ValueError(f"CSV precisa conter a coluna 'rate'. Colunas encontradas: {list(df.columns)}")


SAMPLE_CASHFLOWS_JSON = json.dumps(
    {
        "assets": [
            {"time": 1.0, "amount": 1_000_000},
            {"time": 3.0, "amount": 1_200_000},
            {"time": 5.0, "amount": 1_500_000},
        ],
        "passives": [
            {"time": 0.5, "amount": 800_000},
            {"time": 2.0, "amount": 1_000_000},
            {"time": 4.0, "amount": 1_400_000},
        ],
    },
    indent=2,
)

MODEL_REGISTRY = {
    "CIR": {
        "schemes": ["em", "milstein"],
        "scheme_help": "EM = Euler-Maruyama, Milstein adiciona correção para reduzir viés perto de r=0.",
        "default_scheme_index": 1,
        "simulate": simulate_paths,
        "bond_price": bond_price_mc,
        "zero_coupon": zero_coupon_price,
        "mean": mean_short_rate,
        "variance": variance_short_rate,
        "calibrate": calibrate_zero_coupon_curve,
        "price_curve": price_curve,
        "supports_validation": True,
        "supports_convergence": True,
        "supports_calibration": True,
    },
    "Vasicek": {
        "schemes": ["em", "exact"],
        "scheme_help": "EM aproxima com passos discretos; Exact usa a solução fechada do OU.",
        "default_scheme_index": 1,
        "simulate": simulate_vasicek_paths,
        "bond_price": vasicek_bond_price_mc,
        "zero_coupon": vasicek_zero_coupon_price,
        "mean": vasicek_mean_short_rate,
        "variance": vasicek_variance_short_rate,
        "calibrate": calibrate_vasicek_curve,
        "price_curve": vasicek_price_curve,
        "supports_validation": False,
        "supports_convergence": False,
        "supports_calibration": True,
    },
    "Hull-White": {
        "schemes": ["em", "exact"],
        "scheme_help": "Modelo deslocado: use Exact para aproveitar a solução fechada do OU base.",
        "default_scheme_index": 1,
        "simulate": simulate_hull_white_paths,
        "bond_price": hull_white_bond_price_mc,
        "zero_coupon": hull_white_zero_coupon_price,
        "mean": hull_white_mean_short_rate,
        "variance": hull_white_variance_short_rate,
        "calibrate": None,
        "price_curve": hull_white_price_curve,
        "supports_validation": False,
        "supports_convergence": False,
        "supports_calibration": False,
    },
}

st.set_page_config(page_title="CIR Dashboard", layout="wide")
st.title("CIR Dashboard - Layout base")
st.markdown(
    """
    Este painel foi pensado para leituras rápidas, mesmo para quem não domina o modelo CIR.
    Em cada aba você encontrará descrições do que está sendo exibido e como interpretar os números.
    Use a coluna à esquerda para alterar o preset (conjunto de parâmetros), o esquema numérico e
    habilitar análises extras.
    """
)

sidebar, main = st.columns([1, 3])

with sidebar:
    model_name = st.selectbox(
        "Modelo",
        list(MODEL_REGISTRY),
        index=0,
        help="Altere entre CIR (raiz quadrada), Vasicek (OU) ou Hull-White (deslocado).",
    )
    model_cfg = MODEL_REGISTRY[model_name]
    compare_selection = st.multiselect(
        "Modelos no comparativo",
        list(MODEL_REGISTRY),
        default=list(MODEL_REGISTRY),
        help="Selecione dois ou mais modelos para sobrepor médias, métricas e curvas.",
    )
    if not compare_selection:
        compare_selection = [model_name]
    compare_models_ordered = list(dict.fromkeys(compare_selection))
    if model_name not in compare_models_ordered:
        compare_models_ordered.insert(0, model_name)
    scheme = st.selectbox(
        "Esquema",
        model_cfg["schemes"],
        index=model_cfg["default_scheme_index"],
        help=model_cfg["scheme_help"],
    )
    T = st.slider(
        "Horizonte T",
        min_value=0.5,
        max_value=5.0,
        value=2.5,
        step=0.5,
        help="Tempo final das simulações em anos.",
    )
    steps_per_year = st.slider(
        "Passos por ano",
        min_value=52,
        max_value=365,
        value=252,
        step=26,
        help="Quanto maior, mais fina a malha temporal e mais precisa a simulação.",
    )
    n_paths = st.slider(
        "Numero de caminhos",
        min_value=5,
        max_value=50,
        value=5,
        step=5,
        help="Quantidade de trajetórias exibidas/geradas. Valores maiores usam mais CPU.",
    )
    seed = st.number_input(
        "Seed",
        value=42,
        step=1,
        help="Usado para reter reprodutibilidade. Mudar o seed muda os sorteios.",
    )
    use_real_data = st.checkbox("Usar dados reais (curva + SELIC)", value=True)
    real_curve_kind = "prefixados"
    real_curve_file = None
    real_selic_file = None
    sync_volatility = False
    if use_real_data:
        real_curve_file = st.text_input("Arquivo Curva Zero", value="data/real_data/CurvaZero_17112025.csv")
        real_curve_kind = st.selectbox("Tipo de curva", ["prefixados", "ipca"])
        real_selic_file = st.text_input("Arquivo SELIC diária", value="data/real_data/taxa_selic_apurada.csv")
        sync_volatility = st.checkbox(
            "Sincronizar volatilidade (Vasicek/Hull-White = CIR)",
            value=False,
            help="Forca sigma dos modelos Vasicek/Hull-White a seguir o sigma calibrado do CIR.",
        )
    else:
        st.error("Ative 'Usar dados reais' e informe os CSVs para continuar.")
        st.stop()

    st.markdown("---")
    st.subheader("Ofertas adicionais")
    show_term_structure = st.checkbox(
        "Calcular term structure", value=False, help="Gera tabela + gráfico com preços/yields."
    )
    if model_cfg["supports_convergence"]:
        show_convergence = st.checkbox(
            "Calcular convergencia forte (EM)", value=False, help="Calcula RMSE acoplado para o EM."
        )
    else:
        show_convergence = False
        st.caption("Convergência forte disponível apenas para o CIR.")
    if model_cfg["supports_validation"]:
        show_validation = st.checkbox(
            "Comparar com formulas analíticas",
            value=False,
            help="Requer um número razoável de trajetórias para que a comparação seja significativa.",
        )
    else:
        show_validation = False
        st.caption("Validação analítica disponível apenas para o CIR.")
    if model_cfg.get("supports_calibration", True):
        show_calibration = st.checkbox(
            "Calibrar com dados DI (CSV)",
            value=False,
            help="Utilize um CSV gerado pelos scripts para ajustar os parâmetros aos dados reais.",
        )
    else:
        show_calibration = False
        st.caption("Calibração automática ainda não disponível para este modelo.")
    st.markdown("---")
    st.subheader("Casos de uso")
    show_swaption_demo = st.checkbox("Simular swaption (MC)", value=False)
    swaption_settings = None
    if show_swaption_demo:
        swaption_kind = st.selectbox("Tipo de swaption", ["payer", "receiver"])
        swaption_exercise = st.number_input("Exercício (anos)", value=2.0, min_value=0.5, max_value=10.0, step=0.5)
        swaption_tenor = st.number_input("Tenor do swap (anos)", value=3.0, min_value=0.5, max_value=10.0, step=0.5)
        swaption_freq = st.selectbox("Frequência dos cupons (por ano)", [1, 2, 4], index=1)
        swaption_strike = st.number_input("Strike (taxa fixa)", value=0.04, format="%.3f")
        swaption_paths = st.slider("Caminhos para swaption", min_value=500, max_value=5000, value=1000, step=500)
        swaption_steps = st.slider("Passos/ano (swaption)", min_value=52, max_value=365, value=104, step=26)
        swaption_settings = {
            "kind": swaption_kind,
            "exercise": float(swaption_exercise),
            "tenor": float(swaption_tenor),
            "freq": int(swaption_freq),
            "strike": float(swaption_strike),
            "paths": int(swaption_paths),
            "steps_per_year": int(swaption_steps),
        }
    show_alm_demo = st.checkbox("Simular cenários ALM", value=False)
    alm_settings = None
    if show_alm_demo:
        alm_text = st.text_area("Fluxos (JSON)", value=SAMPLE_CASHFLOWS_JSON, height=200)
        alm_paths = st.slider("Caminhos para curva média", min_value=200, max_value=5000, value=600, step=200)
        alm_steps = st.slider("Passos/ano (ALM)", min_value=52, max_value=365, value=104, step=26)
        alm_seed = st.number_input("Seed (ALM)", value=99, step=1)
        alm_settings = {
            "json": alm_text,
            "paths": int(alm_paths),
            "steps_per_year": int(alm_steps),
            "seed": int(alm_seed),
        }
    else:
        show_calibration = False
        st.caption("Calibração automática ainda não disponível para este modelo.")
    term_paths = None
    if show_term_structure:
        term_paths = st.slider(
            "Caminhos para term structure",
            min_value=500,
            max_value=5000,
            value=1000,
            step=500,
            help="Mais caminhos = menos ruído nos preços/yields simulados. Valores maiores podem ser lentos.",
        )

    conv_paths = conv_steps = None
    if show_convergence:
        conv_paths = st.number_input(
            "Caminhos para convergencia",
            value=2000,
            min_value=500,
            max_value=200000,
            step=500,
            help="Sugere-se valores grandes para reduzir o erro forte.",
        )
        conv_steps = st.multiselect(
            "Steps (use valores que dividem a malha mais fina)",
            options=[52, 104, 208, 416, 832],
            default=[52, 104, 208, 416, 832],
            help="Quanto mais pontos, melhor a regressão log–log do RMSE.",
        )

    val_maturities = val_paths = val_steps = None
    if show_validation:
        val_maturities = st.multiselect(
            "Maturidades para comparar (analitico vs MC)",
            options=[0.25, 0.5, 1.0, 2.0, 5.0],
            default=[0.5, 1.0, 2.0],
        )
        val_paths = st.number_input(
            "Caminhos para validacao", value=2000, min_value=1000, max_value=200000, step=1000
        )
        val_steps = st.slider(
            "Steps/ano para validacao", min_value=52, max_value=365, value=126, step=26
        )

    data_file = None
    calib_maturities = "0.25,0.5,1.0,2.0,3.0,5.0"
    if show_calibration:
        data_file = st.text_input(
            "Arquivo CSV da curva DI",
            value="data/real_data/CurvaZero_17112025.csv",
            help="Formato esperado: colunas 'date' e 'rate' (fração). Use scripts/fetch_di_curve.py.",
        )
        calib_maturities = st.text_input(
            "Maturidades (anos, separadas por vírgula)",
            value="0.25,0.5,1.0,2.0,3.0,5.0",
            help="Pontos nos quais a curva será comparada. Você pode usar as mesmas durações das OTN/DI.",
        )

st.caption(
    "Use a coluna lateral para escolher o modelo/esquema, carregar dados reais e habilitar análises adicionais. "
    "Cada aba abaixo descreve o que está sendo exibido para facilitar a interpretação."
)

params_by_model: dict[str, object] = {}
selic_df = None
curve_components = None
real_curve_date = None
try:
    curve_components = load_curve_components_cached(real_curve_file)
    params_by_model, real_curve_date = calibrate_models_cached(
        real_curve_file, real_curve_kind, sync_volatility=sync_volatility
    )
except Exception as exc:
    st.error(f"Falha ao calibrar a partir da curva real: {exc}")
    st.stop()
try:
    selic_df, latest_rate = load_selic_cached(real_selic_file)
    params_by_model = {name: _replace_r0(p, latest_rate) for name, p in params_by_model.items()}
except Exception as exc:
    st.error(f"Não foi possível carregar a SELIC: {exc}")
params = params_by_model.get(model_name)
if params is None:
    st.error(f"Parâmetros indisponíveis para {model_name}.")
    st.stop()
n_steps = int(T * steps_per_year)
paths = None

try:
    simulate_fn = model_cfg["simulate"]
    t, paths = simulate_fn(
        scheme=scheme,
        params=params,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=int(seed),
    )
except Exception as exc:
    st.error(f"Erro ao simular caminhos: {exc}")

comparison_results: dict[str, dict[str, object]] = {}
if paths is not None:
    comparison_results[model_name] = {
        "t": t,
        "paths": paths,
        "params": params,
        "cfg": model_cfg,
    }
    for idx, cmp_model in enumerate(compare_models_ordered):
        if cmp_model == model_name or cmp_model in comparison_results:
            continue
        cfg = MODEL_REGISTRY[cmp_model]
        cmp_params = params_by_model.get(cmp_model)
        if cmp_params is None:
            continue
        cmp_seed = int(seed) + (idx + 1) * 97
        model_scheme = scheme if scheme in cfg["schemes"] else cfg["schemes"][cfg["default_scheme_index"]]
        try:
            t_cmp, paths_cmp = cfg["simulate"](
                scheme=model_scheme,
                params=cmp_params,
                T=T,
                n_steps=n_steps,
                n_paths=n_paths,
                seed=cmp_seed,
            )
            comparison_results[cmp_model] = {
                "t": t_cmp,
                "paths": paths_cmp,
                "params": cmp_params,
                "cfg": cfg,
            }
        except Exception as exc:  # pragma: no cover - UI feedback
            st.warning(f"Não foi possível simular {cmp_model}: {exc}")

    has_comparison = len(comparison_results) > 1
    tabs = ["Trajetorias", "Distribuicao terminal", "B(0,T) e yield", "Convergencia"]
    if has_comparison:
        tabs.append("Comparativo modelos")
    if use_real_data and curve_components is not None:
        tabs.append("Curva real")
    if swaption_settings:
        tabs.append("Swaption MC")
    if alm_settings:
        tabs.append("Cenarios ALM")
    if show_validation:
        tabs.append("Validacao analitica")
    if show_calibration:
        tabs.append("Calibracao mercado")
    tab_objects = main.tabs(tabs)
    tab_lookup = dict(zip(tabs, tab_objects))
    tab_paths = tab_lookup["Trajetorias"]
    tab_hist = tab_lookup["Distribuicao terminal"]
    tab_term = tab_lookup["B(0,T) e yield"]
    tab_conv = tab_lookup["Convergencia"]
    tab_compare = tab_lookup.get("Comparativo modelos")
    tab_curve_real = tab_lookup.get("Curva real")
    tab_swaption = tab_lookup.get("Swaption MC")
    tab_alm = tab_lookup.get("Cenarios ALM")
    tab_val = tab_lookup.get("Validacao analitica")
    tab_calib = tab_lookup.get("Calibracao mercado")

    with tab_paths:
        st.markdown("### Trajetórias simuladas")
        st.markdown(
            f"""
            Cada linha é uma realização Monte Carlo do processo **{model_name}** calibrado na curva real informada
            (DI/SELIC). Você está vendo o short rate $r(t)$ - a taxa de juros instantânea, como se fossem os
            "juros agora" no limite de $dt \\to 0$. Todas as trajetórias partem de $r_0$ calibrado, sofrem choques
            aleatórios e tendem a voltar ao nível de longo prazo $\\theta$ (reversão à média).
            """
        )
        st.markdown(
            """
            - $\\kappa$: velocidade de retorno ao platô $\\theta$; valores altos fecham o leque mais rápido.
            - $\\theta$: patamar de equilíbrio; espere as linhas "embolando" em torno dele.
            - $\\sigma$: volatilidade; controla o quanto os cenários se afastam/abrem.
            - Cada cor é um cenário possível (Monte Carlo) para a evolução da taxa curta $r(t)$.
            """
        )
        col_params = st.columns(4)
        col_params[0].metric("r₀ (partida)", f"{params.r0:.4f}")
        col_params[1].metric("θ (longo prazo)", f"{params.theta:.4f}")
        col_params[2].metric("κ (mean reversion)", f"{params.kappa:.4f}")
        col_params[3].metric("σ (vol)", f"{params.sigma:.4f}")
        if isinstance(params, CIRParams):
            feller_lhs = 2 * params.kappa * params.theta
            feller_rhs = params.sigma**2
            status = "ok (mantém positividade)" if feller_lhs > feller_rhs else "violada (pode tocar zero)"
            st.caption(
                f"Condição de Feller $2\\kappa\\theta > \\sigma^2$: $2\\kappa\\theta={feller_lhs:.4f}$ vs "
                f"$\\sigma^2={feller_rhs:.4f}$ -> {status}."
            )
        st.caption(
            'Leitura rápida: short rate é a "juros agora". O modelo puxa as trajetórias de volta para $\\theta$; '
            "$\\sigma$ abre o leque de cenários. Use o seed para reproduzir ou alterar os sorteios."
        )
        df_paths = pd.DataFrame(
            paths.T,
            index=np.round(t, 4),
            columns=[f"path_{i+1}" for i in range(paths.shape[0])],
        )
        st.line_chart(df_paths)
        col_stats = st.columns(3)
        terminals = paths[:, -1]
        col_stats[0].metric("Media terminal", f"{terminals.mean():.4f}")
        col_stats[1].metric("Desvio padrao", f"{terminals.std(ddof=1):.4f}")
        col_stats[2].metric("Min / Max", f"{terminals.min():.4f} / {terminals.max():.4f}")

    with tab_hist:
        st.markdown("### Distribuição terminal")
        st.markdown(
            """
            Distribuição de $r_T$ ao final de $T$ anos. Cada barra conta quantas simulações terminaram naquele
            intervalo de taxa.
            - Pico perto de $\\theta$: efeito da reversão à média ($\\kappa$).
            - Largura reflete risco: mais $\\sigma$ ou $T$ -> histograma mais espalhado.
            - Massa perto de zero só aparece se $2\\kappa\\theta \\leq \\sigma^2$ (violação da Condição de Feller).
            - Esperança analítica: $\\mathbb{E}[r_T] = \\theta + (r_0-\\theta) e^{-\\kappa T}$.
            """
        )
        terminals = paths[:, -1]
        mean_term = terminals.mean()
        median_term = float(np.median(terminals))
        p10, p90 = np.percentile(terminals, [10, 90])
        col_hist = st.columns(4)
        col_hist[0].metric("Média (MC)", f"{mean_term:.4f}")
        col_hist[1].metric("Mediana", f"{median_term:.4f}")
        col_hist[2].metric("Desvio padrão", f"{terminals.std(ddof=1):.4f}")
        col_hist[3].metric("P10 / P90", f"{p10:.4f} / {p90:.4f}")
        bins = min(40, max(5, paths.shape[1] // 4))
        fig_hist, ax_hist = plt.subplots(figsize=(7, 3))
        ax_hist.hist(terminals, bins=bins, alpha=0.8, color="#76b7fb", edgecolor="black")
        ax_hist.set(xlabel="Taxa final $r_T$", ylabel="Contagem", title="Histograma dos terminais")
        ax_hist.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
        st.pyplot(fig_hist)
        st.caption(
            "Como ler: barra alta = maior probabilidade de terminar naquele nível. Se o leque estiver muito aberto, "
            "ajuste $\\sigma$ ou reduza $T$; se o pico não fica perto de $\\theta$, veja os parâmetros calibrados."
        )

    with tab_term:
        st.markdown("### Estrutura a termo simulada")
        st.markdown(
            f"""
            Preços zero-coupon $B(0,T)$ e yields $y_T = -\\ln B(0,T)/T$ simulados para o modelo **{model_name}**.
            - $B(0,T)$: quanto vale hoje R\\$1 a ser recebido em $T$ anos (desconto estocástico).
            - $y_T$: taxa implícita do preço. Linha azul (preço) normalmente cai com $T$; linha vermelha (yield) mostra a curva de juros.
            - Ajuste Monte Carlo: mais caminhos e passos/ano reduzem ruído; recalcule sempre que mudar parâmetros.
            """
        )
        if show_term_structure and term_paths is not None:
            maturities = np.linspace(0.25, 10.0, 40)
            run_term = st.button("Calcular term structure", key="btn_term_structure")
            if run_term:
                ts_df = _term_structure_cached(
                    maturities=maturities,
                    model_name=model_name,
                    params=params,
                    steps_per_year=steps_per_year,
                    scheme=scheme,
                    n_paths=term_paths,
                    base_seed=int(seed) * 100,
                )
                st.dataframe(ts_df, use_container_width=True)

                fig, ax_price = plt.subplots(figsize=(7, 4))
                ax_yield = ax_price.twinx()
                ax_price.plot(ts_df["T"], ts_df["price"], "o-", color="#1f77b4", label=r"$B(0,T)$")
                ax_yield.plot(ts_df["T"], ts_df["zero_rate"], "s--", color="#d62728", label="Yield $y_T$")
                ax_price.set(xlabel="Maturidade T (anos)", ylabel=r"Preço $B(0,T)$", title="Curva zero-coupon")
                ax_yield.set(ylabel=r"Yield $y_T$")
                ax_price.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
                lines = ax_price.get_lines() + ax_yield.get_lines()
                labels = [line.get_label() for line in lines]
                ax_price.legend(lines, labels, loc="best")
                st.pyplot(fig)
                st.caption(
                    "Leitura rápida: pontos azuis são preços de face 1 descontados; quanto maior T, menor o preço. "
                    "Pontos vermelhos são yields extraídos desses preços. Se a curva estiver ruidosa, aumente "
                    "caminhos ou passos/ano."
                )
            else:
                st.info("Clique em 'Calcular term structure' para gerar a curva Monte Carlo.")
        else:
            st.info("Ative a opcao 'Calcular term structure' no menu lateral para ver esta secao.")

    with tab_conv:
        st.markdown("### Estudo de convergência forte")
        st.write(
            "Comparação da malha refinada com malhas mais grosseiras usando RMSE forte. "
            "O gráfico log–log ajuda a validar a ordem do esquema."
        )
        if not model_cfg["supports_convergence"]:
            st.info("Estudo de convergência ainda não disponível para este modelo.")
        elif show_convergence and conv_paths and conv_steps:
            try:
                result = strong_order_convergence(
                    scheme="em",
                    params=params,
                    T=1.0,
                    n_paths=int(conv_paths),
                    base_steps_list=conv_steps,
                    seed=int(seed) * 10,
                )
                st.metric("Inclinacao estimada", f"{result.slope:.3f}")
                conv_df = pd.DataFrame({"dt": result.dts_fit, "rmse": result.errors_fit})
                st.dataframe(conv_df)

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.loglog(result.dts_fit, result.errors_fit, "o-", label="Erro observado")
                ax.loglog(
                    result.dts_fit,
                    np.exp(result.intercept) * result.dts_fit**result.slope,
                    "--",
                    label=f"slope={result.slope:.2f}",
                )
                ax.set(xlabel="Delta t", ylabel="RMSE", title="Convergencia forte (EM)")
                ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.3)
                ax.legend()
                st.pyplot(fig)

                csv_data = conv_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Baixar CSV de convergencia",
                    data=csv_data,
                    file_name="convergencia_em.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Erro ao calcular convergencia: {exc}")
        else:
            st.info("Ative a opcao 'Calcular convergencia forte (EM)' e selecione steps validos para ver esta secao.")
    if has_comparison and tab_compare is not None:
        with tab_compare:
            st.markdown("### Comparativo entre modelos")
            st.write(
                "As curvas abaixo usam a mesma curva real e horizonte. A primeira figura mostra a média temporal "
                "de cada modelo, enquanto a segunda compara as curvas zero-coupon analíticas."
            )
            fig_cmp, ax_cmp = plt.subplots(figsize=(7, 4))
            for name, result in comparison_results.items():
                mean_path = result["paths"].mean(axis=0)
                ax_cmp.plot(result["t"], mean_path, label=name)
            ax_cmp.set(xlabel="Tempo", ylabel="Média das trajetórias", title="Média das trajetórias simuladas")
            ax_cmp.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
            ax_cmp.legend()
            st.pyplot(fig_cmp)

            metrics_rows = []
            for name, result in comparison_results.items():
                terminals = result["paths"][:, -1]
                metrics_rows.append(
                    {
                        "Modelo": name,
                        "Media terminal": terminals.mean(),
                        "Desvio padrao": terminals.std(ddof=1),
                        "Min": terminals.min(),
                        "Max": terminals.max(),
                    }
                )
            metrics_df = pd.DataFrame(metrics_rows).set_index("Modelo")
            st.subheader("Métricas no tempo final")
            st.dataframe(metrics_df.style.format("{:.4f}"))

            maturities = np.linspace(0.25, 5.0, 20)
            fig_curve, ax_curve = plt.subplots(figsize=(7, 4))
            for name, result in comparison_results.items():
                zero_fn = result["cfg"]["zero_coupon"]
                price_curve_vals = np.array([zero_fn(result["params"], float(m)) for m in maturities])
                yields = -np.log(np.maximum(price_curve_vals, 1e-12)) / maturities
                ax_curve.plot(maturities, yields, label=name)
            ax_curve.set(xlabel="Maturidade", ylabel="Yield", title="Curvas zero-coupon (analíticas)")
            ax_curve.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
            ax_curve.legend()
            st.pyplot(fig_curve)
    if tab_curve_real is not None and curve_components is not None:
        with tab_curve_real:
            st.markdown("### Curva real importada")
            st.write(f"Arquivo: `{real_curve_file}` | Data de referência: {curve_components.ref_date}")
            ettj_df = curve_components.ettj.copy()
            if not ettj_df.empty:
                ettj_df["Anos"] = ettj_df["Vertices"] / 252.0
                st.subheader("ETTJ (IPCA vs Prefixado)")
                st.dataframe(ettj_df)
                fig_real, ax_real = plt.subplots(figsize=(7, 4))
                ax_real.plot(ettj_df["Anos"], ettj_df["ETTJ IPCA"], label="IPCA")
                ax_real.plot(ettj_df["Anos"], ettj_df["ETTJ PREF"], label="Prefixado")
                ax_real.set(xlabel="Anos", ylabel="Taxa (% a.a.)", title="ETTJ real")
                ax_real.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
                ax_real.legend()
                st.pyplot(fig_real)
            prefix_df = curve_components.prefixados.copy()
            if not prefix_df.empty:
                prefix_df = prefix_df.rename(columns={"Taxa (%a.a.)": "taxa_pct"})
                prefix_df["Anos"] = prefix_df["Vertices"] / 252.0
                st.subheader("Curva prefixada (Circular 3.361)")
                chart_prefix = (
                    alt.Chart(prefix_df.dropna(subset=["Anos", "taxa_pct"]))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Anos:Q", title="Anos"),
                        y=alt.Y("taxa_pct:Q", title="Taxa (% a.a.)"),
                        tooltip=["Anos", "taxa_pct"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_prefix, use_container_width=True)
            residuals_df = curve_components.residuals.copy()
            if not residuals_df.empty:
                st.subheader("Erro título a título")
                residuals_df = residuals_df.rename(columns={"Erro (%a.a.)": "erro_pct"})
                residuals_df["erro_pct"] = (
                    residuals_df["erro_pct"]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )
                residuals_df["Titulo/Venc"] = residuals_df["Titulo"] + " " + residuals_df["Vencimento"]
                st.dataframe(residuals_df)
                bar_resid = (
                    alt.Chart(residuals_df.dropna(subset=["erro_pct"]))
                    .mark_bar()
                    .encode(
                        x=alt.X("Titulo/Venc:N", sort=None, title="Título/Vencimento"),
                        y=alt.Y("erro_pct:Q", title="Erro (% a.a.)"),
                        tooltip=["Titulo", "Vencimento", "erro_pct"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(bar_resid, use_container_width=True)
            if selic_df is not None:
                st.subheader("Últimos valores da SELIC")
                st.line_chart(selic_df.set_index("date")["rate_annual"])
    if swaption_settings and tab_swaption is not None:
        with tab_swaption:
            st.markdown("### Precificação de swaption via Monte Carlo")
            st.caption("Os resultados utilizam os mesmos parâmetros calibrados a partir da curva real.")
            run_swap = st.button("Calcular swaption (MC)", key="btn_swaption")
            if run_swap:
                try:
                    df_swap = _swaption_cached(
                        compare_models_ordered=compare_models_ordered,
                        params_by_model=params_by_model,
                        swaption_settings=swaption_settings,
                        seed=seed,
                        scheme=scheme,
                    )
                    if not df_swap.empty:
                        st.dataframe(df_swap.style.format({"Preco": "{:.4f}", "Erro padrao": "{:.4f}"}))
                        st.caption("Visualização do preço estimado por modelo.")
                        st.bar_chart(df_swap["Preco"])
                        st.write(
                            f"Swaption {swaption_settings['kind']} | Exercício {swaption_settings['exercise']} | "
                            f"Tenor {swaption_settings['tenor']} | Strike {swaption_settings['strike']:.3%}"
                        )
                    else:
                        st.info("Ative ao menos um modelo válido para visualizar o preço da swaption.")
                except Exception as exc:
                    st.error(f"Falha ao precificar swaption: {exc}")
            else:
                st.info("Clique em 'Calcular swaption (MC)' quando quiser rodar o preço.")
    if alm_settings and tab_alm is not None:
        with tab_alm:
            st.markdown("### Cenários ALM")
            st.write("Aplica choques determinísticos (parallel, steepener, flattener, ramp) sobre uma curva média simulada.")
            run_alm = st.button("Calcular cenários ALM", key="btn_alm")
            if run_alm:
                try:
                    data_json = json.loads(alm_settings["json"])
                    assets_df = _build_cashflow_df(data_json.get("assets", []), "assets")
                    passives_df = _build_cashflow_df(data_json.get("passives", []), "passives")
                    params_alm = params_by_model.get(model_name)
                    if params_alm is None:
                        raise ValueError("Parâmetros indisponíveis para o modelo selecionado.")
                    t_curve, scenario_curves, df_alm, net_base = _alm_cached(
                        model_name=model_name,
                        params=params_alm,
                        alm_settings=alm_settings,
                        scheme=scheme,
                        assets_df=assets_df,
                        passives_df=passives_df,
                    )
                    st.dataframe(
                        df_alm.style.format(
                            {
                                "PV ativos": "{:,.0f}",
                                "PV passivos": "{:,.0f}",
                                "PV liquido": "{:,.0f}",
                                "Duration ativos": "{:.2f}",
                                "Duration passivos": "{:.2f}",
                            }
                        )
                    )
                    st.caption("PV líquido por cenário (positivo = superávit dos ativos).")
                    st.bar_chart(df_alm["PV liquido"])
                    fig_curve, ax_curve = plt.subplots(figsize=(7, 4))
                    for sc_name, curve in scenario_curves.items():
                        ax_curve.plot(t_curve, curve, label=sc_name)
                    ax_curve.set(xlabel="Tempo", ylabel="Taxa média", title="Curvas simuladas + choques")
                    ax_curve.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
                    ax_curve.legend(loc="upper right", fontsize="small")
                    st.pyplot(fig_curve)
                    st.caption("Fluxo líquido (ativos - passivos) por bucket de tempo.")
                    st.bar_chart(net_base)
                except Exception as exc:
                    st.error(f"Erro ao calcular cenários ALM: {exc}")
            else:
                st.info("Clique em 'Calcular cenários ALM' para gerar os choques.")
    if show_validation and tab_val is not None and val_maturities:
        with tab_val:
            st.markdown("### Validação analítica")
            st.write(
                "Tabela e gráficos comparando Monte Carlo com fórmulas fechadas do modelo. "
                "Útil para verificar viés numérico e erro vs. passo."
            )
            run_val = st.button("Rodar validação agora", key="btn_validation")
            if run_val:
                price_df, err_df, moment_result = _validation_cached(
                    params=params,
                    val_maturities=val_maturities,
                    val_paths=val_paths,
                    val_steps=val_steps,
                    seed=seed,
                    scheme=scheme,
                )
                st.subheader("Comparacao zero-coupon (MC vs analitico)")
                st.dataframe(price_df, use_container_width=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(price_df["T"], price_df["abs_error"], "o-", label="|Erro|")
                ax.set(xlabel="Maturidade", ylabel="Erro absoluto", title="Erro de preco vs maturidade")
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
                st.pyplot(fig)

                st.subheader("Erro vs passos (fixando maturidade)")
                st.dataframe(err_df, use_container_width=True)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.loglog(err_df["dt"], err_df["abs_error"], "s-", label="|Erro|")
                ax2.set(xlabel="dt", ylabel="Erro absoluto", title="Erro vs passo (log-log)")
                ax2.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.3)
                st.pyplot(fig2)

                st.subheader("Momentos vs analitico")
                col_m = st.columns(2)
                col_m[0].metric("Media (MC vs analitico)", f"{moment_result['mean_mc']:.4f}", f"ref {moment_result['mean_analytic']:.4f}")
                col_m[1].metric("Var (MC vs analitico)", f"{moment_result['var_mc']:.4f}", f"ref {moment_result['var_analytic']:.4f}")
            else:
                st.info("Clique em 'Rodar validação agora' quando quiser calcular os comparativos.")

    if show_calibration and tab_calib is not None and data_file:
        with tab_calib:
            st.markdown("### Calibração com curva DI")
            st.write(
                "Carrega o CSV com a curva DI (gerado por `scripts/fetch_di_curve.py`), "
                "aplica mínimos quadrados nos preços zero e plota o ajuste."
            )
            try:
                data_path = Path(data_file)
                if not data_path.exists():
                    raise FileNotFoundError(f"Arquivo {data_path} não encontrado.")
                df = _read_csv_fallback(data_path)
                mats, market_prices = _build_market_curve(data_path, calib_maturities, df)
                initial_params = params_by_model.get("CIR")
                if initial_params is None:
                    raise ValueError("Parâmetros do CIR não disponíveis.")
                calib_fn = model_cfg.get("calibrate")
                price_curve_fn = model_cfg.get("price_curve")
                if calib_fn is None or price_curve_fn is None:
                    raise RuntimeError("Calibração indisponível para este modelo.")
                calib_result = calib_fn(mats, market_prices, initial_params)
                if calib_result.success:
                    st.success(f"Calibração ({model_name}) concluída.")
                else:
                    st.warning(f"Otimizador retornou status: {calib_result.message}")
                param_df = pd.DataFrame(
                    {
                        "kappa": [calib_result.params.kappa],
                        "theta": [calib_result.params.theta],
                        "sigma": [calib_result.params.sigma],
                        "r0": [calib_result.params.r0],
                    }
                ).T.rename(columns={0: "valor"})
                st.subheader("Parâmetros calibrados")
                st.table(param_df)
                fitted = price_curve_fn(calib_result.params, mats)
                calib_df = pd.DataFrame(
                    {
                        "T": mats,
                        "market_price": market_prices,
                        "fitted_price": fitted,
                        "abs_error": np.abs(fitted - market_prices),
                    }
                )
                st.dataframe(calib_df, use_container_width=True)
                fig_calib, ax_calib = plt.subplots(figsize=(7, 4))
                ax_calib.plot(calib_df["T"], calib_df["market_price"], "o-", label="Mercado")
                ax_calib.plot(
                    calib_df["T"],
                    calib_df["fitted_price"],
                    "s--",
                    label=f"{model_name} calibrado",
                )
                ax_calib.set(xlabel="Maturidade", ylabel="Preço", title="Curva calibrada")
                ax_calib.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
                ax_calib.legend()
                st.pyplot(fig_calib)
            except Exception as exc:
                st.error(f"Erro na calibração: {exc}")
else:
    main.warning("Aguardando configuracao valida para simular trajetorias.")
