import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.models import (
    HullWhiteParams,
    calibrate_vasicek_curve,
    get_hull_white_preset,
    get_vasicek_preset,
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
from cir.calibration import calibrate_zero_coupon_curve, price_curve
from cir.convergence import strong_order_convergence
from cir.params import get_params_preset
from cir.simulate import simulate_paths
from cir.validation import (
    compare_moments,
    compare_zero_coupon_prices,
    zero_coupon_error_by_steps,
)
from examples.utils.scenario_builders import DEFAULT_SCENARIOS
from examples.utils.swap_helpers import SwapSchedule, swaption_payoff


def _integrated_discount(path: np.ndarray, t: np.ndarray, T: float) -> float:
    mask = t <= T
    if not np.any(mask):
        return 1.0
    integral = np.trapz(path[mask], t[mask])
    return float(math.exp(-integral))


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
    total_T = float(schedule.payment_times[-1])
    n_steps = max(1, int(total_T * steps_per_year))
    scheme = preferred_scheme if preferred_scheme in model_cfg["schemes"] else model_cfg["schemes"][model_cfg["default_scheme_index"]]
    t, paths = model_cfg["simulate"](
        scheme=scheme,
        params=params,
        T=total_T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    payoffs = []
    for path in paths:
        discounts = np.array([_integrated_discount(path, t, float(Ti)) for Ti in schedule.payment_times])
        payoff = swaption_payoff(discounts, schedule, strike, kind)
        disc_ex = _integrated_discount(path, t, exercise)
        payoffs.append(payoff * disc_ex)
    payoffs = np.asarray(payoffs)
    price = float(payoffs.mean())
    stderr = float(payoffs.std(ddof=1) / math.sqrt(len(payoffs))) if len(payoffs) > 1 else 0.0
    return price, stderr


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
        "get_params": get_params_preset,
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
        "get_params": get_vasicek_preset,
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
        "get_params": get_hull_white_preset,
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
        default=[model_name],
        help="Selecione dois ou mais modelos para sobrepor médias, métricas e curvas.",
    )
    if not compare_selection:
        compare_selection = [model_name]
    compare_models_ordered = list(dict.fromkeys(compare_selection))
    if model_name not in compare_models_ordered:
        compare_models_ordered.insert(0, model_name)
    preset = st.selectbox(
        "Preset",
        ["baseline", "slow-revert", "fast-revert"],
        index=0,
        help="Selecione um conjunto de parâmetros iniciais para o modelo escolhido.",
    )
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
        value=5.0,
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
        value=10,
        step=5,
        help="Quantidade de trajetórias exibidas/geradas. Valores maiores usam mais CPU.",
    )
    seed = st.number_input(
        "Seed",
        value=42,
        step=1,
        help="Usado para reter reprodutibilidade. Mudar o seed muda os sorteios.",
    )

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
        swaption_paths = st.slider("Caminhos para swaption", min_value=500, max_value=5000, value=2000, step=500)
        swaption_steps = st.slider("Passos/ano (swaption)", min_value=52, max_value=365, value=126, step=26)
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
        alm_paths = st.slider("Caminhos para curva média", min_value=200, max_value=5000, value=1000, step=200)
        alm_steps = st.slider("Passos/ano (ALM)", min_value=52, max_value=365, value=126, step=26)
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
            value=2000,
            step=500,
            help="Mais caminhos = menos ruído nos preços/yields simulados.",
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
            "Caminhos para validacao", value=5000, min_value=1000, max_value=200000, step=1000
        )
        val_steps = st.slider(
            "Steps/ano para validacao", min_value=52, max_value=365, value=126, step=26
        )

    data_file = None
    calib_initial = "baseline"
    calib_maturities = "0.25,0.5,1.0,2.0,3.0,5.0"
    if show_calibration:
        data_file = st.text_input(
            "Arquivo CSV da curva DI",
            value="data/raw_di_curve.csv",
            help="Formato esperado: colunas 'date' e 'rate' (fração). Use scripts/fetch_di_curve.py.",
        )
        calib_initial = st.selectbox(
            "Preset inicial (calibração)",
            ["baseline", "slow-revert", "fast-revert"],
            index=0,
            help="Serve como chute inicial; o algoritmo move-se a partir daqui.",
        )
        calib_maturities = st.text_input(
            "Maturidades (anos, separadas por vírgula)",
            value="0.25,0.5,1.0,2.0,3.0,5.0",
            help="Pontos nos quais a curva será comparada. Você pode usar as mesmas durações das OTN/DI.",
        )

st.caption(
    "Use a coluna lateral para escolher preset/esquema e habilitar análises adicionais. "
    "Cada aba abaixo descreve o que está sendo exibido para facilitar a interpretação."
)

params = model_cfg["get_params"](preset)
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
        cmp_params = cfg["get_params"](preset)
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
    tab_swaption = tab_lookup.get("Swaption MC")
    tab_alm = tab_lookup.get("Cenarios ALM")
    tab_val = tab_lookup.get("Validacao analitica")
    tab_calib = tab_lookup.get("Calibracao mercado")

    with tab_paths:
        st.markdown("### Trajetórias simuladas")
        st.write(
            f"Cada linha representa uma realização do processo {model_name} com o preset selecionado. "
            "A tabela abaixo exibe a média, desvio e mínimos/máximos no tempo final."
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
        st.write(
            "Histograma dos valores de $r_T$ nas simulações. Observe como a massa se concentra "
            "ao redor de $\\theta$ (nível de longo prazo). Para exportar os dados use a CLI."
        )
        terminals = paths[:, -1]
        bins = min(40, max(5, paths.shape[1] // 4))
        fig_hist, ax_hist = plt.subplots(figsize=(7, 3))
        ax_hist.hist(terminals, bins=bins, alpha=0.8, color="#76b7fb", edgecolor="black")
        ax_hist.set(xlabel="Taxa final r_T", ylabel="Contagem", title="Histograma dos terminais")
        ax_hist.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
        st.pyplot(fig_hist)
        st.caption("Contagem simples é mais intuitiva para ver a concentração das trajetórias.")

    with tab_term:
        st.markdown("### Estrutura a termo simulada")
        st.write(
            f"Monte Carlo da estrutura a termo sob {model_name}. A tabela mostra maturidades, preços e yields; "
            "os dados também podem ser gerados pela CLI (`term-structure`)."
        )
        if show_term_structure and term_paths is not None:
            bond_price_fn = model_cfg["bond_price"]
            maturities = np.linspace(0.25, 10.0, 40)
            rows = []
            base_seed = int(seed) * 100
            for idx, maturity in enumerate(maturities):
                steps = max(1, int(maturity * steps_per_year))
                price, stderr = bond_price_fn(
                    params=params,
                    T=float(maturity),
                    n_paths=term_paths,
                    n_steps=steps,
                    seed=base_seed + idx,
                    scheme=scheme,
                )
                yld = 0.0 if maturity == 0 else -math.log(max(price, 1e-12)) / maturity
                rows.append({
                    "T": maturity,
                    "price": price,
                    "stderr": stderr,
                    "zero_rate": yld,
                })
            ts_df = pd.DataFrame(rows)
            st.dataframe(ts_df, use_container_width=True)

            fig, ax_price = plt.subplots(figsize=(7, 4))
            ax_yield = ax_price.twinx()
            ax_price.plot(ts_df["T"], ts_df["price"], "o-", color="#1f77b4", label="Preco")
            ax_yield.plot(ts_df["T"], ts_df["zero_rate"], "s--", color="#d62728", label="Yield")
            ax_price.set(xlabel="Maturidade", ylabel="Preco", title="Curva zero-coupon")
            ax_yield.set(ylabel="Yield")
            ax_price.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
            lines = ax_price.get_lines() + ax_yield.get_lines()
            labels = [line.get_label() for line in lines]
            ax_price.legend(lines, labels, loc="best")
            st.pyplot(fig)
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
                "As curvas abaixo usam o mesmo preset e horizonte. A primeira figura mostra a média temporal "
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
    if swaption_settings and tab_swaption is not None:
        with tab_swaption:
            st.markdown("### Precificação de swaption via Monte Carlo")
            st.caption("Os resultados utilizam os mesmos presets escolhidos no menu lateral. Para reduzir o tempo, mantenha o número de caminhos moderado.")
            schedule = SwapSchedule.from_tenor(
                exercise=swaption_settings["exercise"],
                tenor=swaption_settings["tenor"],
                freq=swaption_settings["freq"],
            )
            rows = []
            for idx, name in enumerate(compare_models_ordered):
                cfg = MODEL_REGISTRY[name]
                try:
                    params_cmp = cfg["get_params"](preset)
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
                    rows.append(
                        {
                            "Modelo": name,
                            "Preco": price_mc,
                            "Erro padrão": stderr_mc,
                        }
                    )
                except Exception as exc:
                    st.error(f"Falha ao precificar {name}: {exc}")
            if rows:
                df_swap = pd.DataFrame(rows).set_index("Modelo")
                st.dataframe(df_swap.style.format({"Preco": "{:.4f}", "Erro padrão": "{:.4f}"}))
                st.caption("Visualização do preço estimado por modelo.")
                st.bar_chart(df_swap["Preco"])
                st.write(
                    f"Swaption {swaption_settings['kind']} | Exercício {swaption_settings['exercise']} | "
                    f"Tenor {swaption_settings['tenor']} | Strike {swaption_settings['strike']:.3%}"
                )
            else:
                st.info("Ative ao menos um modelo válido para visualizar o preço da swaption.")
    if alm_settings and tab_alm is not None:
        with tab_alm:
            st.markdown("### Cenários ALM")
            st.write("Aplica choques determinísticos (parallel, steepener, flattener, ramp) sobre uma curva média simulada.")
            try:
                data_json = json.loads(alm_settings["json"])
                assets_df = _build_cashflow_df(data_json.get("assets", []), "assets")
                passives_df = _build_cashflow_df(data_json.get("passives", []), "passives")
                max_time = float(max(assets_df["time"].max(), passives_df["time"].max()))
                horizon = max(1.0, max_time + 0.5)
                alm_steps = alm_settings["steps_per_year"]
                n_steps = int(horizon * alm_steps)
                cfg = MODEL_REGISTRY[model_name]
                params_alm = cfg["get_params"](preset)
                scheme_alm = scheme if scheme in cfg["schemes"] else cfg["schemes"][cfg["default_scheme_index"]]
                t_curve, paths_curve = cfg["simulate"](
                    scheme=scheme_alm,
                    params=params_alm,
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
                            "PV líquido": pv_assets - pv_passives,
                            "Duration ativos": _duration(assets_df, t_curve, curve),
                            "Duration passivos": _duration(passives_df, t_curve, curve),
                        }
                    )
                df_alm = pd.DataFrame(records).set_index("Cenario")
                st.dataframe(
                    df_alm.style.format(
                        {
                            "PV ativos": "{:,.0f}",
                            "PV passivos": "{:,.0f}",
                            "PV líquido": "{:,.0f}",
                            "Duration ativos": "{:.2f}",
                            "Duration passivos": "{:.2f}",
                        }
                    )
                )
                st.caption("PV líquido por cenário (positivo = superávit dos ativos).")
                st.bar_chart(df_alm["PV líquido"])
                fig_curve, ax_curve = plt.subplots(figsize=(7, 4))
                for sc_name, curve in scenario_curves.items():
                    ax_curve.plot(t_curve, curve, label=sc_name)
                ax_curve.set(xlabel="Tempo", ylabel="Taxa média", title="Curvas simuladas + choques")
                ax_curve.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
                ax_curve.legend(loc="upper right", fontsize="small")
                st.pyplot(fig_curve)
                net_base = (
                    assets_df.groupby("time")["amount"].sum()
                    - passives_df.groupby("time")["amount"].sum()
                ).reset_index(name="net_amount")
                net_base = net_base.set_index("time")
                st.caption("Fluxo líquido (ativos - passivos) por bucket de tempo.")
                st.bar_chart(net_base)
            except Exception as exc:
                st.error(f"Erro ao calcular cenários ALM: {exc}")
    if show_validation and tab_val is not None and val_maturities:
        with tab_val:
            st.markdown("### Validação analítica")
            st.write(
                "Tabela e gráficos comparando Monte Carlo com fórmulas fechadas do modelo. "
                "Útil para verificar viés numérico e erro vs. passo."
            )
            st.subheader("Comparacao zero-coupon (MC vs analitico)")
            price_df = compare_zero_coupon_prices(
                params=params,
                maturities=val_maturities,
                n_paths=int(val_paths),
                steps_per_year=int(val_steps),
                seed=int(seed) * 3,
            )
            st.dataframe(price_df, use_container_width=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(price_df["T"], price_df["abs_error"], "o-", label="|Erro|")
            ax.set(xlabel="Maturidade", ylabel="Erro absoluto", title="Erro de preco vs maturidade")
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.3)
            st.pyplot(fig)

            st.subheader("Erro vs passos (fixando maturidade)")
            err_df = zero_coupon_error_by_steps(
                params=params,
                maturity=1.0,
                n_paths=int(val_paths),
                steps_list=[32, 64, 128, 256],
                seed=int(seed) * 5,
                scheme=scheme,
            )
            st.dataframe(err_df, use_container_width=True)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.loglog(err_df["dt"], err_df["abs_error"], "s-", label="|Erro|")
            ax2.set(xlabel="dt", ylabel="Erro absoluto", title="Erro vs passo (log-log)")
            ax2.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.3)
            st.pyplot(fig2)

            st.subheader("Momentos vs analitico")
            moment_result = compare_moments(
                params=params,
                T=1.0,
                n_paths=int(val_paths),
                n_steps=int(val_steps),
                seed=int(seed) * 7,
                scheme=scheme,
            )
            col_m = st.columns(2)
            col_m[0].metric("Media (MC vs analitico)", f"{moment_result['mean_mc']:.4f}", f"ref {moment_result['mean_analytic']:.4f}")
            col_m[1].metric("Var (MC vs analitico)", f"{moment_result['var_mc']:.4f}", f"ref {moment_result['var_analytic']:.4f}")

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
                df = pd.read_csv(data_path)
                if "rate" not in df.columns:
                    raise ValueError("CSV precisa conter a coluna 'rate'.")
                last_rate = float(df["rate"].tail(1))
                mats = [float(x.strip()) for x in calib_maturities.split(",") if x.strip()]
                market_prices = np.exp(-last_rate * np.asarray(mats))
                initial_params = model_cfg["get_params"](calib_initial)
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
