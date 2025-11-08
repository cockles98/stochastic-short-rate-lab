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

from cir.analytics import mean_short_rate, variance_short_rate, zero_coupon_price
from cir.bonds import bond_price_mc
from cir.convergence import strong_order_convergence
from cir.params import get_params_preset
from cir.simulate import simulate_paths
from cir.validation import (
    compare_moments,
    compare_zero_coupon_prices,
    zero_coupon_error_by_steps,
)

st.set_page_config(page_title="CIR Dashboard", layout="wide")
st.title("CIR Dashboard - Layout base")

sidebar, main = st.columns([1, 3])

with sidebar:
    preset = st.selectbox("Preset", ["baseline", "slow-revert", "fast-revert"], index=0)
    scheme = st.selectbox("Esquema", ["em", "milstein"], index=1)
    T = st.slider("Horizonte T", min_value=0.5, max_value=5.0, value=5.0, step=0.5)
    steps_per_year = st.slider("Passos por ano", min_value=52, max_value=365, value=252, step=26)
    n_paths = st.slider("Numero de caminhos", min_value=5, max_value=50, value=10, step=5)
    seed = st.number_input("Seed", value=42, step=1)

    st.markdown("---")
    st.subheader("Opcoes adicionais")
    show_term_structure = st.checkbox("Calcular term structure", value=False)
    show_convergence = st.checkbox("Calcular convergencia forte (EM)", value=False)
    show_validation = st.checkbox("Comparar com formulas analiticas", value=False)
    term_paths = None
    if show_term_structure:
        term_paths = st.slider("Caminhos para term structure", min_value=500, max_value=5000, value=2000, step=500)

    if show_convergence:
        conv_paths = st.number_input("Caminhos para convergencia", value=2000, min_value=500, max_value=200000, step=500)
        conv_steps = st.multiselect(
            "Steps (use valores que dividem a malha mais fina)",
            options=[52, 104, 208, 416, 832],
            default=[52, 104, 208, 416, 832],
        )
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
    else:
        conv_paths, conv_steps = None, None
        val_maturities, val_paths, val_steps = None, None, None

st.caption(
    "Controles basicos conectados aos modulos do pacote. As abas abaixo exibem simulacoes, "
    "termostructure, convergencia e validacao analitica (quando habilitado)."
)

params = get_params_preset(preset)
n_steps = int(T * steps_per_year)
paths = None

try:
    t, paths = simulate_paths(
        scheme=scheme,
        params=params,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=int(seed),
    )
except Exception as exc:
    st.error(f"Erro ao simular caminhos: {exc}")

if paths is not None:
    tabs = ["Trajetorias", "Distribuicao terminal", "B(0,T) e yield", "Convergencia"]
    if show_validation:
        tabs.append("Validacao analitica")
    tab_paths, tab_hist, tab_term, tab_conv, *rest = main.tabs(tabs)
    tab_val = rest[0] if rest else None

    with tab_paths:
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
        terminals = paths[:, -1]
        bins = min(40, max(5, paths.shape[1] // 4))
        density, edges = np.histogram(terminals, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        hist_df = pd.DataFrame({"taxa": centers, "densidade": density}).set_index("taxa")
        st.bar_chart(hist_df)
        st.write("Concentracao proxima de theta confirma a reversao a media esperada para o CIR.")

    with tab_term:
        if show_term_structure and term_paths is not None:
            maturities = np.linspace(0.25, 10.0, 40)
            rows = []
            base_seed = int(seed) * 100
            for idx, maturity in enumerate(maturities):
                steps = max(1, int(maturity * steps_per_year))
                price, stderr = bond_price_mc(
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
        if show_convergence and conv_paths and conv_steps:
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
    if show_validation and tab_val is not None and val_maturities:
        with tab_val:
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

else:
    main.warning("Aguardando configuracao valida para simular trajetorias.")
