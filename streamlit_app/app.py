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
    if show_validation:
        tabs.append("Validacao analitica")
    if show_calibration:
        tabs.append("Calibracao mercado")
    created_tabs = iter(main.tabs(tabs))
    tab_paths = next(created_tabs)
    tab_hist = next(created_tabs)
    tab_term = next(created_tabs)
    tab_conv = next(created_tabs)
    tab_compare = next(created_tabs) if has_comparison else None
    tab_val = next(created_tabs) if show_validation else None
    tab_calib = next(created_tabs) if show_calibration else None

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
