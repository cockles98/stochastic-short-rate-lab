# Modelo CIR - Projeto completo

O processo de Cox-Ingersoll-Ross (CIR) descreve a dinamica de uma taxa de juros de curto prazo positiva via

```
dr_t = kappa * (theta - r_t) dt + sigma * sqrt(r_t) dW_t
```

Para manter trajetorias estritamente positivas, aplicamos a **condicao de Feller** `2 * kappa * theta > sigma**2` em todos os presets. A partir dela, perseguimos seis objetivos centrais:

1. Validar parametros confiaveis (presets) e garantir reproducibilidade via RNG dedicado.
2. Implementar os esquemas de Euler-Maruyama e Milstein com clamps de positividade.
3. Simular cenarios e gerar graficos de trajetorias, histogramas e figuras auxiliares automaticamente.
4. Estimar a **ordem de convergencia forte** com graficos log-log e regressao linear.
5. Precificar titulos zero (Monte Carlo), calcular fatores de desconto e construir a **estrutura a termo**.
6. Expor toda a funcionalidade via CLI/Typer, notebooks e scripts reprodutiveis para apoiar estudos e relatorios.

Os modulos em `cir/` encapsulam parametros, RNG, SDEs, simulacao, convergencia, bonds e graficos. Saidas numericas residem em `data/`, figuras em `figures/`, experimentos interativos em `notebooks/` e automacoes em `scripts/`. A suite `tests/` cobre validacoes essenciais via Pytest.

## Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate    # no Linux/macOS use: source .venv/bin/activate
pip install -r requirements.txt
```

## CLI (Typer)

Execute os comandos a partir da raiz `cir-cirprojeto/` usando `python -m cir.cli <comando> [opcoes]`. Resultados tabulares vao para `data/` e figuras para `figures/`.

| Comando | Finalidade | Exemplo |
| --- | --- | --- |
| `simulate-paths` | Gera trajetorias e salva `paths_<scheme>_<preset>.png`. | `python -m cir.cli simulate-paths --scheme milstein --preset baseline --T 5 --steps-per-year 252 --n-paths 10 --seed 42` |
| `convergence` | Calcula RMSEs acoplados, salva CSV + log-log. | `python -m cir.cli convergence --scheme em --preset baseline --T 1 --paths 50000 --base-steps "52,104,208,416,832" --seed 123` |
| `terminal-hist` | Simula muitas trajetorias e plota o histograma terminal. | `python -m cir.cli terminal-hist --scheme milstein --preset fast-revert --T 5 --paths 50000 --steps-per-year 252 --seed 99` |
| `bond-price` | Monte Carlo para `B(0,T)` com erro padrao, grava CSV. | `python -m cir.cli bond-price --scheme em --preset slow-revert --T 5 --paths 5000 --steps-per-year 252 --seed 7` |
| `term-structure` | Varre maturidades, salva tabela + figura da curva. | `python -m cir.cli term-structure --scheme milstein --preset baseline --Tmax 10 --grid 40 --paths 5000 --steps-per-year 252 --seed 777` |
| `calibrate-market` | Ajusta `(kappa, theta, sigma, r0)` a uma curva DI simples usando `data/raw_di_curve*.csv`. | `python -m cir.cli calibrate-market --data data/raw_di_curve.csv --maturities "0.25,0.5,1,2,5"` |

Execute `python -m cir.cli --help` ou `<comando> --help` para mais opcoes.

## Ordem de convergencia

A rotina `convergence` fixa um Browniano de referencia com a malha mais fina, agrega os incrementos para malhas mais grossas e calcula o **erro forte** no tempo final (`RMSE`). Em seguida, ajusta-se uma reta em escala log-log (`log(error)` vs `log(dt)`) usando regressao linear; a inclinacao estimada indica a ordem de convergencia do esquema (~0.5 para Euler-Maruyama, ~1.0 para Milstein). Saidas:

- `data/convergence_<scheme>.csv`: `n_steps`, `dt`, `rmse`.
- `figures/convergence_<scheme>.png`: curva observada + linha ajustada.

## Graficos e saidas

- `paths_<scheme>_<preset>.png`: trajetorias Monte Carlo.
- `hist_terminal_<scheme>_<preset>.png`: distribuicao terminal apos `T`.
- `convergence_<scheme>.png`: grafico log-log com inclinacao estimada.
- `term_structure_<scheme>_<preset>.png`: precos vs yields (dois eixos).

Todos os arquivos ficam em `figures/`, enquanto tabelas (`bond_price_*.csv`, `term_structure_*.csv`, `convergence_*.csv`) sao armazenadas em `data/`.

## Streamlit app + validação analítica

`streamlit_app/app.py` usa os mesmos módulos `cir` e agora inclui:

- Sidebar com seleção de preset/esquema, controle de malha e toggles para term-structure e convergência forte;
- Abas **Trajetórias**, **Distribuição terminal**, **B(0,T)/yield** e **Convergência** (log–log + download de CSV);
- Aba **Validação analítica** (quando ativada) que compara Monte Carlo com as fórmulas fechadas do CIR via `cir.analytics` e `cir.validation`, exibindo:
  * tabela `MC vs analítico` para zero-coupons e gráfico de erro × maturidade;
  * gráfico log–log do erro vs passo (`zero_coupon_error_by_steps`);
  * métricas de momentos (`E[r_T]`, `Var[r_T]`) calculadas tanto por simulação quanto pela solução analítica.
- Um multiseletor que habilita comparativos lado a lado entre **CIR**, **Vasicek** (OU) e **Hull-White**; a aba “Comparativo modelos” sobrepõe médias temporais, métricas terminais e curvas zero-coupon analíticas, destacando diferenças de volatilidade, reversão à média e níveis deslocados.

Execute com:

```bash
streamlit run streamlit_app/app.py
```

### Validação analítica e relatórios

Além do dashboard, o pacote possui utilitários de comparação:

- `cir/analytics.py`: preço fechado do zero-coupon, média e variância de `r_T`;
- `cir/validation.py`: funções para gerar DataFrames com erros absolutos/relativos e curvas erro × passo;
- Testes (`tests/test_analytics.py`, `tests/test_validation.py`) asseguram consistência;
- O notebook `notebooks/projeto1_autonomo.ipynb` explica o pipeline inteiro e demonstra as comparações;
- `report/relatorio_base.md` resume metodologia, convergência e estrutura a termo.

Essas peças podem alimentar relatórios técnicos ou slides de portfólio usando os CSVs/figuras já gerados (`data/`, `figures/`).

### Benchmark multi-modelos e trade-offs

O diretório `benchmarks/` centraliza scripts e modelos auxiliares para comparar processos de taxa curta. O script

```bash
python benchmarks/scripts/run_benchmark.py --preset baseline --maturities 0.5,1,2,5 --hw-shift "0:0.0;5:0.01;10:0.015"
```

gera `benchmarks/data/benchmark_<preset>.csv` com preços analíticos/Monte Carlo e métricas de momentos para CIR, Vasicek e Hull-White. Use esse material para discutir:

- **Volatilidade**: Vasicek/Hull-White permitem taxas negativas devido à difusão aditiva, resultando em caudas mais largas que o CIR;
- **Reversão à média**: a mesma `kappa` produz trajetórias visivelmente diferentes quando combinada à restrição de positividade do CIR;
- **Facilidade de calibração**: Vasicek/Hull-White ajustam qualquer curva inicial via parâmetros lineares ou shift determinístico; o CIR exige respeitar a condição de Feller e apresenta maior acoplamento dos parâmetros;
- **Term structure**: curvas zero-coupon podem ser comparadas diretamente na nova aba do Streamlit para evidenciar deslocamentos e inclinações.

Detalhes adicionais sobre a organização do benchmark estão em `benchmarks/README.md`.

## Calibração e dados de mercado

- `scripts/fetch_di_curve.py`: baixa a série DI (Bacen/SGS 4390), normaliza datas e taxas e salva como CSV/Parquet (`data/raw_di_curve.*`).
- `scripts/fetch_di_curve.sh`: alternativa em shell usando `curl`.
- CLI `python -m cir.cli calibrate-market ...` calibra `(kappa, theta, sigma, r0)` a partir dessa curva e salva comparativos/JSON.
- Notebook `notebooks/projeto1_calibration.ipynb` replica a calibração com gráficos e discute próximos passos (bootstrapping/MLE).

Fluxo sugerido:

1. Baixe a curva:
   ```bash
   python scripts/fetch_di_curve.py --start 01/01/2020 --out data/raw_di_curve.csv
   ```
2. Rode a calibração básica:
   ```bash
   python -m cir.cli calibrate-market --data data/raw_di_curve.csv --maturities "0.25,0.5,1,2,5"
   ```
3. Abra o notebook `notebooks/projeto1_calibration.ipynb` para inspeções detalhadas (tabelas, gráficos, recomendações).

## Scripts uteis

Os scripts em `scripts/` encadeiam os comandos principais:

- `scripts/run_all_em.sh`
- `scripts/run_all_milstein.sh`

Execute-os na raiz do projeto:

```bash
./scripts/run_all_em.sh
./scripts/run_all_milstein.sh
```

Cada script gera trajetorias (3 presets), histograma terminal, estudo de convergencia e estrutura a termo (maturidades de 0.25 a 10 anos, 40 pontos, 5k paths).

### Casos de uso (examples/)

- `examples/price_swaption_mc.py`: precifica payer/receiver swaptions via Monte Carlo, aceitando modelo (CIR, Vasicek, Hull-White), preset, tenor, strike, frequência e número de caminhos. O script imprime um JSON com preço e erro padrão (`stderr`) e pode salvar em arquivo passando `--out`.
  ```bash
  python examples/price_swaption_mc.py --model Vasicek --preset baseline --kind payer --exercise 2 --tenor 3 --strike 0.04 --paths 20000
  ```
  Interprete o resultado comparando o `price` com proxies analíticas (quando disponíveis) ou usando Hull-White para incorporar shifts específicos.

- `examples/alm_scenarios.py`: gera cenários ALM a partir de um JSON simples contendo cash flows de ativos e passivos. Aplica choques determinísticos (parallel shift, steepener, flattener, ramp) e calcula `pv_assets`, `pv_passives`, `pv_net` e `duration_gap` por cenário.
  ```bash
  python examples/alm_scenarios.py --cashflows examples/data/sample_cashflows.json --paths 1000 --steps-per-year 52 --horizon 6 --out examples/output/alm_report.csv
  ```
  Use o CSV final para analisar o efeito dos choques: `pv_net` indica o impacto econômico, enquanto `duration_gap` evidencia descasamentos de sensibilidade entre ativos e passivos.

## Testes e notebooks

Rode `pytest` para validar o pacote. O notebook `notebooks/projeto1_demo.ipynb` demonstra trajetorias, convergencia e precificacao usando os modulos `cir`.

```bash
pytest
```

A suíte cobre parâmetros, SDEs, bonds, validação analítica, calibração e comparações com fórmulas fechadas.

## Creditos

Projeto desenvolvido para **Modelagem Matematica em Financas II (UFRJ, 2025/2)**, cobrindo implementacao numerica do modelo CIR, analises de convergencia, distribuicao terminal, precificacao de bonds e geracao automatizada de figuras e tabelas.
