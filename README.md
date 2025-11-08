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

## Testes e notebooks

Rode `pytest` para validar o pacote. O notebook `notebooks/projeto1_demo.ipynb` demonstra trajetorias, convergencia e precificacao usando os modulos `cir`.

## Creditos

Projeto desenvolvido para **Modelagem Matematica em Financas II (UFRJ, 2025/2)**, cobrindo implementacao numerica do modelo CIR, analises de convergencia, distribuicao terminal, precificacao de bonds e geracao automatizada de figuras e tabelas.
