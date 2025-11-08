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
