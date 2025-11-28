# Modelo CIR e comparativos de taxa curta

Implementacao didatica do processo Cox-Ingersoll-Ross (CIR) com pipeline completo: simulacao (Euler-Maruyama e Milstein), analiticos fechados, estimativa de ordem de convergencia, precificacao Monte Carlo de zero-coupons, calibracao contra curva DI, dashboard Streamlit e benchmarks lado a lado com Vasicek e Hull-White. Os presets respeitam a condicao de Feller (`2 * kappa * theta > sigma**2`).

## Visao geral do repositorio
- `cir/`: parametros, RNG, SDEs, simulacao, graficos, convergencia, validacao, calibracao e CLI Typer.
- `benchmarks/`: modelos Vasicek/Hull-White, scripts de comparacao e CSVs gerados.
- `data_loaders/`: parsers para curva zero (`CurvaZero_*.csv`) e SELIC diaria.
- `streamlit_app/app.py`: dashboard interativo que reaproveita os modulos do pacote e os carregadores de dados reais.
- `examples/`: scripts de precificacao de swaption e cenarios ALM com utilitarios compartilhados.
- `scripts/`: automacoes para baixar curva DI (`fetch_di_curve.py`) e rodar lotes de simulacao.
- `tests/`: suite Pytest cobrindo formulas, SDEs, bonds, calibracao e modelos auxiliares.
- Saidas: tabelas em `data/`, figuras em `figures/`, exemplos de dados reais em `data/real_data/` (`CurvaZero_17112025.csv`, `taxa_selic_apurada.csv`), relatorio em `report/relatorio_base.md` e notebooks em `notebooks/`.

## Instalacao rapida
```bash
python -m venv .venv
.venv\Scripts\activate    # no Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Para rodar `scripts/fetch_di_curve.py`, instale tambem `requests` (`pip install requests`) se ainda nao estiver presente.

## Dados de entrada: presets x dados reais
- Presets CIR em `cir.params`: `baseline`, `slow-revert`, `fast-revert` (todos respeitam Feller).
- Curvas reais: coloque `CurvaZero_*.csv` e `taxa_selic_apurada.csv` em `data/real_data/`. O loader `data_loaders.curves.load_curve_components` separa betas/lambdas, ETTJ e erros por titulo; `data_loaders.selic.load_selic_csv` limpa a serie diaria e `get_latest_rate` devolve o `r0` mais recente.
- Gerando curva DI rapida: `python scripts/fetch_di_curve.py --start 01/01/2020 --out data/raw_di_curve.csv` (usa API Bacen/SGS 4390).

## CLI (Typer)
Comandos sao executados a partir da raiz com `python -m cir.cli <comando> [opcoes]`. Saidas tabulares vao para `data/` e figuras para `figures/`.

| Comando | Finalidade | Exemplo |
| --- | --- | --- |
| `simulate-paths` | Simula trajetorias CIR e salva `paths_<scheme>_<preset>.png`. | `python -m cir.cli simulate-paths --scheme milstein --preset baseline --T 5 --steps-per-year 252 --n-paths 10 --seed 42` |
| `terminal-hist` | Histograma da distribuicao terminal com muitas trajetorias. | `python -m cir.cli terminal-hist --scheme em --preset fast-revert --T 5 --paths 50000 --steps-per-year 252 --seed 99` |
| `convergence` | RMSE acoplado em malhas varias + ajuste log-log. | `python -m cir.cli convergence --scheme em --preset baseline --T 1 --paths 50000 --base-steps "52,104,208,416,832" --seed 123` |
| `bond-price` | Monte Carlo de `B(0,T)` com erro padrao e CSV. | `python -m cir.cli bond-price --scheme em --preset slow-revert --T 5 --paths 5000 --steps-per-year 252 --seed 7` |
| `term-structure` | Curva zero-coupon via MC, salva tabela + figura. | `python -m cir.cli term-structure --scheme milstein --preset baseline --Tmax 10 --grid 40 --paths 5000 --steps-per-year 252 --seed 777` |
| `calibrate-market` | Ajusta `(kappa, theta, sigma, r0)` a uma curva DI simples. | `python -m cir.cli calibrate-market --data data/raw_di_curve.csv --maturities "0.25,0.5,1,2,5"` |

### Como lemos a convergencia
A rotina `convergence` fixa um Browniano de referencia na malha fina, agrega incrementos para malhas grosseiras e calcula RMSE final. Uma regressao linear em log-log (`log(error)` vs `log(dt)`) estima a ordem forte (~0.5 para EM, ~1.0 para Milstein). Saidas: `data/convergence_<scheme>.csv` e `figures/convergence_<scheme>.png`.

## Dashboard Streamlit (dados reais obrigatorios)
O app usa a mesma base de codigo e exige arquivos reais para continuar (se o checkbox nao estiver marcado ele encerra). Passo a passo:
1) Garanta `data/real_data/CurvaZero_*.csv` e `data/real_data/taxa_selic_apurada.csv` ou ajuste os caminhos na sidebar.  
2) Rode `streamlit run streamlit_app/app.py`.  
3) Na sidebar escolha modelo (CIR, Vasicek, Hull-White), esquema e horizontes. O app calibra CIR e Vasicek na curva real, monta o shift do Hull-White e substitui `r0` pelo ultimo valor da SELIC.

Abas disponiveis (ativadas conforme toggles):
- **Trajetorias / Distribuicao terminal / B(0,T) e yield**: visualizam as simulacoes atuais.
- **Convergencia**: somente CIR; calcula RMSE forte no EM e mostra grafico log-log + download do CSV.
- **Comparativo modelos**: sobrepoe medias temporais, metricas terminais e curvas zero-coupon analiticas para os modelos selecionados.
- **Curva real**: mostra ETTJ, curva prefixada e erros titulo a titulo do arquivo `CurvaZero_*`, alem de serie da SELIC.
- **Validacao analitica**: compara precos MC x formulas fechadas, erro vs passo e momentos (somente CIR).
- **Calibracao mercado**: ajusta parametros ao CSV da curva DI (coluna `rate`) e plota mercado vs ajuste.
- **Swaption MC**: precifica payer/receiver via Monte Carlo reutilizando os parametros calibrados.
- **Cenarios ALM**: aplica choques deterministas (parallel, steepener, flattener, ramp) sobre uma curva media simulada e calcula `pv_net`/durations.

## Benchmark multi-modelos (CIR, Vasicek, Hull-White)
`benchmarks/scripts/run_benchmark.py` gera `benchmarks/data/benchmark_<preset>.csv` com preco analitico/MC, media/variancia de `r_T` e, opcionalmente, calibra a partir de uma curva real (gera tambem `calibration_meta_<data>.json`). Exemplo:
```bash
python benchmarks/scripts/run_benchmark.py \
  --preset baseline \
  --maturities 0.5,1,2,5 \
  --hw-shift "0:0.0;5:0.01;10:0.015"
```
Para usar curva real: adicione `--curve-file data/real_data/CurvaZero_17112025.csv --curve-kind prefixados` e o script calibra CIR/Vasicek/shift do Hull-White antes de simular.

## Exemplos e scripts auxiliares
- `examples/price_swaption_mc.py`: Monte Carlo de swaption (CIR/Vasicek/Hull-White). Exemplo:  
  `python examples/price_swaption_mc.py --model Vasicek --preset baseline --kind payer --exercise 2 --tenor 3 --strike 0.04 --paths 20000`
- `examples/alm_scenarios.py`: cenarios ALM a partir de `examples/data/sample_cashflows.json`; gera `examples/output/alm_report.csv`.  
  `python examples/alm_scenarios.py --cashflows examples/data/sample_cashflows.json --paths 1000 --steps-per-year 52 --horizon 6 --out examples/output/alm_report.csv`
- `scripts/run_all_em.sh` / `scripts/run_all_milstein.sh`: pipeline completo (trajetorias 3 presets, histograma terminal, convergencia, term structure) em lote.
- `scripts/fetch_di_curve.py`: baixa a serie DI, normaliza e salva em CSV/Parquet. Use `--force-csv` para forcar CSV.

## Testes, notebooks e relatorio
- Testes: `pytest` roda validacoes de parametros, SDEs, bonds, calibracao e modelos auxiliares.
- Notebooks: `notebooks/projeto1_demo.ipynb` (trajetorias, convergencia, precificacao), `notebooks/projeto1_autonomo.ipynb` (pipeline completo) e `notebooks/projeto1_calibration.ipynb` (ajuste em dados DI).  
- Relatorio base em `report/relatorio_base.md` resume metodologia e pode ser alimentado com os CSVs/figuras do projeto.

## Creditos
Projeto criado para a disciplina **Modelagem Matematica em Financas II (UFRJ, 2025/2)**, cobrindo implementacao numerica do CIR, convergencia forte, distribuicao terminal, precificacao de bonds e comparativos multi-modelos para uso academico ou em portifolios.
