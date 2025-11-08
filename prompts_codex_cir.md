# Prompts para o Codex — Projeto CIR (Modelagem Matemática em Finanças II)

> **Como usar**: copie e cole cada prompt no Codex **um por vez**, na ordem. Eles geram um repositório completo com:
> - Implementação do modelo CIR com **Euler–Maruyama** e **Milstein**
> - Simulações (3 conjuntos de parâmetros, 5–10 trajetórias cada)
> - **Ordem de convergência** (gráfico log–log + regressão linear)
> - **Distribuição terminal**
> - **Precificação de bond** por Monte Carlo e **estrutura a termo**
>
> Linguagem: **Python 3.11+** | Pacotes principais: `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`, `typer`, `pytest`.

---

## Prompt 1 — Estrutura do repositório e `requirements.txt`

Você é meu par programador. Crie um projeto Python organizado para o **Modelo CIR** com esta estrutura:

```
cir-cirprojeto/
├─ cir/
│  ├─ __init__.py
│  ├─ params.py
│  ├─ rng.py
│  ├─ sde.py
│  ├─ simulate.py
│  ├─ convergence.py
│  ├─ bonds.py
│  ├─ plots.py
│  └─ cli.py
├─ figures/            # imagens geradas
├─ data/               # (se precisar salvar saídas intermediárias)
├─ notebooks/          # (opcional)
├─ scripts/            # (serão criados depois)
├─ tests/
│  ├─ test_params.py
│  ├─ test_sde.py
│  └─ test_bonds.py
├─ README.md
└─ requirements.txt
```

**Linguagem:** Python 3.11+  
**Dependências mínimas:** `numpy`, `scipy`, `matplotlib`, `pandas`, `tqdm`, `typer`, `pytest`.  
Crie `requirements.txt`, inicialize `README.md` com instruções curtas de instalação/uso e explique brevemente o escopo do projeto (ligado ao CIR, condição de Feller e objetivos). **Não escreva código ainda**, só os arquivos vazios e o README com “Como rodar”.

---

## Prompt 2 — Parâmetros + checagem de Feller (`params.py`)

Implemente `cir/params.py`:
- Crie um `@dataclass` `CIRParams(kappa: float, theta: float, sigma: float, r0: float)`.
- Valide **condição de Feller**: `2*kappa*theta > sigma**2`. Lance `ValueError` se violar.
- Forneça 3 *presets* seguros que respeitem Feller para testes:
  1) `kappa=1.2, theta=0.05, sigma=0.20, r0=0.03`
  2) `kappa=0.5, theta=0.08, sigma=0.25, r0=0.04`
  3) `kappa=3.0, theta=0.02, sigma=0.10, r0=0.015`
- Função `get_params_preset(name: str) -> CIRParams` com nomes: `"baseline"`, `"slow-revert"`, `"fast-revert"`.
- Docstrings claras citando o modelo CIR e a necessidade de Feller.

---

## Prompt 3 — RNG e reprodutibilidade (`rng.py`)

Implemente `cir/rng.py`:
- Função `make_rng(seed: int | None) -> np.random.Generator` usando `PCG64`.
- Função utilitária `normal_increments(rng, n_paths, n_steps, dt)` que retorna `dW` com shape `(n_paths, n_steps)` ~ `N(0, dt)`.

---

## Prompt 4 — Esquemas de Euler–Maruyama e Milstein (`sde.py`)

Implemente `cir/sde.py` com:
- Assinaturas:
  - `euler_maruyama(params: CIRParams, T: float, n_steps: int, n_paths: int, rng) -> np.ndarray`
  - `milstein(params: CIRParams, T: float, n_steps: int, n_paths: int, rng) -> np.ndarray`
- Ambos retornam `R` com shape `(n_paths, n_steps+1)` incluindo `t0`.
- Fórmulas:
  - **EM:** `R_{t+Δ}= R_t + κ(θ−R_t)Δ + σ√(max(R_t,0))√Δ ξ`
  - **Milstein para CIR:** `R_{t+Δ}= R_t + κ(θ−R_t)Δ + σ√(max(R_t,0))√Δ ξ + 0.25*σ**2*(ξ**2 - 1)*Δ`
- Trate **positividade**: use `max(R_t, 0)` nas raízes e **clamp final** `R = np.maximum(R, 0.0)`.
- Documente limitações (EM pode enviesar perto de zero; Milstein melhora ordem).
- Sem loops externos lentos (vectorize). Teste numérico simples no docstring com `baseline`.

---

## Prompt 5 — Motor de simulação e múltiplos cenários (`simulate.py`)

Implemente `cir/simulate.py`:
- Função `simulate_paths(scheme: Literal["em","milstein"], params: CIRParams, T: float, n_steps: int, n_paths: int, seed: int | None) -> (t, R)`.
- **Cenários**: função `run_scenarios()` que gera **3 gráficos separados** com **5–10 trajetórias** cada, para os 3 *presets* (`baseline`, `slow-revert`, `fast-revert`) e esquema selecionado. Salve em `figures/paths_<scheme>_<preset>.png`.

---

## Prompt 6 — Módulo de gráficos (`plots.py`)

Implemente `cir/plots.py`:
- Funções:
  - `plot_paths(t, R, title, path_png)`
  - `plot_hist_terminal(R_T, bins, title, path_png)` (para distribuição terminal)
  - `plot_loglog_convergence(dts, errors, slope, intercept, path_png)` (para ordem de convergência)
  - `plot_yield_curve(maturities, prices, yields, path_png)` (para estrutura a termo)
- Layout limpo (legendas, rótulos, grade leve). Salve `.png` em `figures/`.

---

## Prompt 7 — Ordem de convergência (forte) (`convergence.py`)

Implemente `cir/convergence.py`:
- Função `strong_order_convergence(scheme, params, T, n_paths, base_steps_list, seed)`:
  - Use **acoplamento de ruído**: gere incrementos com a malha **mais fina** e **agregue** para dts mais grossos (somando blocos de normais) para comparar **no mesmo Browniano**.
  - Erro forte no tempo final `T`: `RMSE = sqrt(mean((R_T_dt - R_T_ref)**2))`, onde `ref` usa o menor `dt`.
  - Ajuste **reta** em gráfico **log–log** (`log(error)` vs `log(dt)`) por **regressão linear** → reporte **inclinação** (ordem).
- Função `run_convergence_report(...)` que salva CSV (`data/convergence_<scheme>.csv`) e a figura `figures/convergence_<scheme>.png`.

---

## Prompt 8 — Distribuição terminal (`simulate.py` + `plots.py`)

Adicione em `simulate.py` função `plot_terminal_distribution(...)`:
- Simule muitas trajetórias (ex.: `n_paths=50_000`, `T=5`, `n_steps=5*252`), pegue `R_T` e plote **histograma** com `bins=100`. Salve `figures/hist_terminal_<scheme>_<preset>.png`.

---

## Prompt 9 — Precificação de bond e estrutura a termo (`bonds.py`)

Implemente `cir/bonds.py`:
- Função `discount_factors_from_paths(t, R)`: aproxime \(\int_0^T R(s)ds\) por regra do **trapézio** na malha temporal; `D = exp(-integral)`.
- Função `bond_price_mc(params, T, n_paths, n_steps, seed, scheme) -> (price, std_err)`:
  - Simule caminhos, compute `D(0,T)` por caminho e retorne **média** + **erro padrão**.
- Função `term_structure(params, maturities, n_paths, steps_per_year, seed, scheme)`:
  - Calcule **B(0,T)** para cada `T` de `maturities` (ex.: `np.linspace(0.25, 10, 40)`).
  - Compute **zero rate**: `y(T) = -ln(B(0,T))/T`.
  - Salve CSV com `T, price, stderr, zero_rate` e gráfico `figures/term_structure_<scheme>_<preset>.png`.

---

## Prompt 10 — CLI com Typer (`cli.py`)

Implemente `cir/cli.py` com `typer`:
- Comandos:
  - `simulate-paths --scheme [em|milstein] --preset [baseline|slow-revert|fast-revert] --T 5 --steps-per-year 252 --n-paths 10 --seed 42`
  - `convergence --scheme [em|milstein] --preset ... --T 1 --paths 50_000 --base-steps "52,104,208,416,832"`
  - `terminal-hist --scheme ... --preset ... --T 5 --paths 50_000 --steps-per-year 252`
  - `bond-price --scheme ... --preset ... --T 5 --paths 5_000 --steps-per-year 252`
  - `term-structure --scheme ... --preset ... --Tmax 10 --grid 40 --paths 5_000 --steps-per-year 252`
- Salve todas as figuras em `figures/` e resultados tabulares em `data/`.
- No `__main__`, exponha `app()` para `python -m cir.cli ...`.

---

## Prompt 11 — Testes (`tests/`)

Escreva testes:
- `test_params.py`: presets respeitam **Feller**; lançar erro se violar.
- `test_sde.py`: para `n_paths` médio e `T` curto, **não negatividade** do `R` (após clamp) e **estabilidade numérica** (sem `nan/inf`).
- `test_bonds.py`: **monotonicidade**: `B(0, T2) <= B(0, T1)` para `T2>T1`; **taxas zero** não-negativas para parâmetros razoáveis.
Use sementes fixas para reprodutibilidade.

---

## Prompt 12 — README completo

Reescreva `README.md` com:
- Resumo do modelo CIR, **condição de Feller** e os **6 objetivos** do projeto (liste-os).
- Como instalar e rodar cada comando da CLI (exemplos de uso).
- Explicação breve sobre **ordem de convergência** (log–log + regressão linear).
- Significado dos gráficos gerados e onde são salvos.
- Créditos/curso/disciplina (UFRJ, 2025/2), mencionando o enunciado do projeto.

---

## Prompt 13 — Notebook demonstrativo (opcional)

Crie `notebooks/projeto1_demo.ipynb` com células que:
1) Rodem simulação de trajetórias (3 presets) e salvem os gráficos;
2) Rodem estudo de convergência e exibam a inclinação estimada;
3) Calculem `B(0,T)` para alguns `T` e, por fim, a **estrutura a termo**;
Use `matplotlib` inline e caminhos relativos do pacote.

---

## Prompt 14 — Scripts de exemplo (conveniência)

Adicione no README uma seção “Scripts úteis” e crie scripts `.sh` (Unix) em `scripts/`:
- `run_all_em.sh` e `run_all_milstein.sh` que:
  - Geram **paths** (3 presets),
  - **histograma** terminal,
  - **convergência**,
  - **term structure** (maturidades de 0.25 a 10 anos, 40 pontos, 5k paths),
  - salvando tudo em `figures/` e `data/`.
Dê `chmod +x` nos scripts.

---

## Prompt 15 — Qualidade e performance

Faça uma passada de otimização:
- Vectorize tudo que for possível.
- Evite cópias desnecessárias de arrays.
- Garanta que `n_paths` grande (p. ex. 100k) funcione em máquinas comuns (memória) — permita `chunking` opcional nos loops externos quando precisar.
- Verifique docstrings, *type hints* e mensagens de erro.

---

## Prompt 16 — Execução de verificação final

Gere, com `python -m cir.cli ...`, as seguintes saídas para checar:
1) 3 figuras de trajetórias (`paths_<scheme>_<preset>.png`) para `scheme=milstein`, `n_paths=10`, `T=5`, `252 steps/yr`;
2) Convergência para `scheme=em` com `base-steps: 52,104,208,416,832` (salvar CSV + PNG);
3) Histograma terminal `T=5`, `n_paths=50_000`, `scheme=milstein`;
4) `bond-price` para `T ∈ {1,3,5}` com `n_paths=5_000`;
5) Estrutura a termo de 0.25 a 10 anos (`grid=40`), `n_paths=5_000`, `scheme=milstein`.
Confirme no console os caminhos dos arquivos salvos.

---

## Prompt 17 — Pequeno relatório `.md`

Crie `report/relatorio_base.md` (Markdown) com:
- **Introdução** (modelo CIR e Feller);
- **Métodos numéricos** (EM e Milstein, escolhas de malha e tratamento da positividade);
- **Resultados** (insira as figuras geradas por caminho relativo);
- **Convergência** (inclinação estimada e breve análise);
- **Bond e Estrutura a Termo** (tabela de `B(0,T)` e curva de `y(T)`);
- **Conclusões e próximos passos**.
Use texto objetivo, em português, pronto para ser adaptado ao relatório final da disciplina.

---

### Observações
- Estes prompts cobrem todos os objetivos do enunciado: implementação EM/Milstein, simulações, ordem de convergência, histograma terminal, `B(0,T)` por Monte Carlo e *yield curve*.
- Se necessário, posso gerar diretamente o código do repositório com base nestes prompts.
