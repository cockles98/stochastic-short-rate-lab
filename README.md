# 📉 Stochastic Short Rate Lab (CIR, Vasicek & Hull-White)

<div align="center">

![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Math](https://img.shields.io/badge/Method-Euler--Maruyama%20%2F%20Milstein-purple?style=for-the-badge)
![Academic](https://img.shields.io/badge/Academic-UFRJ-green?style=for-the-badge)

</div>

> **Framework completo para simulação, precificação e calibração de modelos de taxa curta, com dashboard interativo e análise de convergência forte.**

Este repositório contém uma implementação robusta do processo **Cox-Ingersoll-Ross (CIR)** e benchmarks comparativos (Vasicek, Hull-White). O projeto abrange desde a resolução numérica de Equações Diferenciais Estocásticas (SDEs) até a calibração com dados reais da **Curva DI brasileira**.

---

## 🎯 Destaques do Projeto

* **Pipeline Completo:** Simulação (Euler-Maruyama e Milstein), Precificação de Zeros e Bonds, e Calibração.
* **Rigor Matemático:** Validação da **Condição de Feller** ($2\kappa\theta > \sigma^2$) e estimativa de ordem de convergência forte.
* **Dados Reais:** Utilização de dados brasileiros reais da curva DI e taxa Selic.
* **Interatividade:** Dashboard **Streamlit** para análise de sensibilidade e cenários de ALM (Asset Liability Management).

---

## 📊 Galeria Visual

### 1. Dashboard Interativo (Streamlit)
*Visualização em tempo real das trajetórias, yield curves e calibração.*
<div align="center">
  <img src="figures/cir/trajectories.png" alt="Streamlit Dashboard Demo" width="700"/>
</div>
<div align="center">
  <img src="figures/cir/yeld_curves.png" alt="Streamlit Dashboard Demo" width="700"/>
</div>
<div align="center">
  <img src="figures/cir/calibration.png" alt="Streamlit Dashboard Demo" width="700"/>
</div>

### 2. Análise de Convergência
*Comparativo de erro forte (RMSE) da discretização de Euler Maruyama.*
<div align="center">
  <img src="figures/cir/convergence_em.png" alt="Convergence Analysis" width="700"/>
</div>

### 3. Exposição dos Dados Econômicos
*Demonstração gráfica dos dados da taxa selic e da curva pré-fixada.*
<div align="center">
  <img src="figures/cir/selic-values.png" alt="Convergence Analysis" width="800"/>
</div>
<div align="center">
  <img src="figures/cir/prefixed-curve.png" alt="Convergence Analysis" width="800"/>
</div>

---

## 📐 Fundamentação Teórica

O modelo CIR segue a seguinte dinâmica estocástica:

$$dr_t = \kappa(\theta - r_t)dt + \sigma \sqrt{r_t} dW_t$$

Onde a implementação garante a **positividade** da taxa e estabilidade numérica através do esquema de Milstein modificado para processos de raiz quadrada.

---

## 🚀 Instalação Rápida

```bash
# 1. Clone o repositório
git clone https://github.com/cockles98/cir-short-rate-lab.git
cd cir-short-rate-lab

# 2. Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

-----

## 🖥️ Dashboard & CLI

### Modo Interativo (Streamlit)

A maneira mais fácil de explorar o modelo.

```bash
streamlit run streamlit_app/app.py
```

> **Ou acessando diretamente o dashboard online através do [link](https://cockles98-stochastic-short-rate-lab-streamlit-appapp-slmkui.streamlit.app).**

*Funcionalidades:* Calibração em tempo real, Comparativo Visual (CIR vs Vasicek), Cenários de Stress (ALM).

### Modo CLI (Linha de Comando)

Para execuções em lote e geração de relatórios, utilize o módulo `cir.cli`.

| Comando | Descrição | Exemplo |
| :--- | :--- | :--- |
| `simulate-paths` | Gera trajetórias estocásticas | `python -m cir.cli simulate-paths --preset baseline` |
| `convergence` | Análise de erro forte (Log-Log) | `python -m cir.cli convergence --scheme milstein` |
| `term-structure` | Gera curva Zero-Coupon via MC | `python -m cir.cli term-structure --Tmax 10` |
| `calibrate-market` | Ajusta parâmetros à curva DI | `python -m cir.cli calibrate-market --data data/raw_di_curve.csv` |

-----

## 📂 Estrutura do Repositório

  * **`cir/`**: Núcleo da biblioteca (SDEs, Solvers, Calibração).
  * **`benchmarks/`**: Implementações comparativas (Vasicek, Hull-White).
  * **`streamlit_app/`**: Código do frontend interativo.
  * **`scripts/`**: Utilitários para download de dados (Data Fetchers).
  * **`tests/`**: Suite de testes automatizados (`pytest`) para validação matemática.
  * **`notebooks/`**: Estudos de caso e validações exploratórias.

-----

## 📜 Créditos e Contexto

Projeto desenvolvido para a disciplina de **Modelagem Matemática em Finanças II (UFRJ, 2025/2)**.

  * **Objetivo:** Implementação numérica rigorosa de modelos de taxa curta para precificação de derivativos e gestão de portfólio.
