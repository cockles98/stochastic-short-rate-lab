# Benchmark multi-modelos

Esta pasta contém utilitários para comparar o comportamento dos modelos de taxa curta implementados no projeto:

- **CIR** – difusão raiz quadrada, estritamente positiva sob a condição de Feller;
- **Vasicek** – Ornstein-Uhlenbeck aditivo, permite taxas negativas e tem fórmulas mais simples;
- **Hull-White** – Vasicek com deslocamento determinístico `s(t)` que “encaixa” qualquer curva inicial.

## Estrutura

- `models/vasicek.py` e `models/hull_white.py`: implementações modulares com simulação, precificação analítica e Monte Carlo.
- `scripts/run_benchmark.py`: gera um CSV com métricas comparativas (preços, médias, variâncias) para cada modelo a partir de um preset e grade de maturidades.
- `data/`: destino dos arquivos `.csv` produzidos pelos scripts; podem ser reutilizados no Streamlit ou em notebooks.

## Como rodar

```bash
python benchmarks/scripts/run_benchmark.py \
  --preset baseline \
  --maturities 0.5,1,2,5 \
  --hw-shift "0:0.0;5:0.01;10:0.015"
```

Argumentos principais:

- `--preset`: usa os mesmos presets do pacote `cir.params` (baseline, slow-revert, fast-revert);
- `--maturities`: lista de maturidades (anos) para precificação analítica e Monte Carlo;
- `--hw-shift`: agenda do deslocamento determinístico do Hull-White (`tempo:valor` separado por `;`), permitindo deslocar a curva inicial sem recalibrar outros parâmetros.

O script produz `benchmarks/data/benchmark_<preset>.csv` com colunas:

| modelo | maturity | analytic_price | mc_price | mc_std | mean_rT | var_rT |

## Trade-offs destacados

| Modelo    | Volatilidade / suporte | Reversão à média | Calibração | Observações |
|-----------|-----------------------|------------------|------------|-------------|
| CIR       | Difusão raiz quadrada, mantém taxas positivas; menor risco de valores negativos. | Reversão forte quando `kappa` alto, mas a condição de Feller acopla `theta` e `sigma`. | Mais rígido: ajustar a curva exige respeitar Feller e lidar com parâmetros correlacionados. | Ideal quando positividade estrita é requisito regulatório. |
| Vasicek   | Difusão constante; permite taxas negativas, resultando em caudas mais largas. | Reversão linear simples, ortogonal aos demais parâmetros. | Calibração rápida (solução fechada para `P(0,T)`), ótima para dados sintéticos ou estimativas preliminares. | Pode violar positividade no curto prazo, mas facilita análises gaussianas. |
| Hull-White| Mesmo núcleo Vasicek, porém com shift determinístico que ajusta a curva inicial exatamente. | Idêntico ao Vasicek para a parte estocástica; o shift altera apenas o nível. | Muito flexível: a agenda `s(t)` absorve discrepâncias da curva de entrada. Requer definir/estimar o deslocamento. | Útil para stress tests em curvas curvas específicas sem reconfigurar toda a dinâmica. |

Essas comparações alimentam a nova aba “Comparativo modelos” do Streamlit e fornecem narrativa para relatórios ou discussões sobre model risk. Ajuste o shift do Hull-White e os presets para ilustrar volatilidade relativa, convergência e aderência à curva zero-coupon.
