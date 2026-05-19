# 🚗 Ford × FIAP — Sprint de IA & ML
## Segmentação e Classificação de Clientes para Retenção na Rede Oficial de Serviços

---

### Equipe

| Nome | RM |
|---|---|
| *Arthur Bobadilla Franchi* | RM 555056 |
| *Luan Orlandelli Ramos* | RM 554747 |
| *Jorge Luiz* | RM 554418 |

## 📋 Visão Geral

Este projeto foi desenvolvido como parte de um Sprint de Inteligência Artificial e Machine Learning em parceria com a Ford e a FIAP. O objetivo central é resolver um problema de **assimetria de informação no ponto de venda**: no momento da compra do veículo, a concessionária não sabe como o cliente se comportará em relação à manutenção futura na rede oficial.

A solução foi estruturada em **duas etapas analíticas complementares**:

```
Base 1 (Histórico Completo)
        │
        ▼
[Etapa 1] K-Means Clustering ──► 4 Segmentos Comportamentais
                                          │
                                          ▼ (rótulos como target)
Base 2 (Momento da Compra)
        │
        ▼
[Etapa 2] Logistic Regression ──► Previsão do Perfil no Ato da Compra
                                          │
                                          ▼
                            Protocolo de Retenção Personalizado
```

---

### Dados de entrada (não incluídos no repositório)

| Arquivo | Descrição | Uso |
|---|---|---|
| `ford_clientes_historico_completo.csv` | 500k clientes com histórico pós-venda completo | Etapa 1 — Segmentação |
| `ford_clientes_operacional_compra.csv` | 500k clientes com dados apenas do momento da compra | Etapa 2 — Classificação |

> ⚠️ Os datasets originais são confidenciais e não estão incluídos neste repositório.

---

## 🧠 Metodologia

### Parte 1 — Entendimento e Preparação dos Dados (EDA)

- Contextualização do problema de retenção na rede Ford
- Classificação das 37 variáveis em: **momento da compra** vs. **comportamentais pós-venda**
- Tratamento de valores nulos (imputação por mediana)
- Correção de inconsistências lógicas (flag `compra_a_vista` vs. `prazo_financiamento`)
- Winsorização (1%–99%) para variáveis com alta dispersão
- One-Hot Encoding para variáveis categóricas
- Validação exploratória das hipóteses de negócio

### Parte 2 — Segmentação Comportamental (Etapa 1)

- **Algoritmo:** K-Means Clustering
- **Features:** 12 variáveis comportamentais pós-venda (exclusivas da Base 1)
- **Definição de k:** Método do Cotovelo + Silhouette Score → **k = 4**
- **Silhouette Score final:** 0.32 (dataset completo, 500k registros)
- **Validação:** >91% de concordância com o ground truth `perfil_latente`
- Nomeação das personas com linguagem de negócio
- Estratégias de retenção propostas por segmento

### Parte 3 — Classificação Preditiva (Etapa 2)

- **Target:** rótulo do cluster da Etapa 1 (`segmento_kmeans`)
- **Features:** 44 variáveis do momento da compra (Base 2) — **zero data leakage**
- **Split:** 80% treino / 20% teste, estratificado
- **Modelos comparados:** Logistic Regression, Decision Tree, Random Forest
- **Modelo eleito:** Logistic Regression
- **Métricas finais:**

| Métrica | Resultado |
|---|---|
| Acurácia | 79,0% |
| F1-Score (macro) | 0,77 |
| F1-Score (weighted) | 0,79 |

---

## 👥 Os 4 Segmentos Descobertos

| Segmento | Tamanho | Churn | Share Rede | Gasto 24m | Satisfação | F1 do Modelo |
|---|---|---|---|---|---|---|
| 🟢 **Fiel** | 30,1% | 2,2% | 93,1% | R$ 4.266 | 9,4/10 | **0,97** |
| 🟠 **Econômico** | 25,4% | 7,1% | 57,4% | R$ 1.429 | 7,1/10 | **0,83** |
| 🔴 **Esquecido** | 23,8% | 25,3% | 26,2% | R$ 315 | 3,9/10 | **0,71** |
| 🔵 **Abandono** | 20,7% | 27,5% | 17,4% | R$ 213 | 4,8/10 | **0,56** |

### Estratégias por Segmento

**🟢 Fiel** — Tratamento premium, programa de fidelidade, pipeline de recompra, monitoramento de NPS.

**🟠 Econômico** — Plano de manutenção com desconto no ato da compra, alertas de campanhas, gamificação de benefícios.

**🔴 Esquecido** — Comunicação proativa 30/15 dias antes do prazo, agendamento simplificado, app de lembretes.

**🔵 Abandono** — Experiência diferenciada na 1ª revisão, oferta de captura condicionada ao retorno, diagnóstico gratuito na 2ª visita.

---

## ⚠️ Regra Crítica — Anti Data Leakage

Este projeto implementa uma separação rigorosa de variáveis:

```
✅ PERMITIDO na Etapa 2 (Classificação):
   idade, renda, score de crédito, distância, modelo do veículo,
   valor, prazo, garantia estendida, plano de manutenção, etc.

🚫 PROIBIDO na Etapa 2 (são variáveis futuras):
   revisões realizadas, meses até 1ª revisão, share da rede,
   gasto com manutenção, churn, satisfação, etc.
```

Qualquer uso de variáveis pós-venda na etapa de classificação caracteriza **data leakage** e invalida completamente o modelo.

---

## 🛠️ Requisitos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn reportlab nbformat jupyter
```

### Versões utilizadas

| Biblioteca | Versão |
|---|---|
| Python | 3.12 |
| pandas | ≥ 2.0 |
| scikit-learn | ≥ 1.3 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| reportlab | ≥ 4.0 |

---

## 🚀 Como Executar

1. Clone o repositório e adicione os datasets na raiz do projeto.

2. Execute os notebooks na ordem:

```bash
jupyter notebook Ford_Sprint_Parte1_EDA.ipynb
jupyter notebook Ford_Sprint_Parte2_Segmentacao.ipynb
jupyter notebook Ford_Sprint_Parte3_Classificacao.ipynb
```

3. O arquivo `ford_segmentos_kmeans.csv` é gerado automaticamente ao final da Parte 2 e consumido pela Parte 3.

---

## 📊 Resultados e Entregáveis

| Entregável | Descrição |
|---|---|
| `Ford_Sprint_Parte1_EDA.ipynb` | Análise exploratória completa com justificativas técnicas |
| `Ford_Sprint_Parte2_Segmentacao.ipynb` | Pipeline de clustering, Elbow, Silhouette, personas e estratégias |
| `Ford_Sprint_Parte3_Classificacao.ipynb` | Modelo supervisionado, comparação de algoritmos, métricas e interpretação |
| `ford_segmentos_kmeans.csv` | Dataset com `cliente_id`, `cluster` e `segmento_kmeans` |
| `Ford_Relatorio_Executivo.pdf` | Relatório executivo completo (~18 páginas) com 8 visualizações |

---

## 🏗️ Arquitetura de Produção Sugerida

```
Sistema de CRM da Concessionária
        │
        ▼ (dados do ato da venda)
API REST — Modelo de Classificação
        │
        ▼
Perfil Previsto: fiel | economico | esquecido | abandono
        │
        ▼
Protocolo de Retenção Ativado Automaticamente
```

O modelo tem latência negligenciável (Logistic Regression) e pode ser servido como endpoint REST sem infraestrutura especial.

---

## 🔭 Próximos Passos

- **Monitoramento de drift:** reavaliar o modelo periodicamente à medida que o perfil da base de clientes evolui.
- **Enriquecimento de features:** dados externos (CEP socioeconômico, índice de concorrência local) podem melhorar o F1 do segmento Abandono.
- **Gradient Boosting:** explorar XGBoost/LightGBM para ganhos nas classes difíceis.
- **Teste A/B:** validar impacto real das estratégias de retenção em grupos de controle por segmento.
- **Re-treinamento contínuo:** pipeline MLOps com re-treinamento automático ao acumular novos dados históricos.

---

*Sprint IA & ML — Ford × FIAP*
