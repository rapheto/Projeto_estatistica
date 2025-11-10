# Relatório Analítico — E-commerce (Análise exploratória e inferência)

Este relatório documenta a preparação dos dados, EDA (exploratory data analysis), inferência estatística e KPIs solicitados.
Os artefatos reprodutíveis estão incluídos: `analysis.py` (Python) e `analysis.sql` (SQL). A pasta `figures/` é gerada pelo script Python com os gráficos.

## Resumo executivo
- Base: `FACT_Orders.csv`, `DIM_Delivery.csv`, `DIM_Customer.csv`, `DIM_Products.csv`, `DIM_Shopping.csv`.
- Objetivos: medir receita, ticket, frete, atraso de entrega, conversão por meio de análise descritiva e ICs estatísticos.


## 1) Qualidade e preparação dos dados
- Arquivos lidos como CSV; colunas de tempo parseadas (`Order_Date`, `D_Forecast`, `D_Date`).
- Checagens realizadas no script:
  - Contagem de linhas/colunas e somas de NAs por arquivo.
  - Deduplicação por `Id` (nenhuma duplicata significativa encontrada nos heads).
  - Assunção empregada: os arquivos compartilham um campo `Id` que alinha as linhas (se seu DWH tiver chaves relacionais diferentes, adaptar joins no SQL).
- Tipos coerentes: datas convertidas, valores monetários mantidos como float.
- Tratamento de strings: trimming implícito ao ler CSV (pode-se aplicar .str.strip() em produção).
- Outliers: apontados no EDA (boxplots) e recomendação de regra IQR para flagging (o script salva os gráficos; extensão sugerida: marcar e salvar índices outliers com regra IQR ou z-score).

Diagrama de integridade (resumo):
- `FACT_Orders(Id)` ← LEFT JOIN `DIM_Delivery(Id)`, `DIM_Customer(Id)`
- `DIM_Shopping` e `DIM_Products` representam itens; para análises por produto/elasticidade, é necessário mapear linhas de item a orders (coluna order_id ausente no arquivo sample — adaptável).


## 2) Análise descritiva (EDA)
Principais estatísticas (dataset inteiro):
- Número de pedidos: ver `analysis.py` output
- Receita total: soma de `Total` (veja `analysis.py` output)
- Ticket médio: média de `Total` (IC abaixo)
- Desconto médio (%): média de `Discount`
- Frete médio por serviço: ver tabela `logistics performance` no script

Gráficos gerados (em `figures/`):
- `ticket_hist.png` — distribuição do ticket (Total)
- `leadtime_box.png` — boxplot de lead time (dias)
- `correlation_heatmap.png` — correlações entre Subtotal, Total, Discount, P_Sevice, lead time, delay
- `monthly_revenue.png` — série temporal de receita agregada por mês

Observações do EDA:
- A distribuição do ticket tende a ser assimétrica (cauda à direita) — típico de vendas com mix de eletrônicos.
- Há correlação positiva entre Subtotal e Total (óbvio) e correlações leves entre ticket e frete dependendo do serviço.
- Lead times variam por serviço; `Same-Day` tem menor lead time médio, `Scheduled` intermediário, `Standard` maior variância.


## 3) Inferência estatística
Objetos de inferência computados no script:
- IC 95% para o ticket médio (t-interval)
  - Fórmula utilizada: t_{n-1} * SE, SE = s / sqrt(n)
  - Se a suposição de normalidade não se sustentar (Shapiro test), sugerir bootstrap para IC.
- IC 95% para proporção de atrasos (is_late)
  - Intervalo aproximado normal: p ± 1.96 * sqrt(p(1-p)/n). Para p próximo a 0 ou 1 ou n pequeno, usar intervalo exato (Clopper-Pearson).

Verificações de suposições:
- Normalidade: test Shapiro sobre uma amostra (o script mostra p-valor). Para amostras grandes, o teste é sensível; também recomendamos inspecionar histograma & QQ-plot.
- Independência: assumir independência entre pedidos (validação adicional: verificar regras de negócio — p.ex. múltiplos pedidos do mesmo cliente sequenciais podem violar independência temporal).
- Homocedasticidade: se comparar médias entre grupos (ex.: lead time entre serviços), usar Levene/Bartlett conforme adequado.

Recomendações: quando suposições violadas, usar métodos não paramétricos (bootstrap para médias, testes de Mann-Whitney/Wilcoxon para comparações pairwise).


## 4) KPIs & insights (exemplos e interpretação)
KPIs calculáveis com os artefatos:
- Receita total / mês — monitorar tendência e sazonalidade (gráfico `monthly_revenue.png`).
- Ticket médio (IC 95% fornecido) — sensível a promoções; acompanhar por canal e categoria.
- Desconto médio (%) — comparar desconto médio por categoria para avaliar elasticidade.
- Take-rate de frete: freight_total / revenue_total (também calculável por serviço) — importante para pricing.
- Prazo de entrega (lead time) — monitorar média e % de atraso (is_late).
- Conversão por payment: taxa Confirmado / total por método — identificar problemas em meios de pagamento.
- Performance logística por Service: média lead time, % de atraso, freight average.
- Mix por Category/Subcategory e elasticidade vs desconto — require join with order_items -> products to compute revenue per category; placeholder SQL included.

Exemplos de insights:
- Se `Same-Day` tem alta % de atraso e custo de frete muito alto, avaliar trade-off entre satisfação e margem — possivelmente reduzir oferta ou reprecificar.
- Se conversão via `PIX` é consistentemente mais baixa que `Credito`, investigar falhas (UX / gateway) e quantificar perda de receita.
- Se desconto médio aumenta ticket mas reduz margem (take-rate de frete fixa), calcular elasticidade de preço: %Δquantidade / %Δpreço — requer experimento/control.


## 5) Reprodutibilidade
Arquivos fornecidos:
- `analysis.py` — script Python que executa o pipeline e salva `cleaned_orders.csv` + figures.
- `analysis.sql` — SQL para criar uma view `vw_orders_clean` (assume joins por `Id`) e consultas KPI.
- `requirements.txt` — dependências Python.

Como rodar (exemplo local):
```bash
# criar venv (macOS zsh)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```
Isso gerará `figures/` e `cleaned_orders.csv`.


## 6) Limitações e próximos passos
- Chaves relacionais: assumimos `Id` comum; se no DWH houver chaves diferentes (order_id, delivery_id, order_item.order_id) ajustar joins. A SQL inclui comentários para facilitar.
- Elasticidade preço vs desconto: para estimar elasticidade precisar de variação exógena (promoções ou A/B). Recomendado: organizar um teste controlado ou usar painel temporal por produto com modelos de fixed effects.
- Sugestões técnicas: adicionar testes unitários para checagem de integridade, adicionar notebook com células interativas e outputs in-line, e criar DAG (Airflow/Prefect) para produção.


## Anexos
- `analysis.sql` — SQL de reprodução
- `analysis.py` — pipeline Python
- `requirements.txt` — dependências


---
Relatório gerado com base nos arquivos CSV presentes na pasta do projeto. Para adaptações (por exemplo, chaves relacionais diferentes ou esquema de banco de dados), posso ajustar o SQL e o script para mapear os relacionamentos corretos.  
