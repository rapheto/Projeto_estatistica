Projeto: Análise exploratória e inferência — E-commerce

Este repositório contém um pipeline reproducível em Python que carrega os CSVs de origem, aplica limpeza e feature engineering, produz análises descritivas e inferência estatística, e gera artefatos (CSV, figuras, relatórios). Recentemente o script foi aprimorado para:

- Imprimir logs em português (pt-BR) com cabeçalhos claros.
- Tratar corretamente o mapeamento entre itens de pedido (`DIM_Shopping`) e o catálogo (`DIM_Products`) por nome do produto.
- Salvar itens de pedido que não encontrarem correspondência no catálogo em `e-commerce_projeto_est/unmatched_order_items.csv` para auditoria.
- Gerar visualizações de inferência (ticket médio com IC 95% e proporção de pedidos atrasados com IC) em `figures/`.

Conteúdo principal
- `FACT_Orders.csv`, `DIM_Delivery.csv`, `DIM_Customer.csv`, `DIM_Products.csv`, `DIM_Shopping.csv` — dados fonte (CSV)
- `analysis.py` — script Python que executa a pipeline (ETL → EDA → inferência) e salva artefatos
- `analysis.sql` — SQL (Postgres-style) que reproduz a view `vw_orders_clean` e consultas de KPI
- `report.md` — relatório analítico (sumário, observações e próximos passos)
- `requirements.txt` — dependências Python

Principais artefatos gerados
- `e-commerce_projeto_est/cleaned_orders.csv` — tabela de pedidos com features (lead time, atraso, ticket, etc.)
- `e-commerce_projeto_est/data_quality_report.md` — relatório rápido de qualidade de dados
- `e-commerce_projeto_est/outliers.csv` — linhas marcadas como outliers pela regra IQR/z-score
- `e-commerce_projeto_est/monthly_revenue.csv` — série de receita mensal (para auditoria)
- `e-commerce_projeto_est/unmatched_order_items.csv` — itens de pedido que não mapearam ao catálogo (útil para corrigir nomes / aplicar fuzzy matching)
- `e-commerce_projeto_est/figures/` — imagens geradas (ex.: `ticket_mean_ci.png`, `late_rate_ci.png`, `category_revenue_bar.png`, etc.)

Como executar (macOS / zsh)

1) Criar e ativar venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Rodar a análise

```bash
python analysis.py
```

Saída esperada (no console)
- Cabeçalhos em português com KPIs e tabelas legíveis (ex.: "KPIs", "Conversão por Forma de Pagamento", "Mix de Receita por Categoria de Produto", "Preço x Desconto (Elasticidade Simplificada)").
- Mensagens que indicam onde os arquivos foram gravados (ex.: `data_quality_report.md`, `outliers.csv`, imagens em `figures/`).

Notas e dicas de auditoria
- Se o `Mix de Receita por Categoria` mostrar apenas uma categoria (por exemplo "Eletrônicos"), verifique `DIM_Products.csv` — o script também loga as categorias únicas do catálogo.
- Se houver itens não mapeados, abra `e-commerce_projeto_est/unmatched_order_items.csv` para revisar e decidir entre: correção manual dos nomes, regras de normalização (trim/lower), ou fuzzy matching.

Próximos passos recomendados
- Rodar uma rotina de fuzzy matching (ex.: `fuzzywuzzy` ou `rapidfuzz`) para sugerir correspondências entre `DIM_Shopping.Product` e `DIM_Products.Product_Name` — útil quando houver diferenças pequenas de ortografia ou formatação.
- Integrar `rich` para saída de console mais bonita (opcional) — se quiser, eu adiciono ao `requirements.txt` e ajusto o script.
- Gerar um relatório final em Markdown que inclua automaticamente as imagens geradas e as tabelas de resumo (posso implementar isso automaticamente).

Problemas conhecidos
- O CSV de entrada deve manter as colunas esperadas (ex.: `Id`, `Order_Date`, `Subtotal`, `Total` em `FACT_Orders.csv`; `Product` em `DIM_Shopping.csv`; `Product_Name` em `DIM_Products.csv`). Se suas chaves forem diferentes, adapte as junções no script/SQL.

Contato
- Se quiser que eu aplique fuzzy matching automático ou inclua as tabelas/figuras no `report.md`, diga qual opção prefere e eu faço a alteração.

---
Atualizado: versão local do script (logs em pt-BR e melhorias de produto) — execute `python analysis.py` para gerar os artefatos.
