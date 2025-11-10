"""
analysis.py
Repositório reproduzível: carrega os CSVs do projeto, faz limpeza básica, feature
engineering, EDA e inferência estatística. Gera um CSV limpo e figuras em
`figures/`.

Como usar:
  python analysis.py

O script detecta automaticamente uma subpasta `e-commerce_projeto_est` (onde
normalmente estão os CSVs). Se os arquivos estiverem na mesma pasta do script,
eles também serão encontrados.
"""

from pathlib import Path
import sys
import warnings
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def find_data_dir(root: Path) -> Path:
	"""Retorna o diretório onde os CSVs estão (procura subpasta 'e-commerce_projeto_est')."""
	candidate = root / 'e-commerce_projeto_est'
	if candidate.exists() and candidate.is_dir():
		return candidate
	return root


def safe_read_csv(path: Path, **kwargs):
	if not path.exists():
		warnings.warn(f"Arquivo não encontrado: {path}")
		return None
	return pd.read_csv(path, **kwargs)


def fmt_currency(x):
	try:
		return f"R$ {x:,.2f}"
	except Exception:
		return x


def _log_section(title: str):
	logging.info('\n' + '=' * 8 + f' {title} ' + '=' * 8)


def print_kpis(merged: pd.DataFrame):
	"""Pretty-print a compact KPI summary to the console."""
	kpis = {}
	kpis['n_orders'] = len(merged)
	if 'Total' in merged:
		kpis['revenue_total'] = merged['Total'].sum()
		kpis['ticket_mean'] = merged['Total'].mean()
	if 'Subtotal' in merged:
		kpis['subtotal_total'] = merged['Subtotal'].sum()
	if 'P_Sevice' in merged.columns:
		kpis['freight_total'] = merged['P_Sevice'].sum()

	# build pretty lines
	lines = [f"{'Métrica':<28} {'Valor':>18}", '-' * 46]
	# friendly Portuguese labels
	label_map = {
		'n_orders': 'Número de pedidos',
		'revenue_total': 'Receita total',
		'ticket_mean': 'Ticket médio',
		'subtotal_total': 'Subtotal total',
		'freight_total': 'Total de frete'
	}
	for key, val in kpis.items():
		label = label_map.get(key, key.replace('_', ' ').title())
		if key in ['revenue_total', 'subtotal_total', 'freight_total', 'ticket_mean']:
			display = fmt_currency(val)
		elif isinstance(val, (int, np.integer)):
			display = f"{int(val):,}"
		elif isinstance(val, float):
			display = f"{val:,.2f}"
		else:
			display = str(val)
		lines.append(f"{label:<28} {display:>18}")

	_log_section('KPIs')
	logging.info('\n'.join(lines))


def generate_data_quality_report(merged: pd.DataFrame, orders: pd.DataFrame, delivery: pd.DataFrame, customers: pd.DataFrame, shopping: pd.DataFrame, data_dir: Path, remove_duplicates: bool = False):
	"""Gera relatório de qualidade de dados:
	- Aplica trimming em colunas texto
	- Conta NAs por coluna
	- Detecta duplicatas por Id (flag, não remove por padrão)
	- Checa integridade referencial básica (orders x delivery x customers)
	- Detecta outliers (IQR e z-score) nas colunas numéricas e salva CSVs
	- Salva um markdown resumido e arquivos auxiliares em data_dir
	"""
	report_lines = []
	n_rows = len(merged)
	report_lines.append(f"# Data quality report\n\nRows analyzed: {n_rows}\n")

	# Trim string columns in place for a few dataframes
	def _trim_df(df: pd.DataFrame):
		for c in df.select_dtypes(include=['object']).columns:
			df[c] = df[c].astype(str).str.strip()

	try:
		_trim_df(merged)
		if customers is not None:
			_trim_df(customers)
		if delivery is not None:
			_trim_df(delivery)
		if shopping is not None:
			_trim_df(shopping)
	except Exception as e:
		logging.warning('Trimming warning: %s', e)

	# Nulls
	nulls = merged.isnull().sum().sort_values(ascending=False)
	null_pct = (nulls / max(1, n_rows) * 100).round(2)
	report_lines.append('## Missing values (top)\n')
	report_lines.append('|column|n_nulls|pct_null|')
	report_lines.append('|---:|---:|---:|')
	for col in nulls.index[:50]:
		report_lines.append(f"|{col}|{int(nulls[col])}|{null_pct[col]}%|")

	# Dtypes
	dtypes = merged.dtypes
	report_lines.append('\n## Column types\n')
	report_lines.append('|column|dtype|')
	report_lines.append('|---|---|')
	for col, dt in dtypes.items():
		report_lines.append(f"|{col}|{dt}|")

	# Duplicates
	dup_mask = merged.duplicated(subset=['Id'], keep=False) if 'Id' in merged.columns else pd.Series([False]*n_rows)
	n_dup = int(dup_mask.sum())
	report_lines.append(f"\n## Duplicates\nTotal duplicated rows (by Id): {n_dup}\n")
	if n_dup:
		dup_df = merged[dup_mask]
		dup_path = data_dir / 'duplicated_rows.csv'
		dup_df.to_csv(dup_path, index=False)
		report_lines.append(f"Duplicated rows saved to `{dup_path.name}`\n")
		if remove_duplicates:
			merged.drop_duplicates(subset=['Id'], inplace=True)

	# Referential integrity basics
	report_lines.append('\n## Referential integrity checks\n')
	if delivery is not None and 'Id' in delivery.columns and 'Id' in orders.columns:
		orders_ids = set(orders['Id'].dropna().unique())
		delivery_ids = set(delivery['Id'].dropna().unique())
		orphan_deliveries = sorted(list(delivery_ids - orders_ids))[:10]
		orphan_orders = sorted(list(orders_ids - delivery_ids))[:10]
		report_lines.append(f"Orders without delivery (sample up to 10): {orphan_orders}\n")
		report_lines.append(f"Deliveries without order (sample up to 10): {orphan_deliveries}\n")

	# Outliers: IQR and z-score
	num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
	# exclude index-like and counters if present
	exclude = ['Id']
	num_cols = [c for c in num_cols if c not in exclude]
	outlier_flags = pd.DataFrame(index=merged.index)
	outlier_flags['any_iqr'] = False
	outlier_flags['any_z'] = False
	for c in num_cols:
		col = merged[c].dropna()
		if col.shape[0] < 3:
			continue
		q1 = col.quantile(0.25)
		q3 = col.quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		mask_iqr = (merged[c] < lower) | (merged[c] > upper)
		# z-score
		try:
			z = np.abs(stats.zscore(merged[c].fillna(merged[c].mean())))
			mask_z = z > 3
		except Exception:
			mask_z = pd.Series([False]*len(merged), index=merged.index)
		outlier_flags['any_iqr'] = outlier_flags['any_iqr'] | mask_iqr.fillna(False)
		outlier_flags['any_z'] = outlier_flags['any_z'] | mask_z

	outliers_any = outlier_flags[(outlier_flags['any_iqr']) | (outlier_flags['any_z'])]
	report_lines.append(f"\n## Outliers\nNumeric columns checked: {len(num_cols)}\nTotal rows flagged as outlier (IQR or z): {len(outliers_any)}\n")
	if len(outliers_any):
		out_df = merged.loc[outliers_any.index].copy()
		out_df['_outlier_iqr'] = outlier_flags.loc[outliers_any.index, 'any_iqr']
		out_df['_outlier_z'] = outlier_flags.loc[outliers_any.index, 'any_z']
		out_path = data_dir / 'outliers.csv'
		out_df.to_csv(out_path, index=False)
		report_lines.append(f"Outliers saved to `{out_path.name}`\n")

	# Save monthly revenue CSV for audit
	if 'Order_Date' in merged and 'Total' in merged:
		rev_month = merged.set_index('Order_Date').resample('ME')['Total'].sum()
		rev_path = data_dir / 'monthly_revenue.csv'
		rev_month.to_csv(rev_path, header=['revenue'])
		report_lines.append(f"Monthly revenue series saved to `{rev_path.name}`\n")

	# Save report
	report_path = data_dir / 'data_quality_report.md'
	report_text = '\n'.join(report_lines)
	report_path.write_text(report_text)
	logging.info('Data quality report written to %s', report_path)
	return {
		'report_path': report_path,
		'duplicates_path': dup_path if n_dup else None,
		'outliers_path': out_path if len(outliers_any) else None,
		'monthly_revenue_path': rev_path if 'rev_path' in locals() else None
	}


def main():
	ROOT = Path(__file__).parent
	DATA_DIR = find_data_dir(ROOT)
	FIG_DIR = DATA_DIR / 'figures'
	FIG_DIR.mkdir(exist_ok=True)

	# configure logging for prettier console output
	logging.basicConfig(level=logging.INFO, format='%(message)s')
	logging.info(f"Usando diretório de dados: {DATA_DIR}")

	# Carregar arquivos
	orders = safe_read_csv(DATA_DIR / 'FACT_Orders.csv', parse_dates=['Order_Date'])
	delivery = safe_read_csv(DATA_DIR / 'DIM_Delivery.csv', parse_dates=['D_Forecast','D_Date'])
	customers = safe_read_csv(DATA_DIR / 'DIM_Customer.csv')
	products = safe_read_csv(DATA_DIR / 'DIM_Products.csv')
	shopping = safe_read_csv(DATA_DIR / 'DIM_Shopping.csv')

	# Log unique categories in the product catalog (quick sanity check)
	if products is not None and 'Category' in products.columns:
		try:
			cat_counts = products['Category'].value_counts(dropna=False)
			_log_section('Categorias no catálogo de produtos')
			for cat, cnt in cat_counts.items():
				logging.info('  %s: %s', str(cat), f"{int(cnt):,}")
		except Exception as e:
			logging.warning('Erro ao contar categorias: %s', e)

	if orders is None:
		print('Arquivo FACT_Orders.csv não encontrado. Saindo.')
		sys.exit(1)

	# Inspeção básica
	logging.info('\nColumns:')
	logging.info(' Orders: %s', orders.columns.tolist())
	if delivery is not None:
		logging.info(' Delivery: %s', delivery.columns.tolist())
	if customers is not None:
		logging.info(' Customers: %s', customers.columns.tolist())
	if products is not None:
		logging.info(' Products: %s', products.columns.tolist())
	if shopping is not None:
		logging.info(' Shopping: %s', shopping.columns.tolist())

	# Merge (assunção: Id alinhado)
	merged = orders.copy()
	if delivery is not None and 'Id' in delivery.columns:
		merged = merged.merge(delivery, on='Id', how='left', suffixes=('','_del'))

	if customers is not None and 'Id' in customers.columns:
		# manter colunas de cliente úteis
		keep = ['Id','Customer_Id','City','State','Region']
		keep_present = [c for c in keep if c in customers.columns]
		merged = merged.merge(customers[keep_present], on='Id', how='left')

	# Feature engineering
	logging.info('\nFeature engineering...')
	# converte colunas de data caso não estejam como datetime
	for col in ['D_Date','D_Forecast']:
		if col in merged.columns and not pd.api.types.is_datetime64_any_dtype(merged[col]):
			merged[col] = pd.to_datetime(merged[col], errors='coerce')

	merged['delivery_delay_days'] = (merged.get('D_Date') - merged.get('D_Forecast')) / pd.Timedelta(days=1)
	merged['delivery_lead_time'] = (merged.get('D_Date') - merged.get('Order_Date')) / pd.Timedelta(days=1)
	merged['is_late'] = (merged['D_Date'] > merged['D_Forecast']).astype('Int64') if 'D_Date' in merged and 'D_Forecast' in merged else pd.NA
	merged['is_confirmed'] = merged['Purchase_Status'].str.lower().eq('confirmado').astype('Int64') if 'Purchase_Status' in merged else pd.NA
	# freight share: P_Sevice / Total
	if 'P_Sevice' in merged.columns and 'Total' in merged.columns:
		merged['freight_share'] = merged['P_Sevice'] / merged['Total'].replace({0: np.nan})
	else:
		merged['freight_share'] = pd.NA

	if 'Discount' in merged and 'Subtotal' in merged:
		merged['discount_abs'] = merged['Discount'] * merged['Subtotal']

	if 'Order_Date' in merged:
		merged['order_month'] = merged['Order_Date'].dt.to_period('M')

	# Data quality checks (trimming, NAs, duplicates, referential checks, outliers)
	logging.info('\nRunning data quality checks...')
	dq = generate_data_quality_report(merged, orders, delivery, customers, shopping, DATA_DIR, remove_duplicates=False)
	if dq.get('report_path'):
		logging.info('Relatório de qualidade de dados: %s', dq['report_path'].name)
	if dq.get('duplicates_path'):
		logging.info('Linhas duplicadas salvas: %s', dq['duplicates_path'].name)
	if dq.get('outliers_path'):
		logging.info('Outliers salvos: %s', dq['outliers_path'].name)

	# KPIs básicos (formatado)
	print_kpis(merged)

	# Conversion by payment
	if 'payment' in merged and 'Purchase_Status' in merged:
		conv = merged.groupby('payment')['Purchase_Status'].value_counts().unstack(fill_value=0)
		if 'Confirmado' in conv.columns:
			conv['conversion_rate_confirmado'] = conv['Confirmado'] / conv.sum(axis=1)
		# format for display
		conv_display = conv.copy()
		if 'conversion_rate_confirmado' in conv_display.columns:
			conv_display['conversion_rate_confirmado'] = (conv_display['conversion_rate_confirmado'] * 100).map('{:.1f}%'.format)
		_log_section('Conversão por Forma de Pagamento (top)')
		# rename conversion column for display if present
		disp = conv_display.copy()
		if 'conversion_rate_confirmado' in disp.columns:
			disp = disp.rename(columns={'conversion_rate_confirmado': 'Taxa de conversão (Confirmado)'})
		logging.info('\n%s', disp.sort_values('Taxa de conversão (Confirmado)', ascending=False).to_string())

	# Logistics performance by Services
	if 'Services' in merged:
		log_perf = merged.groupby('Services').agg(
			n_orders=('Id','count'),
			mean_lead_time=('delivery_lead_time','mean'),
			pct_late=('is_late','mean'),
			avg_freight=('P_Sevice','mean')
		).reset_index()
		lp = log_perf.copy()
		if 'mean_lead_time' in lp.columns:
			lp['mean_lead_time'] = lp['mean_lead_time'].map(lambda x: f"{x:.1f} d")
		if 'pct_late' in lp.columns:
			lp['pct_late'] = (lp['pct_late'] * 100).map('{:.1f}%'.format)
		if 'avg_freight' in lp.columns:
			lp['avg_freight'] = lp['avg_freight'].map(fmt_currency)
		_log_section('Desempenho Logístico por Serviço')
		# human-friendly column names
		disp_lp = lp.rename(columns={
			'Services': 'Serviço',
			'n_orders': 'Pedidos',
			'mean_lead_time': 'Prazo médio',
			'pct_late': '% Atraso',
			'avg_freight': 'Frete médio'
		})
		logging.info('\n%s', disp_lp.to_string(index=False))

	# --- Product-level analysis ---
	if shopping is not None and products is not None:
		logging.info('\n--- Análise de Produtos ---')
		# Create a new df for order items. Use the original orders table for the base.
		# include order-level Discount so we can evaluate whether the item had a discount
		order_items = orders[['Id', 'Order_Date', 'Discount']].merge(shopping, on='Id', how='inner')
		# DIM_Shopping stores product NAMES in column 'Product'. DIM_Products has 'Product_Name' and Product_Id.
		# Join by product name so we can get category and product metadata. If you have a product id column in shopping,
		# prefer joining on that for better reliability.
		# Merge products: avoid silent column collisions by using explicit suffixes.
		# prefer the price coming from the shopping/item row (left), and keep product metadata on the right.
		order_items = order_items.merge(products, left_on='Product', right_on='Product_Name', how='left', suffixes=('_shop', '_prod'))

		# Normalize price column: shopping.Price (price charged at item level) is preferred.
		if 'Price_shop' in order_items.columns:
			order_items['Price'] = order_items['Price_shop']
		elif 'Price' in order_items.columns:
			# fallback (unlikely because of suffixing) - keep existing
			pass
		elif 'Price_prod' in order_items.columns:
			# fallback to product catalog price if shopping price missing
			order_items['Price'] = order_items['Price_prod']

		# Calculate item-level revenue and check for discounts
		order_items['item_revenue'] = order_items['Quantity'] * order_items['Price']
		order_items['has_discount'] = order_items['Discount'] > 0

		# 1. Category Mix
		cat_mix = order_items.groupby('Category').agg(
			revenue=('item_revenue', 'sum'),
			n_items=('Quantity', 'sum')
		).sort_values('revenue', ascending=False)

		# Report unmapped items (after merge) so user can see if product names didn't match
		if order_items['Category'].isnull().any():
			unmatched = order_items[order_items['Category'].isnull()].copy()
			unmatched_path = DATA_DIR / 'unmatched_order_items.csv'
			try:
				unmatched.to_csv(unmatched_path, index=False)
				logging.info('Itens de pedido sem mapeamento para catálogo salvos em: %s (ex.: %d linhas)', unmatched_path.name, len(unmatched))
			except Exception as e:
				logging.warning('Não foi possível salvar unmatched_order_items.csv: %s', e)
		total_revenue = cat_mix['revenue'].sum()
		if total_revenue > 0:
			cat_mix['revenue_share'] = cat_mix['revenue'] / total_revenue
		else:
			cat_mix['revenue_share'] = 0

		# Format for display
		cat_mix_display = cat_mix.copy()
		# format numeric columns
		cat_mix_display['Receita'] = cat_mix_display['revenue'].map(fmt_currency)
		cat_mix_display['Quantidade'] = cat_mix_display['n_items'].map(lambda x: f"{int(x):,}")
		if 'revenue_share' in cat_mix_display.columns:
			cat_mix_display['Participacao_receita'] = (cat_mix_display['revenue_share'] * 100).map('{:.1f}%'.format)
		else:
			cat_mix_display['Participacao_receita'] = ''
		# keep only friendly columns in desired order
		cat_mix_display = cat_mix_display[['Receita', 'Quantidade', 'Participacao_receita']]
		_log_section('Mix de Receita por Categoria de Produto')
		# index name in Portuguese
		cat_mix_display.index.name = 'Categoria'
		# rename columns for display (with spaces)
		cat_mix_display = cat_mix_display.rename(columns={
			'Receita': 'Receita',
			'Quantidade': 'Quantidade',
			'Participacao_receita': 'Participação da receita'
		})
		logging.info('\n%s', cat_mix_display.to_string())

		# 2. Simplified Elasticity Analysis
		# Compare avg price for items with and without discount
		elasticity_analysis = order_items.groupby('has_discount').agg(
			avg_price=('Price', 'mean'),
			total_quantity=('Quantity', 'sum')
		).reset_index()
		# Make display-friendly
		elasticity_display = elasticity_analysis.copy()
		# map boolean/int to labels
		elasticity_display['Discount Status'] = elasticity_display['has_discount'].map({True: 'Com desconto', False: 'Sem desconto'})
		if 'avg_price' in elasticity_display.columns:
			elasticity_display['Average Price'] = elasticity_display['avg_price'].map(lambda x: fmt_currency(x) if pd.notna(x) else '')
		if 'total_quantity' in elasticity_display.columns:
			elasticity_display['Total Quantity'] = elasticity_display['total_quantity'].map(lambda x: f"{int(x):,}")
		# select and order columns for neat output
		elasticity_display = elasticity_display[['Discount Status', 'Average Price', 'Total Quantity']]
		# translate column names to Portuguese for display
		elasticity_display = elasticity_display.rename(columns={
			'Discount Status': 'Status do desconto',
			'Average Price': 'Preço médio',
			'Total Quantity': 'Quantidade total'
		})
		# Build a human-friendly aligned table and ensure both rows exist (Com desconto / Sem desconto)
		rows = {}
		for _, r in elasticity_display.iterrows():
			status = r['Status do desconto']
			rows[status] = {
				'price': r.get('Preço médio', ''),
				'qty': r.get('Quantidade total', '')
			}
		# ensure both keys present
		for key in ['Com desconto', 'Sem desconto']:
			if key not in rows:
				rows[key] = {'price': '', 'qty': 0}

		# format lines
		head = f"{'Status do desconto':<18} {'Preço médio':>18} {'Quantidade':>14}"
		sep = '-' * len(head)
		lines = [head, sep]
		for key in ['Com desconto', 'Sem desconto']:
			price = rows[key]['price']
			qty = rows[key]['qty']
			# price: may already be a formatted string (from earlier steps) or numeric
			if price is None or (isinstance(price, str) and price.strip() == ''):
				price_disp = '-'
			elif isinstance(price, (int, float, np.integer, np.floating)):
				price_disp = fmt_currency(price)
			elif isinstance(price, str) and price.strip().startswith('R$'):
				price_disp = price
			else:
				# try to coerce numeric from string
				try:
					p = float(str(price).replace('R$', '').replace(',', ''))
					price_disp = fmt_currency(p)
				except Exception:
					price_disp = str(price)

			# qty: accept numeric or strings with thousands separator
			try:
				if isinstance(qty, (int, np.integer)):
					qty_disp = f"{int(qty):,}"
				elif isinstance(qty, float) or isinstance(qty, np.floating):
					qty_disp = f"{int(qty):,}"
				elif isinstance(qty, str):
					qty_clean = qty.replace(',', '').strip()
					qty_disp = f"{int(qty_clean):,}" if qty_clean not in ('', '0') else '0'
				else:
					qty_disp = '0'
			except Exception:
				qty_disp = '0'
			lines.append(f"{key:<18} {price_disp:>18} {qty_disp:>14}")
		_table = '\n'.join(lines)
		_log_section('Preço x Desconto (Elasticidade Simplificada)')
		logging.info('\n%s', _table)

		# New plots for product analysis
		try:
			# Category revenue bar plot
			if not cat_mix.empty:
				plt.figure(figsize=(10, 6))
				cat_mix['revenue'].sort_values(ascending=True).plot(kind='barh')
				plt.title('Total Revenue by Product Category')
				plt.xlabel('Total Revenue (BRL)')
				plt.ylabel('Category')
				plt.tight_layout()
				plt.savefig(FIG_DIR / 'category_revenue_bar.png')
				plt.close()

			# Price vs Discount boxplot
			if 'Price' in order_items.columns:
				plt.figure(figsize=(8, 5))
				sns.boxplot(data=order_items, x='has_discount', y='Price')
				plt.title('Price Distribution: With vs. Without Discount')
				plt.xlabel('Has Discount')
				plt.ylabel('Price (BRL)')
				plt.xticks([0, 1], ['No', 'Yes'])
				plt.tight_layout()
				plt.savefig(FIG_DIR / 'price_discount_box.png')
				plt.close()
		except Exception as e:
			warnings.warn(f"Erro ao gerar gráficos de produtos: {e}")

	# EDA plots
	sns.set(style='whitegrid')
	try:
		if 'Total' in merged:
			plt.figure(figsize=(8,5))
			sns.histplot(merged['Total'].dropna(), kde=True, bins=40)
			plt.title('Distribution of Order Ticket (Total)')
			plt.xlabel('Total (BRL)')
			plt.tight_layout()
			plt.savefig(FIG_DIR / 'ticket_hist.png')
			plt.close()

		if 'delivery_lead_time' in merged:
			plt.figure(figsize=(8,5))
			sns.boxplot(x=merged['delivery_lead_time'].dropna())
			plt.title('Lead time (days) - boxplot')
			plt.xlabel('Days')
			plt.tight_layout()
			plt.savefig(FIG_DIR / 'leadtime_box.png')
			plt.close()

		corr_cols = [c for c in ['Subtotal','Total','Discount','P_Sevice','delivery_lead_time','delivery_delay_days'] if c in merged.columns]
		if len(corr_cols) >= 2:
			plt.figure(figsize=(8,6))
			corr_df = merged[corr_cols].corr()
			sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm')
			plt.title('Correlation matrix')
			plt.tight_layout()
			plt.savefig(FIG_DIR / 'correlation_heatmap.png')
			plt.close()

		if 'Order_Date' in merged and 'Total' in merged:
			# use month-end resampling to avoid future deprecation warnings
			rev_month = merged.set_index('Order_Date').resample('ME')['Total'].sum()
			plt.figure(figsize=(10,4))
			rev_month.plot(marker='o')
			plt.ylabel('Revenue (BRL)')
			plt.title('Monthly Revenue')
			plt.tight_layout()
			plt.savefig(FIG_DIR / 'monthly_revenue.png')
			plt.close()
	except Exception as e:
		warnings.warn(f"Erro ao gerar gráficos: {e}")

	# Inference
	print('\nInferência estatística:')
	# --- Ticket médio (IC t) ---
	if 'Total' in merged:
		tickets = merged['Total'].dropna()
		if len(tickets) > 1:
			ticket_mean = tickets.mean()
			ticket_se = stats.sem(tickets)
			try:
				ticket_ci = stats.t.interval(0.95, df=len(tickets)-1, loc=ticket_mean, scale=ticket_se)
				# pretty print
				print(f"Ticket médio: {fmt_currency(ticket_mean)} | IC 95%: ({fmt_currency(ticket_ci[0])}, {fmt_currency(ticket_ci[1])})")
				# visualization: bar with error bar
				try:
					plt.figure(figsize=(4,6))
					plt.bar(0, ticket_mean, color='#2a9d8f', width=0.6)
					plt.errorbar(0, ticket_mean, yerr=[[ticket_mean - ticket_ci[0]], [ticket_ci[1] - ticket_mean]], fmt='none', ecolor='black', capsize=8)
					plt.xlim(-0.5,0.5)
					plt.xticks([])
					plt.ylabel('Valor (BRL)')
					plt.title('Ticket médio com IC 95%')
					# annotate values
					plt.text(0, ticket_mean, fmt_currency(ticket_mean), ha='center', va='bottom')
					plt.tight_layout()
					plt.savefig(FIG_DIR / 'ticket_mean_ci.png')
					plt.close()
					logging.info('Figura de inferência salva: %s', (FIG_DIR / 'ticket_mean_ci.png').name)
				except Exception as e:
					logging.warning('Não foi possível gerar figura do ticket: %s', e)
			except Exception:
				print('Não foi possível calcular o IC t (dados insuficientes).')

	# --- Proporção de pedidos atrasados (IC aproximado) ---
	if 'is_late' in merged:
		late = merged['is_late'].dropna()
		if len(late) > 0:
			n = len(late)
			k = int(late.sum())
			prop = k / n
			prop_se = np.sqrt(prop*(1-prop)/n)
			prop_ci = (prop - 1.96*prop_se, prop + 1.96*prop_se)
			print(f"Proporção de atrasos: {prop:.3%}, IC 95% (aprox.): ({prop_ci[0]:.3%}, {prop_ci[1]:.3%})")
			# visualization: proportion with CI (bar + error)
			try:
				plt.figure(figsize=(4,4))
				plt.bar(0, prop, color='#e76f51', width=0.6)
				plt.errorbar(0, prop, yerr=[[prop - prop_ci[0]], [prop_ci[1] - prop]], fmt='none', ecolor='black', capsize=8)
				plt.xlim(-0.5,0.5)
				plt.ylim(0,1)
				plt.xticks([])
				plt.ylabel('Proporção')
				plt.title('Proporção de pedidos atrasados (IC 95%)')
				plt.text(0, prop, f"{prop:.1%}", ha='center', va='bottom')
				plt.tight_layout()
				plt.savefig(FIG_DIR / 'late_rate_ci.png')
				plt.close()
				logging.info('Figura de inferência salva: %s', (FIG_DIR / 'late_rate_ci.png').name)
			except Exception as e:
				logging.warning('Não foi possível gerar figura da proporção de atrasos: %s', e)

	# Save cleaned dataset
	outpath = DATA_DIR / 'cleaned_orders.csv'
	merged.to_csv(outpath, index=False)
	print(f"Dataset limpo salvo em: {outpath}")


if __name__ == '__main__':
	main()

