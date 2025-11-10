-- analysis.sql
-- SQL to create a cleaned order-level view and compute key KPIs.
-- ASSUMPTION: the tables/files share a common "Id" surrogate key aligning rows (replace joins with real keys if you have them).

-- 1) Create a cleaned order-level view
CREATE VIEW vw_orders_clean AS
SELECT
  o."Id" as order_id,
  o."Order_Date"::timestamp as order_date,
  o."Subtotal" as subtotal,
  o."Total" as total,
  o."Discount" as discount_pct,
  o."payment" as payment_method,
  o."Purchase_Status" as purchase_status,
  d."Delivery_Id" as delivery_id,
  d."Services" as service,
  d."P_Sevice" as freight_amt,
  d."D_Forecast"::timestamp as d_forecast,
  d."D_Date"::timestamp as d_date,
  c."Customer_Id" as customer_id,
  c."State" as state,
  c."Region" as region,
  EXTRACT(EPOCH FROM (d."D_Date"::timestamp - d."D_Forecast"::timestamp))/86400.0 as delivery_delay_days,
  EXTRACT(EPOCH FROM (d."D_Date"::timestamp - o."Order_Date"::timestamp))/86400.0 as delivery_lead_time,
  CASE WHEN d."D_Date"::timestamp > d."D_Forecast"::timestamp THEN 1 ELSE 0 END as is_late,
  CASE WHEN lower(o."Purchase_Status") = 'confirmado' THEN 1 ELSE 0 END as is_confirmed,
  (d."P_Sevice" / NULLIF(o."Total",0))::numeric as freight_share,
  (o."Discount" * o."Subtotal") as discount_abs
FROM FACT_Orders o
LEFT JOIN DIM_Delivery d ON o."Id" = d."Id"
LEFT JOIN DIM_Customer c ON o."Id" = c."Id";

-- 2) KPI aggregations: monthly revenue, average ticket, freight take-rate
-- monthly revenue
SELECT date_trunc('month', order_date) as month,
       sum(total) as revenue,
       avg(total) as avg_ticket,
       sum(freight_amt) as freight_total,
       avg(discount_pct) as avg_discount_pct
FROM vw_orders_clean
GROUP BY 1
ORDER BY 1;

-- freight take-rate by service
SELECT service,
       count(*) as n_orders,
       sum(freight_amt) as freight_total,
       sum(total) as revenue_total,
       sum(freight_amt)/NULLIF(sum(total),0) as freight_take_rate
FROM vw_orders_clean
GROUP BY 1
ORDER BY freight_take_rate desc;

-- conversion rate by payment method
SELECT payment_method,
       sum(is_confirmed) as confirmed_count,
       count(*) as total_count,
       sum(is_confirmed)::float / NULLIF(count(*),0) as conversion_rate
FROM vw_orders_clean
GROUP BY 1
ORDER BY conversion_rate desc;

-- logistics: mean lead time and late % by service
SELECT service,
       avg(delivery_lead_time) as mean_lead_days,
       avg(is_late) as pct_late
FROM vw_orders_clean
GROUP BY 1
ORDER BY pct_late desc;

-- category mix and elasticity analysis
-- Create a view for order items, joining orders, shopping items, and products
CREATE VIEW vw_order_items AS
SELECT
  o."Id" as order_id,
  o."Order_Date"::timestamp as order_date,
  s."Product_Id" as product_id,
  s."Quantity" as quantity,
  s."Price" as price,
  s."Discount" as item_discount_pct,
  p."Category" as category,
  p."Subcategory" as subcategory,
  (s."Quantity" * s."Price") as item_revenue,
  (s."Quantity" * s."Price" * s."Discount") as item_discount_abs
FROM FACT_Orders o
JOIN DIM_Shopping s ON o."Id" = s."Id" -- Assuming "Id" is the order identifier
-- DIM_Shopping contains product names in column "Product"; DIM_Products has product metadata in "Product_Name"
JOIN DIM_Products p ON s."Product" = p."Product_Name";

-- Revenue mix by category
SELECT
  category,
  SUM(item_revenue) as total_revenue,
  SUM(quantity) as total_items_sold,
  SUM(item_revenue) / SUM(SUM(item_revenue)) OVER () as revenue_share
FROM vw_order_items
GROUP BY category
ORDER BY total_revenue DESC;

-- Simplified elasticity: compare avg price and quantity for items with/without discount
SELECT
  CASE WHEN item_discount_pct > 0 THEN 'With Discount' ELSE 'Without Discount' END as discount_status,
  AVG(price) as avg_price,
  SUM(quantity) as total_quantity_sold
FROM vw_order_items
GROUP BY 1;
