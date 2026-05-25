-- Q6: Monthly active customers (MAC) + monthly revenue
-- Supports RQ2: motivates the use of monthly sequence features by
-- exposing the temporal heterogeneity of the customer base.
SELECT
    DATE_TRUNC('month', InvoiceDate)::DATE AS year_month,
    COUNT(DISTINCT CustomerID)             AS active_customers,
    COUNT(DISTINCT Invoice)                AS n_invoices,
    ROUND(SUM(Quantity * Price), 2)        AS revenue,
    ROUND(SUM(Quantity * Price)
          / NULLIF(COUNT(DISTINCT CustomerID), 0), 2) AS revenue_per_active
FROM orders
GROUP BY DATE_TRUNC('month', InvoiceDate)
ORDER BY year_month;
