-- Q4: Revenue and customer count by country (UK vs non-UK)
-- Supports RQ1: empirical justification for the IsUK feature
-- used in every downstream model.
SELECT
    Country,
    COUNT(DISTINCT CustomerID)                AS n_customers,
    COUNT(DISTINCT Invoice)                   AS n_invoices,
    ROUND(SUM(Quantity * Price), 2)           AS total_revenue,
    ROUND(SUM(Quantity * Price)
          / NULLIF(COUNT(DISTINCT CustomerID), 0), 2) AS revenue_per_customer,
    ROUND(100.0 * SUM(Quantity * Price)
          / SUM(SUM(Quantity * Price)) OVER (), 3)    AS pct_of_revenue,
    CASE WHEN Country = 'United Kingdom' THEN 'UK' ELSE 'Non-UK' END AS region
FROM orders
GROUP BY Country
ORDER BY total_revenue DESC;
