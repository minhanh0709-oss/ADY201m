-- Q3: Top-100 VIPs by historical Monetary
-- Supports RQ3: defines the marketing ground-truth VIP list that
-- predictive models are evaluated against (Revenue Capture @ Top-K%).
WITH customer_revenue AS (
    SELECT
        CustomerID,
        ROUND(SUM(Quantity * Price), 2) AS total_revenue,
        COUNT(DISTINCT Invoice)         AS n_invoices,
        MIN(InvoiceDate)::DATE          AS first_purchase,
        MAX(InvoiceDate)::DATE          AS last_purchase,
        MODE() WITHIN GROUP (ORDER BY Country) AS country
    FROM orders
    GROUP BY CustomerID
),
total_revenue AS (
    SELECT SUM(total_revenue) AS grand_total FROM customer_revenue
)
SELECT
    ROW_NUMBER() OVER (ORDER BY total_revenue DESC)    AS rank,
    CustomerID,
    total_revenue,
    n_invoices,
    first_purchase,
    last_purchase,
    country,
    ROUND(100.0 * total_revenue
          / (SELECT grand_total FROM total_revenue), 3) AS pct_of_total_revenue,
    ROUND(100.0 * SUM(total_revenue) OVER
          (ORDER BY total_revenue DESC
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          / (SELECT grand_total FROM total_revenue), 3) AS cumulative_pct
FROM customer_revenue
ORDER BY total_revenue DESC
LIMIT 100;
