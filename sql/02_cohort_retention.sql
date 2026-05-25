-- Q2: Cohort retention matrix
-- Supports RQ2: validates the time-aware framing by quantifying how
-- much of each first-purchase cohort returns in subsequent months.
-- Output: rows = cohort_month (first-purchase month),
--         cols = observation_month, value = unique active customers.
WITH first_purchase AS (
    SELECT
        CustomerID,
        DATE_TRUNC('month', MIN(InvoiceDate))::DATE AS cohort_month
    FROM orders
    GROUP BY CustomerID
),
monthly_activity AS (
    SELECT DISTINCT
        CustomerID,
        DATE_TRUNC('month', InvoiceDate)::DATE AS obs_month
    FROM orders
)
SELECT
    fp.cohort_month,
    ma.obs_month,
    DATEDIFF('month', fp.cohort_month, ma.obs_month) AS month_offset,
    COUNT(DISTINCT ma.CustomerID)                    AS active_customers
FROM first_purchase fp
JOIN monthly_activity ma
  ON fp.CustomerID = ma.CustomerID
GROUP BY fp.cohort_month, ma.obs_month
ORDER BY fp.cohort_month, ma.obs_month;
