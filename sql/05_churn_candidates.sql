-- Q5: Churn candidates (Recency > 90 days at snapshot)
-- Supports RQ3: builds the retention campaign target list, segmented
-- by historical Monetary tier so marketing can prioritise high-value
-- at-risk customers.
WITH snap AS (
    SELECT DATE '2011-12-10' AS snapshot_date
),
customer_summary AS (
    SELECT
        CustomerID,
        DATEDIFF('day', MAX(InvoiceDate)::DATE,
                 (SELECT snapshot_date FROM snap)) AS recency_days,
        COUNT(DISTINCT Invoice)                    AS frequency,
        ROUND(SUM(Quantity * Price), 2)            AS monetary
    FROM orders
    GROUP BY CustomerID
),
tiered AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY monetary)          AS monetary_quartile,
        CASE
            WHEN NTILE(4) OVER (ORDER BY monetary) = 4 THEN 'Q4_High'
            WHEN NTILE(4) OVER (ORDER BY monetary) = 3 THEN 'Q3_MidHigh'
            WHEN NTILE(4) OVER (ORDER BY monetary) = 2 THEN 'Q2_MidLow'
            ELSE 'Q1_Low'
        END                                        AS monetary_tier
    FROM customer_summary
)
SELECT
    CustomerID,
    recency_days,
    frequency,
    monetary,
    monetary_tier,
    CASE
        WHEN recency_days BETWEEN 91  AND 180 THEN 'At-Risk'
        WHEN recency_days BETWEEN 181 AND 365 THEN 'Lapsed'
        ELSE 'Lost'
    END AS churn_segment
FROM tiered
WHERE recency_days > 90
ORDER BY monetary DESC;
