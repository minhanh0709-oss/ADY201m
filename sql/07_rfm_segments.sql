-- Q7: RFM segmentation — quintile scores + named customer segments
-- Supports Project 8 requirement: "Phân khúc RFM groups"
-- Scoring mirrors dashboard/_make_predictions.py (R+F+M quintile sum):
--   Champion (>=13), Loyal (10-12), At-Risk (7-9), Lost (5-6), New (<5)
WITH snap AS (
    SELECT DATE '2011-12-10' AS snapshot_date
),
customer_rfm AS (
    SELECT
        CustomerID,
        DATEDIFF('day', MAX(InvoiceDate)::DATE,
                 (SELECT snapshot_date FROM snap))           AS Recency,
        COUNT(DISTINCT Invoice)                              AS Frequency,
        ROUND(SUM(Quantity * Price), 2)                      AS Monetary,
        ROUND(SUM(Quantity * Price)
              / NULLIF(COUNT(DISTINCT Invoice), 0), 2)       AS AvgOrderValue,
        MODE() WITHIN GROUP (ORDER BY Country)               AS Country
    FROM orders
    GROUP BY CustomerID
    HAVING SUM(Quantity * Price) > 0
),
scored AS (
    SELECT
        *,
        6 - NTILE(5) OVER (ORDER BY Recency ASC)             AS R_score,
        NTILE(5) OVER (ORDER BY Frequency ASC)               AS F_score,
        NTILE(5) OVER (ORDER BY Monetary ASC)                AS M_score
    FROM customer_rfm
),
segmented AS (
    SELECT
        *,
        R_score + F_score + M_score                          AS RFM_score,
        CASE
            WHEN R_score + F_score + M_score >= 13 THEN 'Champion'
            WHEN R_score + F_score + M_score >= 10 THEN 'Loyal'
            WHEN R_score + F_score + M_score >= 7  THEN 'At-Risk'
            WHEN R_score + F_score + M_score >= 5  THEN 'Lost'
            ELSE 'New'
        END                                                  AS RFM_Segment
    FROM scored
)
SELECT
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    AvgOrderValue,
    Country,
    R_score,
    F_score,
    M_score,
    RFM_score,
    RFM_Segment
FROM segmented
ORDER BY Monetary DESC;
