-- Q1: RFM per customer (Recency / Frequency / Monetary)
-- Supports RQ1: baseline customer-value descriptor.
-- Snapshot date is set to the day after the last invoice in the dataset
-- so that Recency is non-negative for every customer.
WITH snap AS (
    SELECT DATE '2011-12-10' AS snapshot_date
)
SELECT
    CustomerID,
    DATEDIFF('day', MAX(InvoiceDate)::DATE,
             (SELECT snapshot_date FROM snap))           AS Recency,
    COUNT(DISTINCT Invoice)                              AS Frequency,
    ROUND(SUM(Quantity * Price), 2)                      AS Monetary,
    ROUND(SUM(Quantity * Price)
          / NULLIF(COUNT(DISTINCT Invoice), 0), 2)       AS AvgOrderValue,
    MIN(InvoiceDate)::DATE                               AS FirstPurchase,
    MAX(InvoiceDate)::DATE                               AS LastPurchase,
    MODE() WITHIN GROUP (ORDER BY Country)               AS Country
FROM orders
GROUP BY CustomerID
HAVING SUM(Quantity * Price) > 0
ORDER BY Monetary DESC;
