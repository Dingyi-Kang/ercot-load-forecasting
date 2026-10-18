-- Availability-aware load features for issue 18:00 America/Chicago on D = T-1.
-- Policy A0: only operating days <= T-2 are known at the issue. All joins are on operating date
-- arithmetic + local clock slot, NOT on "24 rows earlier" (LAG(...,24) breaks on DST/missing hours).

-- 1) daily slot matrix: one row per (op_date, slot 1..24)
CREATE OR REPLACE TEMP TABLE slot_matrix AS
WITH base AS (                      -- fall-back day: two HE02 rows are averaged into slot 2
    SELECT op_date, he AS slot, AVG(ERCOT) AS mw FROM load_hourly GROUP BY op_date, he
), spring_fill AS (                 -- spring-forward day: slot 3 does not exist; average of 2 and 4
    SELECT b2.op_date, 3 AS slot, (b2.mw + b4.mw) / 2 AS mw
    FROM base b2 JOIN base b4 ON b2.op_date = b4.op_date AND b2.slot = 2 AND b4.slot = 4
    WHERE NOT EXISTS (SELECT 1 FROM base b3 WHERE b3.op_date = b2.op_date AND b3.slot = 3)
)
SELECT * FROM base UNION ALL SELECT * FROM spring_fill;

-- 2) target rows (real intervals of day T) with lags taken from days T-k, k >= 2
CREATE OR REPLACE TEMP TABLE sql_features AS
SELECT t.interval_start_utc, t.op_date AS T, t.he AS slot, t.ERCOT AS y,
       l2.mw  AS lag_d2,
       l7.mw  AS lag_d7,
       d2.mean_mw AS d2_mean,
       d2.max_mw  AS d2_max,
       CAST(t.op_date AS DATE) - INTERVAL 2 DAY AS src_max_day
FROM load_hourly t
LEFT JOIN slot_matrix l2 ON l2.op_date = CAST(t.op_date AS DATE) - INTERVAL 2 DAY AND l2.slot = t.he
LEFT JOIN slot_matrix l7 ON l7.op_date = CAST(t.op_date AS DATE) - INTERVAL 7 DAY AND l7.slot = t.he
LEFT JOIN (SELECT op_date, AVG(mw) AS mean_mw, MAX(mw) AS max_mw FROM slot_matrix GROUP BY op_date) d2
       ON d2.op_date = CAST(t.op_date AS DATE) - INTERVAL 2 DAY
WHERE t.op_date >= DATE '2017-02-01';
