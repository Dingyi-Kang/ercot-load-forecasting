-- Evaluation from a predictions file (one row per issue x target interval).
-- Grain check first: exactly one prediction per (issue_utc, interval_start_utc) for a run.
SELECT COUNT(*) AS n, COUNT(DISTINCT (issue_utc, interval_start_utc)) AS n_distinct FROM preds;

SELECT fold,
       COUNT(*)                         AS n,
       AVG(ABS(pred - y))               AS mae_mw,
       SQRT(AVG((pred - y) * (pred - y))) AS rmse_mw,
       100 * AVG(ABS(pred - y) / y)     AS mape_pct,
       AVG(pred - y)                    AS bias_mw
FROM preds GROUP BY ROLLUP (fold) ORDER BY fold NULLS LAST;
