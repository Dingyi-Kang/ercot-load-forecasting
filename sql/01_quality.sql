-- Data-quality report on the hourly load table (DuckDB dialect; not Snowflake).
-- Grain of load_hourly: one row per real UTC hour (interval_start_utc is unique).
WITH per_day AS (
    SELECT op_date, COUNT(*) AS n_rows, SUM(CASE WHEN dst_tag THEN 1 ELSE 0 END) AS n_dst_rows,
           MIN(ERCOT) AS min_mw, MAX(ERCOT) AS max_mw
    FROM load_hourly GROUP BY op_date
)
SELECT
    (SELECT COUNT(*) FROM load_hourly)                                         AS total_rows,
    (SELECT COUNT(*) - COUNT(DISTINCT interval_start_utc) FROM load_hourly)    AS duplicate_utc_hours,
    (SELECT COUNT(*) FROM per_day WHERE n_rows = 23)                           AS days_23h,
    (SELECT COUNT(*) FROM per_day WHERE n_rows = 25)                           AS days_25h,
    (SELECT COUNT(*) FROM per_day WHERE n_rows NOT IN (23, 24, 25))            AS days_bad_length,
    (SELECT COUNT(*) FROM load_hourly WHERE ERCOT IS NULL OR ERCOT <= 0)       AS null_or_nonpositive,
    (SELECT COUNT(*) FROM load_hourly
       WHERE ABS(COAST+EAST+FWEST+NORTH+NCENT+SOUTH+SCENT+WEST - ERCOT) > 1)    AS zone_sum_mismatch_rows;
