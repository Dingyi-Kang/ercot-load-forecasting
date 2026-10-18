# Open-Meteo previous-runs semantics check (2026-10-13)

Question: for `temperature_2m_previous_dayN` at valid time v, which model run produced the value?
The documentation only says "predicted 24*N hours before valid time".

Test: Dallas (32.78, -96.80), model `gfs_global`, valid date 2026-06-11 (UTC). Compared
previous_day1/day2 values with explicit runs from the single-runs API (2026-06-09T00Z ... 06-10T12Z).

Result (exact value matches):
- previous_day2, valid 00-05Z -> run 06-09T00Z; 06-11Z -> 06-09T06Z; 12-17Z -> 06-09T12Z;
  18-23Z -> 06-09T18Z. i.e. run = floor6(v - 48h).
- previous_day1, valid 00-05Z -> 06-10T00Z; 06-11Z -> 06-10T06Z; 12-22Z no exact match
  with the single-runs output (06-10T12Z single run differs slightly; 06-10T18Z was unavailable).
  Consistent with run = floor6(v - 24h) where checkable.

Decision: treat dayN as run floor6(v - 24N h), usable 6 h after initialisation (GFS is normally
complete ~4-5 h after init). For the 18:00 America/Chicago issue this means day1 is used for
early target hours (through ~13:00 CDT / ~18:00 CST) and day2 otherwise; checked per row in code.
Limitation: the 12Z day1 mismatch is not fully explained; a sensitivity run using day2 for all
hours (strictly older vintages) bounds the risk.

Coverage probe (non-null counts on one day): gfs_global temperature from 2021-03-25; dew point,
radiation etc. only from 2024; ecmwf_ifs025 from 2024; ecmwf_ifs (9 km) has no previous-runs data.

## JMA GSM check (rejected as a vintage source)
Same test for `jma_gsm` (archived since 2018). previous_day1 and previous_day2 values are often
identical, and for valid 14-19Z the previous_day2 value matches run 06-10T18Z (only ~20 h before
valid time), which contradicts the "48 h before" definition. Coarse/rounded output makes matches
ambiguous, but the evidence is enough to reject it: using it could leak post-cutoff information.
Consequence: the GFS archive gap 2023-12-30..2024-01-19 (covering the Jan-2024 cold snap) is
served by the weather-free fallback model; metrics are also reported excluding these days.
