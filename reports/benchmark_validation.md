# Validation benchmark (2024-01-01 .. 2025-12-31, 8 quarterly rolling-origin folds)

Generated from experiments/registry.jsonl by src/elf/report.py.
Validation runs so far: 48 (budget 60).

| # | experiment | status | MAE | RMSE | MAPE % | bias | peak MAE | high-load MAE |
|---|---|---|---|---|---|---|---|---|
| 1 | B0_naive_d7 | ok | 4615 | 6478 | 8.52 | -39 | 5733 | 5642 |
| 2 | B1_ridge_abs | ok | 2819 | 3923 | 5.11 | -526 | 3791 | 4160 |
| 3 | B2_ridge_rel | ok | 2885 | 4027 | 5.32 | 217 | 3774 | 3588 |
| 4 | B3_lgbm_rel | ok | 2854 | 4047 | 5.22 | -48 | 3804 | 3668 |
| 5 | W1_lgbm_fcst_temp | ok | 1658 | 2522 | 3.03 | 593 | 2169 | 2181 |
| 6 | W2_ridge_fcst_temp | error | – | – | – | – | – | – |
| 7 | W3_lgbm_obs2fcst_temp | ok | 2203 | 3084 | 3.93 | 1121 | 2966 | 3539 |
| 8 | WX_lgbm_backcast_temp | ok | 1144 | 1580 | 2.13 | 341 | 1484 | 1320 |
| 9 | W2_ridge_fcst_temp | ok | 1735 | 2540 | 3.18 | 640 | 2253 | 2326 |
| 10 | W4_lgbm_fcst_bias7 | ok | 1616 | 2446 | 2.94 | -34 | 2051 | 2282 |
| 11 | K1_kalman_q1e-3 | ok | 1734 | 2545 | 3.19 | 76 | 2115 | 2287 |
| 12 | K2_kalman_q1e-4 | ok | 1639 | 2444 | 3.02 | 114 | 2002 | 2078 |
| 13 | W5_lgbm_dyn_bias7 | ok | 1596 | 2423 | 2.90 | -34 | 2070 | 2263 |
| 14 | E1_ens_lgbmbias_kalman | ok | 1484 | 2289 | 2.71 | 40 | 1861 | 2026 |
| 15 | D1_lgbm_dyn_dam_bias7 | ok | 1591 | 2413 | 2.90 | -39 | 2040 | 2217 |
| 16 | E2_ens_dyn | error | – | – | – | – | – | – |
| 17 | E3_ens_dyn_dam | error | – | – | – | – | – | – |
| 18 | E2_ens_dyn | ok | 1466 | 2269 | 2.68 | 37 | 1869 | 2013 |
| 19 | E3_ens_dyn_dam | ok | 1458 | 2259 | 2.66 | 24 | 1858 | 2013 |
| 20 | K3_kalman_dyn_q3e-5 | ok | 1617 | 2422 | 2.98 | 117 | 1989 | 2072 |
| 21 | K4_kalman_dyn_q1e-4 | ok | 1627 | 2432 | 3.00 | 108 | 1999 | 2093 |
| 22 | K5_kalman_dyn_q3e-4 | ok | 1655 | 2461 | 3.05 | 93 | 2032 | 2157 |
| 23 | W6_lgbm_obs2fcst_dyn_bias7 | ok | 1643 | 2439 | 3.01 | 87 | 2092 | 2048 |
| 24 | W7_lgbm_dyn_bias3 | ok | 1637 | 2469 | 2.98 | -45 | 2118 | 2333 |
| 25 | W8_lgbm_dyn_bias14 | ok | 1578 | 2409 | 2.87 | -20 | 2063 | 2235 |
| 26 | T1_chronos2_uni | ok | 2858 | 4036 | 5.22 | -289 | 3757 | 3977 |
| 27 | T2_chronos2_cov | ok | 1909 | 2856 | 3.56 | -271 | 2284 | 2093 |
| 28 | W9_lgbm_dyn_bias7_blend | ok | 1572 | 2390 | 2.86 | -59 | 2012 | 2281 |
| 29 | E4_ens_dyn_blend | ok | 1412 | 2212 | 2.58 | -29 | 1769 | 1981 |
| 30 | W10_lgbm_dyn_bias28 | ok | 1574 | 2407 | 2.86 | -1 | 2056 | 2205 |
| 31 | E5_onlineens_lgbm14_kalman3e-5 | ok | 1451 | 2256 | 2.65 | 52 | 1853 | 1981 |
| 32 | Z1_zonal_lgbm_bias14 | ok | 1562 | 2380 | 2.83 | -6 | 2011 | 2248 |
| 33 | E6_ens_lgb14_k3_blend | ok | 1406 | 2203 | 2.57 | -29 | 1762 | 1978 |
| 34 | E7_onlineens_lgb14_k3_blend | ok | 1399 | 2197 | 2.56 | -23 | 1754 | 1969 |
| 35 | E8_onlineens_lgb14_k3_chronos_blend | ok | 1326 | 2128 | 2.42 | -62 | 1662 | 1848 |
| 36 | W11_lgbm14_hybrid_extra_blend | ok | 1570 | 2388 | 2.85 | -66 | 2053 | 2359 |
| 37 | W12_lgbm14_blend | ok | 1563 | 2380 | 2.85 | -56 | 2005 | 2278 |
| 38 | T3_chronos2_cov_blend | ok | 1845 | 2810 | 3.45 | -273 | 2270 | 1994 |
| 39 | E9_onlineens_lgb14_k3_chronosblend | ok | 1310 | 2113 | 2.39 | -64 | 1656 | 1822 |
| 40 | E10_onlineens_4members | ok | 1313 | 2115 | 2.39 | -53 | 1663 | 1864 |
| 41 | T4_chronos2_cov_blend_ctx8192 | ok | 1890 | 2825 | 3.52 | -195 | 2301 | 2080 |
| 42 | T5_chronos2_ft_lora1000 | ok | 1393 | 2235 | 2.56 | -88 | 1775 | 1821 |
| 43 | T6_chronos2_ft_full1000 | ok | 1408 | 2253 | 2.59 | -63 | 1776 | 1819 |
| 44 | E11_onlineens_lgb14_k3_chronoslora | ok | 1247 | 2052 | 2.27 | -35 | 1587 | 1775 |
| 45 | E12_onlineens_lgb14_k3_chronosfull | ok | 1250 | 2057 | 2.28 | -25 | 1580 | 1770 |
| 46 | T7_chronos2_ft_lora_eval_seeds3 | ok | 1325 | 2201 | 2.42 | -83 | 1667 | 1805 |
| 47 | T8_chronos2_ft_lora_eval_seed0 | ok | 1344 | 2212 | 2.46 | -90 | 1694 | 1806 |
| 48 | E13_onlineens_lgb14_k3_chronoslora3 | ok | 1247 | 2050 | 2.27 | -33 | 1590 | 1777 |

Hypotheses for each experiment are stored in the registry and in src/elf/experiments.py.

# Final holdout (2026-01-01 .. 2026-09-30, opened once after configs/final.yaml was locked)

| experiment | MAE | RMSE | MAPE % | bias | peak MAE | high-load MAE |
|---|---|---|---|---|---|---|
| B0_naive_d7 | 4286 | 5949 | 7.38 | -417 | 5548 | 4008 |
| B1_ridge_abs | 2755 | 3745 | 4.65 | -667 | 3784 | 2935 |
| B3_lgbm_rel | 2887 | 4005 | 4.94 | -88 | 4000 | 2608 |
| W8_lgbm_dyn_bias14 | 1292 | 1763 | 2.15 | -64 | 1639 | 1541 |
| K3_kalman_dyn_q3e-5 | 1472 | 1987 | 2.48 | 211 | 1885 | 1693 |
| E2_ens_dyn | 1255 | 1693 | 2.10 | 82 | 1628 | 1505 |
| E9_onlineens_lgb14_k3_chronosblend | 1112 | 1487 | 1.86 | 28 | 1466 | 1305 |
