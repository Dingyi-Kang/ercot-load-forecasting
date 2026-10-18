"""Registered experiments. Each has a hypothesis written BEFORE running."""
from .models import SeasonalNaive, RidgePerSlot, LGBM

EXPERIMENTS = {
    "B0_naive_d7": dict(hypothesis="Baseline: same local hour one week before target day.",
                        model=SeasonalNaive),
    "B1_ridge_abs": dict(hypothesis="Baseline: per-slot Ridge on absolute-MW lags+calendar; expected to beat naive but suffer from trend.",
                         model=lambda: RidgePerSlot(alpha=1.0, relative=False), config={"alpha": 1.0, "relative": False}),
    "B2_ridge_rel": dict(hypothesis="Expressing load features and target relative to the 7-day level handles the upward trend better than absolute MW.",
                         model=lambda: RidgePerSlot(alpha=1.0, relative=True), config={"alpha": 1.0, "relative": True}),
    "B3_lgbm_rel": dict(hypothesis="A global LightGBM (relative target, default params) captures nonlinear calendar x hour interactions and beats Ridge.",
                        model=lambda: LGBM(relative=True), config={"relative": True, "params": "default"}),
}

# ---------------------------------------------------------------- weather experiments (amendment A1)
from .wfeatures import add_weather, WeatherWrap, wcols
from .models import RidgeWeather, KalmanRidge, LAGS as _LAGS

_W_TRAIN_START = "2021-04-01"   # forecast vintages exist from 2021-03-25
_LGB_BASE = _LAGS + ["slot", "dow", "month", "doy_sin", "doy_cos", "is_holiday", "holiday_d7", "holiday_d2",
                     "holiday_next", "holiday_prev", "xmas_week", "is_dst", "dst_dup"]
_LGB_W_TEMP = _LGB_BASE + wcols(("temp",)) + ["o2_temp_wavg", "o7_temp_wavg", "o2_tday_mean"]


def _wdata(df):
    return add_weather(df)


EXPERIMENTS.update({
    "W1_lgbm_fcst_temp": dict(
        hypothesis="Adding issue-safe GFS temperature forecasts (11 cities + weighted mean + day aggregates) and observed T-2/T-7 temperature to LightGBM cuts MAE substantially vs B3 (weather explains most day-ahead variance).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(LGBM(relative=True, features=_LGB_W_TEMP, name="lgbm_wtemp"), "fcst"),
        config={"weather": "gfs temp", "mode": "fcst"}),
    "W2_ridge_fcst_temp": dict(
        hypothesis="Per-slot Ridge with hinge temperature basis on GFS forecasts beats B1/B2 by a large margin and is competitive with W1.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(RidgeWeather(alpha=1.0, extra=()), "fcst"),
        config={"alpha": 1.0, "weather": "gfs temp hinge", "mode": "fcst"}),
    "W3_lgbm_obs2fcst_temp": dict(
        hypothesis="Training on ERA5 temperature from 2017 (longer history) but predicting with GFS forecasts beats W1 despite train/test input mismatch.",
        data=_wdata, train_start="2017-02-01",
        model=lambda: WeatherWrap(LGBM(relative=True, features=_LGB_W_TEMP, name="lgbm_wtemp"), "obs2fcst"),
        config={"weather": "era5 train / gfs predict", "mode": "obs2fcst"}),
    "WX_lgbm_backcast_temp": dict(
        hypothesis="BACKCAST ONLY (realized ERA5 weather at target time): upper reference for weather value; not a forecast result.",
        data=_wdata, train_start="2017-02-01",
        model=lambda: WeatherWrap(LGBM(relative=True, features=_LGB_W_TEMP, name="lgbm_wtemp"), "backcast"),
        config={"weather": "era5 at target (backcast)", "mode": "backcast"}),
})

from .models import OnlineBias

EXPERIMENTS.update({
    "W4_lgbm_fcst_bias7": dict(
        hypothesis="W1 has a persistent +600 MW bias (non-stationarity). A per-slot online bias correction (EWMA half-life 7 days of residuals known by T-2) removes most bias and lowers MAE.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineBias(lambda: LGBM(relative=True, features=_LGB_W_TEMP, name="lgbm_wtemp"), halflife=7.0), "fcst"),
        config={"halflife": 7, "warm": 28}),
    "K1_kalman_q1e-3": dict(
        hypothesis="Kalman-adapted per-slot Ridge (q=1e-3) tracks drift and beats static W2 and possibly W1.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(KalmanRidge(q=1e-3, p0=1e-2, extra=()), "fcst"), config={"q": 1e-3, "p0": 1e-2}),
    "K2_kalman_q1e-4": dict(
        hypothesis="Smaller process noise (q=1e-4) is a better adaptation speed than 1e-3.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(KalmanRidge(q=1e-4, p0=1e-2, extra=()), "fcst"), config={"q": 1e-4, "p0": 1e-2}),
})

from .models import Ensemble

_DYN = ["w_tlag1", "w_tlag2", "w_tlag3", "w_tlag6", "w_tsmooth"]
_LGB_W_DYN = _LGB_W_TEMP + _DYN


def _lgb_bias(features, halflife=7.0, name="lgbm_w", params=None, seed=0):
    return lambda: OnlineBias(lambda: LGBM(relative=True, features=features, name=name, params=params, seed=seed), halflife=halflife)


EXPERIMENTS.update({
    "W5_lgbm_dyn_bias7": dict(
        hypothesis="Hourly temperature dynamics (forecast temp 1-6 h earlier + smoothed temp; thermal inertia of buildings) reduce error vs W4, mainly in afternoon/evening hours.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN)(), "fcst"), config={"features": "W_TEMP+DYN", "halflife": 7}),
    "E1_ens_lgbmbias_kalman": dict(
        hypothesis="Equal-weight average of LightGBM+online bias (W4) and Kalman Ridge q=1e-4 (K2) beats both members (diverse error structure).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(Ensemble([_lgb_bias(_LGB_W_TEMP), lambda: KalmanRidge(q=1e-4, p0=1e-2, extra=())], name="ens2"), "fcst"),
        config={"members": ["W4", "K2"], "weights": "equal"}),
})

_DAM = ["dam_log", "dam_rel", "dam_day_max", "dam_zone_max"]
_LGB_W_DYN_DAM = _LGB_W_DYN + _DAM
_KR_DYN = dict(extra=(), hinge_cols=("w_temp_wavg", "w_tday_mean", "w_tday_max", "w_tsmooth"))
_KR_DYN_DAM = dict(extra=("dam_log", "dam_rel"), hinge_cols=("w_temp_wavg", "w_tday_mean", "w_tday_max", "w_tsmooth"))

EXPERIMENTS.update({
    "D1_lgbm_dyn_dam_bias7": dict(
        hypothesis="+DAM: day-ahead hub prices for T (posted ~13:30 D) carry market expectations of load/scarcity and reduce MAE vs W5, especially on high-load hours.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN_DAM)(), "fcst"), config={"features": "W_TEMP+DYN+DAM"}),
    "E2_ens_dyn": dict(
        hypothesis="No-DAM ensemble with dynamics in both members (LGBM W5 + Kalman ridge with tsmooth hinge) improves on E1.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(Ensemble([_lgb_bias(_LGB_W_DYN), lambda: KalmanRidge(q=1e-4, p0=1e-2, **_KR_DYN)]), "fcst"),
        config={"members": ["W5", "K2+tsmooth"]}),
    "E3_ens_dyn_dam": dict(
        hypothesis="+DAM ensemble (D1 + Kalman ridge with tsmooth and DAM) is the best +DAM model.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(Ensemble([_lgb_bias(_LGB_W_DYN_DAM), lambda: KalmanRidge(q=1e-4, p0=1e-2, **_KR_DYN_DAM)]), "fcst"),
        config={"members": ["D1", "K2+tsmooth+DAM"]}),
})

EXPERIMENTS.update({
    "K3_kalman_dyn_q3e-5": dict(hypothesis="Kalman ridge (tsmooth hinge) with slower drift q=3e-5: check adaptation speed.",
        data=_wdata, train_start=_W_TRAIN_START, model=lambda: WeatherWrap(KalmanRidge(q=3e-5, p0=1e-2, **_KR_DYN), "fcst"), config={"q": 3e-5}),
    "K4_kalman_dyn_q1e-4": dict(hypothesis="Kalman ridge (tsmooth hinge) q=1e-4 alone (member of E2).",
        data=_wdata, train_start=_W_TRAIN_START, model=lambda: WeatherWrap(KalmanRidge(q=1e-4, p0=1e-2, **_KR_DYN), "fcst"), config={"q": 1e-4}),
    "K5_kalman_dyn_q3e-4": dict(hypothesis="Kalman ridge (tsmooth hinge) with faster drift q=3e-4.",
        data=_wdata, train_start=_W_TRAIN_START, model=lambda: WeatherWrap(KalmanRidge(q=3e-4, p0=1e-2, **_KR_DYN), "fcst"), config={"q": 3e-4}),
    "W6_lgbm_obs2fcst_dyn_bias7": dict(
        hypothesis="Training LightGBM on ERA5 weather from 2017 (more extreme events) and predicting with GFS, with online bias correction absorbing the GFS-vs-ERA5 offset, beats W5 (trained on GFS from 2021-04).",
        data=_wdata, train_start="2017-02-01",
        model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN)(), "obs2fcst"), config={"mode": "obs2fcst", "halflife": 7}),
    "W7_lgbm_dyn_bias3": dict(hypothesis="Faster bias adaptation (half-life 3 days) beats 7 days.",
        data=_wdata, train_start=_W_TRAIN_START, model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN, halflife=3.0)(), "fcst"), config={"halflife": 3}),
    "W8_lgbm_dyn_bias14": dict(hypothesis="Slower bias adaptation (half-life 14 days) beats 7 days.",
        data=_wdata, train_start=_W_TRAIN_START, model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN, halflife=14.0)(), "fcst"), config={"halflife": 14}),
})

from .tsfm import Chronos2Zero

EXPERIMENTS.update({
    "T1_chronos2_uni": dict(
        hypothesis="Zero-shot Chronos-2 on load history only (context 2048 h ending at T-2) beats the seasonal naive and Ridge baselines (B0-B2) but not weather-based models.",
        model=lambda: Chronos2Zero(covariates=False), config={"context": 2048, "covariates": None}),
    "T2_chronos2_cov": dict(
        hypothesis="Zero-shot Chronos-2 with known-future GFS weighted temperature + calendar covariates is competitive with trained weather models (W5, ~1600 MAE).",
        model=lambda: Chronos2Zero(covariates=True), config={"context": 2048, "covariates": ["gfs temp wavg", "hour", "dow", "holiday"]}),
})

EXPERIMENTS.update({
    "W9_lgbm_dyn_bias7_blend": dict(
        hypothesis="Feeding the GFS+ECMWF mean temperature at prediction time (training unchanged on GFS) lowers weather-input error and MAE vs W5 (decided a priori: multi-model NWP means are usually more accurate).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN)(), "blend"), config={"predict_weather": "mean(gfs, ecmwf_ifs025)"}),
    "E4_ens_dyn_blend": dict(
        hypothesis="E2 with GFS+ECMWF blended inputs at prediction time beats E2.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(Ensemble([_lgb_bias(_LGB_W_DYN), lambda: KalmanRidge(q=1e-4, p0=1e-2, **_KR_DYN)]), "blend"),
        config={"members": ["W5", "K4"], "predict_weather": "blend"}),
})

from .models import OnlineEnsemble

EXPERIMENTS.update({
    "W10_lgbm_dyn_bias28": dict(hypothesis="Bias half-life 14 beat 7 (1578 vs 1596); 28 days may be better still (slower drift).",
        data=_wdata, train_start=_W_TRAIN_START, model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN, halflife=28.0)(), "fcst"), config={"halflife": 28}),
    "E5_onlineens_lgbm14_kalman3e-5": dict(
        hypothesis="Online-weighted ensemble (weights from recent errors <= T-2, per slot group) of the best LGBM (bias hl14) and best Kalman ridge (q=3e-5) beats the equal-weight E2.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_lgb_bias(_LGB_W_DYN, halflife=14.0), lambda: KalmanRidge(q=3e-5, p0=1e-2, **_KR_DYN)]), "fcst"),
        config={"members": ["W8", "K3"], "halflife": 14, "eta": 50}),
})

from .zonal import Zonal

EXPERIMENTS.update({
    "Z1_zonal_lgbm_bias14": dict(
        hypothesis="Bottom-up: 8 zone models (zone's own city temperatures + statewide mean; LGBM dyn + bias hl14) summed beat the system-level W8 (1578), because zones have different weather and growth (e.g. Far West oil/data-center load).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(Zonal(lambda z: OnlineBias(lambda: LGBM(relative=True, features=_LGB_W_DYN + ["w_temp_state"], name="lgbm_zone"), halflife=14.0)), "fcst"),
        config={"zones": 8, "member": "W8-style"}),
})

_M_LGB14 = _lgb_bias(_LGB_W_DYN, halflife=14.0)
_M_K3 = lambda: KalmanRidge(q=3e-5, p0=1e-2, **_KR_DYN)

EXPERIMENTS.update({
    "E6_ens_lgb14_k3_blend": dict(
        hypothesis="Equal-weight ensemble of the tuned members (LGBM bias hl14, Kalman q=3e-5) with GFS+ECMWF blended inputs beats E4 (1412).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(Ensemble([_M_LGB14, _M_K3]), "blend"), config={"members": ["W8", "K3"], "weights": "equal", "weather": "blend"}),
    "E7_onlineens_lgb14_k3_blend": dict(
        hypothesis="Same members as E6 with online weights; isolates the value of online weighting (E5 vs E2 confounded member changes).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_M_LGB14, _M_K3]), "blend"), config={"members": ["W8", "K3"], "weights": "online", "weather": "blend"}),
    "E8_onlineens_lgb14_k3_chronos_blend": dict(
        hypothesis="Adding zero-shot Chronos-2 (with covariates) as a third, structurally different member to E7 lowers MAE; online weights protect against its weaker average accuracy.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_M_LGB14, _M_K3, lambda: Chronos2Zero(covariates=True)]), "blend"),
        config={"members": ["W8", "K3", "T2"], "weights": "online", "weather": "blend (Chronos uses GFS)"}),
})

_EXTRA_W = ["w_dew_wavg", "w_cloud_wavg", "w_swr_wavg", "w_wind_wavg"]

EXPERIMENTS.update({
    "W11_lgbm14_hybrid_extra_blend": dict(
        hypothesis="Humidity (dew point), cloud cover, solar radiation and wind add information beyond temperature (Texas summer humidity; behind-the-meter solar lowers midday native load). Train: GFS temp + ERA5 extras; predict: blended temp + GFS extras. Beats W8-style LGBM with blend (no extras).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN + _EXTRA_W, halflife=14.0)(), "hybrid_blend"),
        config={"extras": _EXTRA_W, "train_extras": "ERA5", "predict_extras": "GFS"}),
    "W12_lgbm14_blend": dict(
        hypothesis="Control for W11: same LGBM (bias hl14, dynamics) with blended temperature, no extra variables.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(_lgb_bias(_LGB_W_DYN, halflife=14.0)(), "blend"), config={"extras": None}),
})

from .zonal import Zonal as _Zonal

_M_CHR_BLEND = lambda: Chronos2Zero(covariates=True, blend=True)
_M_ZONAL = lambda: _Zonal(lambda z: OnlineBias(lambda: LGBM(relative=True, features=_LGB_W_DYN + ["w_temp_state"], name="lgbm_zone"), halflife=14.0))

EXPERIMENTS.update({
    "T3_chronos2_cov_blend": dict(
        hypothesis="Chronos-2 zero-shot with GFS+ECMWF blended temperature covariate beats T2 (GFS only), as blending helped the trained models.",
        model=_M_CHR_BLEND, config={"covariates": ["temp blend", "hour", "dow", "holiday"]}),
    "E9_onlineens_lgb14_k3_chronosblend": dict(
        hypothesis="E8 with the blended-temperature Chronos member (T3) beats E8.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_M_LGB14, _M_K3, _M_CHR_BLEND]), "blend"), config={"members": ["W8", "K3", "T3"]}),
    "E10_onlineens_4members": dict(
        hypothesis="Adding the zonal LGBM (Z1) as a 4th member adds diversity and beats E9.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_M_LGB14, _M_K3, _M_CHR_BLEND, _M_ZONAL]), "blend"), config={"members": ["W8", "K3", "T3", "Z1"]}),
})

EXPERIMENTS.update({
    "T4_chronos2_cov_blend_ctx8192": dict(
        hypothesis="A longer Chronos-2 context (8192 h ~ 11 months, the model maximum) captures annual/seasonal level and beats T3 (2048 h).",
        model=lambda: Chronos2Zero(covariates=True, blend=True, context_hours=8192), config={"context": 8192}),
})

# ---- Study S2 (PROTOCOL A4): GPU fine-tuned Chronos-2. Settings chosen on the 2023-H2 pilot
# (training period): LoRA 1000 steps lr 1e-5 (pilot MAE 1340) and full 1000 steps lr 5e-6 (1338)
# vs zero-shot 1657. Fine-tuned at every fold start on data <= S-2.
from .tsfm import Chronos2FT as _Chronos2FT

_M_CHR_LORA = lambda: _Chronos2FT(mode="lora", steps=1000, lr=1e-5, blend=True, eval_fix=False)   # as run (dropout on)
_M_CHR_FULL = lambda: _Chronos2FT(mode="full", steps=1000, lr=5e-6, blend=True, eval_fix=False)

EXPERIMENTS.update({
    "T5_chronos2_ft_lora1000": dict(
        hypothesis="S2: Chronos-2 LoRA fine-tuned per fold (1000 steps, lr 1e-5) beats zero-shot T3 (1845) on validation, as in the pilot (-19%).",
        model=_M_CHR_LORA, config={"finetune": "lora", "steps": 1000, "lr": 1e-5, "ft_context": 2048, "series": "ERCOT + 8 zones"}),
    "T6_chronos2_ft_full1000": dict(
        hypothesis="S2: Chronos-2 full fine-tuned per fold (1000 steps, lr 5e-6) beats zero-shot T3 on validation.",
        model=_M_CHR_FULL, config={"finetune": "full", "steps": 1000, "lr": 5e-6, "ft_context": 2048, "series": "ERCOT + 8 zones"}),
    "E11_onlineens_lgb14_k3_chronoslora": dict(
        hypothesis="S2: E9 with the LoRA fine-tuned Chronos member (T5) instead of zero-shot T3 beats E9 (CI excludes 0), unless fine-tuning makes Chronos errors more correlated with the other members.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_M_LGB14, _M_K3, _M_CHR_LORA]), "blend"), config={"members": ["W8", "K3", "T5"]}),
    "E12_onlineens_lgb14_k3_chronosfull": dict(
        hypothesis="S2: E9 with the fully fine-tuned Chronos member (T6) instead of T3 beats E9 (CI excludes 0).",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_M_LGB14, _M_K3, _M_CHR_FULL]), "blend"), config={"members": ["W8", "K3", "T6"]}),
})

# ---- PROTOCOL A5: eval-mode fix + seed averaging (pilot on 2023-H2 only)
from .tsfm import Chronos2FTSeeds as _Seeds
_M_CHR_LORA3 = lambda: _Seeds(seeds=(0, 1, 2), mode="lora", steps=1000, lr=1e-5, blend=True)
_M_CHR_LORA1 = lambda: _Chronos2FT(mode="lora", steps=1000, lr=1e-5, blend=True, seed=0, cache=True)

EXPERIMENTS.update({
    "T7_chronos2_ft_lora_eval_seeds3": dict(
        hypothesis="A5: LoRA fine-tuned Chronos-2 predicting in eval mode, averaged over 3 seeds, beats T5 (1393, dropout on) on validation; pilot 1263 vs 1344.",
        model=_M_CHR_LORA3, config={"finetune": "lora", "steps": 1000, "lr": 1e-5, "seeds": [0, 1, 2], "eval_fix": True}),
    "T8_chronos2_ft_lora_eval_seed0": dict(
        hypothesis="A5: the eval-mode fix alone (seed 0, otherwise T5) beats T5; isolates the fix from seed averaging. Reuses T7's seed-0 weights.",
        model=_M_CHR_LORA1, config={"finetune": "lora", "steps": 1000, "lr": 1e-5, "seeds": [0], "eval_fix": True}),
    "E13_onlineens_lgb14_k3_chronoslora3": dict(
        hypothesis="A5: E9 with the T7 member beats E11 (1247) with CI excluding 0.",
        data=_wdata, train_start=_W_TRAIN_START,
        model=lambda: WeatherWrap(OnlineEnsemble([_M_LGB14, _M_K3, _M_CHR_LORA3]), "blend"), config={"members": ["W8", "K3", "T7"]}),
})
