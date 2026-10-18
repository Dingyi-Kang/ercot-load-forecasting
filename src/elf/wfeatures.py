"""Attach weather features to the A0 feature table.

Columns produced (X = temp, dew, cloud, swr, wind):
  wf_X_<city>, wf_X_wavg : issue-safe forecast at target interval (Open-Meteo gfs_global vintages)
  wo_X_<city>, wo_X_wavg : ERA5 at target interval  -> only for training (train-obs) or BACKCAST
  o2_temp_wavg, o7_temp_wavg : ERA5 at same time on T-2 / T-7 (past, known at issue)
  plus daily aggregates (mean/max/min over the target day) for wf_ and wo_ temperature.
Models choose which family they see through `WeatherWrap(mode=...)`.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from . import weather as W
from .weather_download import CITIES

VARMAP = {"temperature_2m": "temp", "dew_point_2m": "dew", "cloud_cover": "cloud",
          "shortwave_radiation": "swr", "wind_speed_10m": "wind"}
CACHE = Path("data/processed/features_A0_weather.parquet")


def add_weather(df, rebuild=False):
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)
    load = pd.read_parquet("data/processed/load_hourly.parquet")
    wts = W.city_weights(load)
    parts = [df]
    f = W.forecast_features(df, "gfs_global", tuple(VARMAP), wts, prefix="wf")
    o = W.observed_features(df, tuple(VARMAP), wts, prefix="wo", day_offset=0)
    o2 = W.observed_features(df, ("temperature_2m",), wts, prefix="o2", day_offset=2)
    o7 = W.observed_features(df, ("temperature_2m",), wts, prefix="o7", day_offset=7)
    for fr in (f, o, o2, o7):
        fr.columns = [c.replace("temperature_2m", "temp") for c in fr.columns]
        for k, v in VARMAP.items():
            fr.columns = [c.replace(k, v) for c in fr.columns]
    parts += [f, o, o2[["o2_temp_wavg"]], o7[["o7_temp_wavg"]]]
    out = pd.concat(parts, axis=1)
    for p in ("wf", "wo"):
        g = out.groupby("T")[f"{p}_temp_wavg"]
        out[f"{p}_tday_mean"] = g.transform("mean")
        out[f"{p}_tday_max"] = g.transform("max")
        out[f"{p}_tday_min"] = g.transform("min")
    out["o2_tday_mean"] = out.groupby("T")["o2_temp_wavg"].transform("mean")
    # temperature dynamics: weighted-mean temperature 1,2,3,6 h before the interval (same vintage rule;
    # earlier valid times are never fresher than allowed). Forecast and ERA5 versions.
    for h in (1, 2, 3, 6):
        f_h = W.forecast_features(df, "gfs_global", ("temperature_2m",), wts, prefix="wf", shift_h=h)
        o_h = W.observed_features(df.assign(interval_start_utc=df["interval_start_utc"] - pd.Timedelta(hours=h)),
                                  ("temperature_2m",), wts, prefix="wo")
        out[f"wf_tlag{h}"] = f_h["wf_temperature_2m_wavg"].values
        out[f"wo_tlag{h}"] = o_h["wo_temperature_2m_wavg"].values
    for p in ("wf", "wo"):
        out[f"{p}_tsmooth"] = 0.4 * out[f"{p}_temp_wavg"] + 0.25 * out[f"{p}_tlag1"] + 0.15 * out[f"{p}_tlag2"] + 0.1 * out[f"{p}_tlag3"] + 0.1 * out[f"{p}_tlag6"]
    # ECMWF IFS 0.25 vintages (2024-02+) and a GFS/ECMWF blend ("wb_"): mean where both exist,
    # GFS otherwise. Used only at prediction time in mode "blend" (training stays on GFS).
    e = W.forecast_features(df, "ecmwf_ifs025", ("temperature_2m",), wts, prefix="we")
    for c in [c for c in e.columns if c.startswith("we_temperature_2m_")]:
        g = c.replace("we_temperature_2m_", "wf_temp_")
        out["wb_temp_" + c[len("we_temperature_2m_"):]] = np.where(np.isnan(e[c].values), out[g].values, (e[c].values + out[g].values) / 2)
    for h in (1, 2, 3, 6):
        eh = W.forecast_features(df, "ecmwf_ifs025", ("temperature_2m",), wts, prefix="we", shift_h=h)["we_temperature_2m_wavg"].values
        out[f"wb_tlag{h}"] = np.where(np.isnan(eh), out[f"wf_tlag{h}"].values, (eh + out[f"wf_tlag{h}"].values) / 2)
    gb = out.groupby("T")["wb_temp_wavg"]
    out["wb_tday_mean"], out["wb_tday_max"], out["wb_tday_min"] = gb.transform("mean"), gb.transform("max"), gb.transform("min")
    out["wb_tsmooth"] = 0.4 * out["wb_temp_wavg"] + 0.25 * out["wb_tlag1"] + 0.15 * out["wb_tlag2"] + 0.1 * out["wb_tlag3"] + 0.1 * out["wb_tlag6"]
    out["ecmwf_avail"] = (~np.isnan(e["we_temperature_2m_wavg"].values)).astype(int)
    from .dam import dam_features
    out = pd.concat([out, dam_features(out)], axis=1)
    out.to_parquet(CACHE, index=False)
    return out


def wcols(variables=("temp",), cities=True, daily=True):
    cols = []
    for v in variables:
        cols.append(f"w_{v}_wavg")
        if cities:
            cols += [f"w_{v}_{c}" for c in CITIES]
    if daily:
        cols += ["w_tday_mean", "w_tday_max", "w_tday_min"]
    return cols


class WeatherWrap:
    """mode: 'fcst' (train+predict on forecasts), 'obs2fcst' (train on ERA5, predict on forecasts),
    'backcast' (train+predict on ERA5 at target time: NOT a forecast)."""
    def __init__(self, base, mode="fcst", fallback=True):
        self.base, self.mode = base, mode
        self.name = f"{base.name}_{mode}"
        # Operational fallback when no archived forecast exists for a target interval
        # (GFS archive gap 2023-12-30..2024-01-19): weather-free LightGBM fit on the same fold.
        self.fallback = None
        if fallback:
            from .models import LGBM
            self.fallback = LGBM(relative=True, name="fallback_lgbm")

    EXTRA = ("dew", "cloud", "swr", "wind")

    @classmethod
    def _view(cls, df, src, extra_src=None):
        """src supplies temperature-family columns; extra_src (if given) supplies dew/cloud/swr/wind."""
        ren = {c: "w_" + c[3:] for c in df.columns if c.startswith(src + "_")}
        if extra_src:
            ren = {c: n for c, n in ren.items() if not any(n.startswith(f"w_{e}_") for e in cls.EXTRA)}
            ren.update({c: "w_" + c[3:] for c in df.columns if c.startswith(extra_src + "_")
                        and any(c[3:].startswith(e + "_") for e in cls.EXTRA)})
        keep = [c for c in df.columns if not c.startswith(("wf_", "wo_", "wb_"))]
        return df[keep + list(ren)].rename(columns=ren)

    def fit(self, tr):
        src = "wo" if self.mode in ("obs2fcst", "backcast") else "wf"
        # hybrid: temperature from GFS forecasts, extra variables from ERA5 (GFS extras exist only 2024+)
        v = self._view(tr, src, "wo" if self.mode.startswith("hybrid") else None)
        if self.fallback is not None:
            self.fallback.fit(tr)
        v = v.dropna(subset=[c for c in ("w_temp_wavg", "w_tsmooth") if c in v.columns])
        self.base.fit(v)
        return self

    def predict(self, df):
        src = {"backcast": "wo", "blend": "wb", "hybrid_blend": "wb"}.get(self.mode, "wf")
        v = self._view(df, src, "wf" if self.mode.startswith("hybrid") else None)
        miss = v[[c for c in ("w_temp_wavg", "w_tsmooth") if c in v.columns]].isna().any(axis=1).values
        p = np.full(len(df), np.nan)
        if (~miss).any():
            p[~miss] = self.base.predict(v[~miss])
        if miss.any():
            p[miss] = self.fallback.predict(df[miss])
        return p
