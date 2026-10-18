"""Turn raw Open-Meteo JSON into issue-time-safe weather features.

Forecast vintage rule (reports/weather_semantics.md): `<var>_previous_dayN` at valid time v comes
from the run initialised at R = floor6(v - 24N h). We assume a run is usable PUB_DELAY hours after
initialisation. For each target interval we use the freshest N in (1, 2) with R + PUB_DELAY <= issue.
Weather value for an hourly load interval = mean of the instantaneous values at its start and end.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from .weather_download import CITIES

PUB_DELAY_H = 6
RAW = Path("data/weather/raw")
CACHE = Path("data/processed")

# fixed city weights ~ share of ERCOT load by weather zone (zone means over 2021-2023, i.e. before
# the first validation fold); zones with two cities split their weight equally
ZONE_CITIES = {}
for c, (_, _, z) in CITIES.items():
    ZONE_CITIES.setdefault(z, []).append(c)


def city_weights(load):
    tr = load[(load["local_start"] >= "2021-01-01") & (load["local_start"] < "2024-01-01")]
    share = tr[list(ZONE_CITIES)].mean() / tr[list(ZONE_CITIES)].mean().sum()
    return {c: share[z] / len(cs) for z, cs in ZONE_CITIES.items() for c in cs}


def _read(kind, model):
    rows = []
    for f in sorted(glob.glob(str(RAW / kind / model / "*.json"))):
        city = Path(f).name.split("_")[0]
        h = json.loads(Path(f).read_text())["hourly"]
        df = pd.DataFrame(h)
        df["city"] = city
        rows.append(df)
    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    return df.groupby(["city", "time"]).first().sort_index()  # first non-null per column across overlapping files


def obs_table():
    p = CACHE / "weather_obs_era5.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = _read("obs", "era5")
    df.to_parquet(p)
    return df


def fcst_table(model):
    p = CACHE / f"weather_fcst_{model}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    df = _read("fcst", model)
    df.to_parquet(p)
    return df


def _lookup(tab, city, times, col):
    s = tab[col].xs(city, level="city") if col in tab.columns else None
    if s is None:
        return np.full(len(times), np.nan)
    return s.reindex(times).values


def interval_values(tab, city, starts, col):
    a = _lookup(tab, city, starts, col)
    b = _lookup(tab, city, starts + pd.Timedelta(hours=1), col)
    return (a + b) / 2


PUB_DELAY_BY_MODEL = {"gfs_global": 6, "ecmwf_ifs025": 8}


def forecast_features(tgt, model="gfs_global", variables=("temperature_2m",), weights=None, prefix="f", shift_h=0):
    """Per target interval: issue-safe forecast value per city and the load-weighted mean."""
    tab = fcst_table(model)
    starts = pd.DatetimeIndex(tgt["interval_start_utc"]) - pd.Timedelta(hours=shift_h)
    ends = starts + pd.Timedelta(hours=1)
    issue = pd.DatetimeIndex(tgt["issue_utc"])
    out = {}
    # vintage choice is evaluated at the later instant (interval end): stricter for both endpoints
    delay = pd.Timedelta(hours=PUB_DELAY_BY_MODEL.get(model, PUB_DELAY_H))
    ok1 = (ends - pd.Timedelta(hours=24)).floor("6h") + delay <= issue
    ok2 = (ends - pd.Timedelta(hours=48)).floor("6h") + delay <= issue
    assert ok2.all(), "day2 vintage must always be available at 18:00 issue"
    out[f"{prefix}_used_day1"] = ok1.astype(int)
    for var in variables:
        cols = []
        for city in CITIES:
            v1 = interval_values(tab, city, starts, f"{var}_previous_day1")
            v2 = interval_values(tab, city, starts, f"{var}_previous_day2")
            v = np.where(ok1 & ~np.isnan(v1), v1, v2)
            out[f"{prefix}_{var}_{city}"] = v
            cols.append(f"{prefix}_{var}_{city}")
        if weights:
            W = np.array([weights[c] for c in CITIES])
            M = np.column_stack([out[c] for c in cols])
            out[f"{prefix}_{var}_wavg"] = (M * W).sum(1) / (~np.isnan(M) * W).sum(1)
    return pd.DataFrame(out, index=tgt.index)


def observed_features(tgt, variables=("temperature_2m",), weights=None, prefix="o", day_offset=0):
    """ERA5 values at target interval shifted by day_offset days (0 = target time: BACKCAST ONLY)."""
    tab = obs_table()
    starts = pd.DatetimeIndex(tgt["interval_start_utc"]) - pd.Timedelta(days=day_offset)
    out = {}
    for var in variables:
        cols = []
        for city in CITIES:
            out[f"{prefix}_{var}_{city}"] = interval_values(tab, city, starts, var)
            cols.append(f"{prefix}_{var}_{city}")
        if weights:
            W = np.array([weights[c] for c in CITIES])
            M = np.column_stack([out[c] for c in cols])
            out[f"{prefix}_{var}_wavg"] = (M * W).sum(1) / W.sum()
    return pd.DataFrame(out, index=tgt.index)
