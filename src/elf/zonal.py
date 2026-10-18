"""Bottom-up forecasting: one model per ERCOT weather zone, summed to system.

Zone-level load features are built with the same A0 code (features.build with col=<zone>), keyed by
interval_start_utc. Each zone model sees the zone's own city temperatures as its primary weather
input (w_temp_wavg replaced by the zone mean) plus the statewide weighted mean (w_temp_state).
Evaluation is still against the ERCOT total column (y), on the identical sample.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from . import features as F
from .weather_download import CITIES

ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]
CACHE = Path("data/processed/zone_features.parquet")
LOADCOLS = ["lag_d2", "lag_d3", "lag_d4", "lag_d5", "lag_d6", "lag_d7", "lag_d14", "lag_d21", "lag_d28", "lag_d364",
            "lag_d2_prev", "lag_d2_next", "lag_d7_prev", "lag_d7_next", "d2_mean", "d2_max", "d2_min", "d2_last",
            "d3_mean", "d9_mean", "level7", "level28", "level7_lastyear"]


def zone_tables():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    load = pd.read_parquet("data/processed/load_hourly.parquet")
    parts = []
    for z in ZONES:
        f = F.build(load, col=z)[["interval_start_utc", "y"] + LOADCOLS]
        f = f.rename(columns={"y": "zy", **{c: f"z_{c}" for c in LOADCOLS}})
        f["zone"] = z
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    return out


class Zonal:
    """factory(zone) -> model operating on a frame whose y/lag/level columns are the zone's."""
    def __init__(self, factory, name="zonal"):
        self.factory, self.name = factory, name
        self.Z = zone_tables()

    def _zone_frame(self, df, z):
        zt = self.Z[self.Z["zone"] == z].set_index("interval_start_utc")
        f = df.copy()
        zz = zt.reindex(pd.DatetimeIndex(f["interval_start_utc"]))
        f["y"] = zz["zy"].values
        for c in LOADCOLS:
            f[c] = zz[f"z_{c}"].values
        cities = [c for c, (_, _, zc) in CITIES.items() if zc == z]
        f["w_temp_state"] = f["w_temp_wavg"]
        f["w_temp_wavg"] = f[[f"w_temp_{c}" for c in cities]].mean(axis=1)
        return f

    def fit(self, tr):
        self.ms = {z: self.factory(z).fit(self._zone_frame(tr, z)) for z in ZONES}
        return self

    def predict(self, df):
        return np.sum([self.ms[z].predict(self._zone_frame(df, z)) for z in ZONES], axis=0)
