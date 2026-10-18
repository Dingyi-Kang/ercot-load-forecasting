"""ERCOT Day-Ahead Market hub/zone prices (NP4-180-ER yearly archives).

Availability: DAM for operating day T clears and is posted early afternoon on D = T-1 (ERCOT
posts DAM results by ~13:30), before the 18:00 issue. Treated as known at issue (amendment A2).
Keyed by (delivery date, hour-ending, repeated-hour flag) -> slot like the load table.
"""
import glob, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path("data/processed/dam_prices.parquet")


def load_dam():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    parts = []
    for f in sorted(glob.glob("data/raw/ercot_dam_spp/*.zip")):
        z = zipfile.ZipFile(f)
        x = pd.ExcelFile(z.open(z.namelist()[0]))
        for s in x.sheet_names:
            parts.append(x.parse(s, dtype=str))
    d = pd.concat(parts, ignore_index=True)
    d["T"] = pd.to_datetime(d["Delivery Date"], format="%m/%d/%Y")
    d["slot"] = d["Hour Ending"].str.slice(0, 2).astype(int)
    d["rep"] = (d["Repeated Hour Flag"].str.upper() == "Y").astype(int)
    d["price"] = d["Settlement Point Price"].astype(float)
    p = d.pivot_table(index=["T", "slot", "rep"], columns="Settlement Point", values="price", aggfunc="mean").reset_index()
    p.columns.name = None
    p.to_parquet(CACHE, index=False)
    return p


def dam_features(tgt):
    p = load_dam()
    key = pd.DataFrame({"T": tgt["T"].values, "slot": tgt["slot"].values, "rep": tgt["dst_dup"].values})
    m = key.merge(p[["T", "slot", "rep", "HB_HUBAVG", "LZ_HOUSTON", "LZ_NORTH", "LZ_WEST", "LZ_SOUTH"]],
                  on=["T", "slot", "rep"], how="left")
    out = pd.DataFrame(index=tgt.index)
    out["dam_hub"] = m["HB_HUBAVG"].values
    out["dam_log"] = np.sign(out["dam_hub"]) * np.log1p(np.abs(out["dam_hub"]))
    out["dam_zone_max"] = m[["LZ_HOUSTON", "LZ_NORTH", "LZ_WEST", "LZ_SOUTH"]].max(axis=1).values
    g = out.groupby(tgt["T"].values)["dam_hub"]
    out["dam_day_mean"] = g.transform("mean").values
    out["dam_day_max"] = g.transform("max").values
    out["dam_rel"] = out["dam_hub"] / out["dam_day_mean"]
    return out
