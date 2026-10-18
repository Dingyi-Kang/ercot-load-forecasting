"""Build the issue/target feature table under availability policy A0.

One row per (issue, target interval). Issue = 18:00 America/Chicago on D, target day T = D+1.
Under A0 only operating days <= T-2 (= D-1) of load are available. Every load-derived feature
records the latest operating day it used (`src_max_day`) so tests can assert the cutoff.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, USColumbusDay,
                                    nearest_workday)

TZ = "America/Chicago"
LAG_DAYS = [2, 3, 4, 5, 6, 7, 14, 21, 28, 364]


class LoadHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("new_year", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
        Holiday("july4", month=7, day=4, observance=nearest_workday),
        USLaborDay, USColumbusDay,
        Holiday("veterans", month=11, day=11),
        USThanksgivingDay,
        Holiday("christmas", month=12, day=25, observance=nearest_workday),
    ]


def holiday_table(start, end):
    """Known-in-advance calendar: holiday id per date plus a few load-relevant special days."""
    cal = LoadHolidays().holidays(start, end, return_name=True)
    hol = pd.Series(cal.values, index=cal.index.date)
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame(index=days.date)
    out["holiday"] = hol.reindex(out.index).fillna("none")
    tg = [d for d, n in hol.items() if "Thanksgiving" in n]
    for d in tg:
        f = d + pd.Timedelta(days=1)
        if f in out.index:
            out.loc[f, "holiday"] = "black_friday"
    for y in sorted({d.year for d in out.index}):
        for (m, dd, name) in [(12, 24, "christmas_eve"), (12, 26, "dec26"), (12, 31, "nye")]:
            d = pd.Timestamp(y, m, dd).date()
            if d in out.index and out.loc[d, "holiday"] == "none":
                out.loc[d, "holiday"] = name
    # Christmas-to-New-Year week is consistently low load
    idx = pd.DatetimeIndex(out.index)
    out["xmas_week"] = (((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 2))).astype(int)
    return out


def daily_slot_matrix(load: pd.DataFrame, col="ERCOT"):
    """op_date x slot(1..24) matrix by local clock hour-ending.

    Fall-back day: slot 2 = mean of the two HE02 rows. Spring-forward day: slot 3 (absent) =
    mean of slots 2 and 4 of the same day. Used only for *features*; targets keep real intervals.
    """
    m = load.groupby(["op_date", "he"])[col].mean().unstack("he")
    m.columns = [int(c) for c in m.columns]
    m = m.reindex(columns=range(1, 25))
    gap = m[3].isna()
    m.loc[gap, 3] = (m.loc[gap, 2] + m.loc[gap, 4]) / 2
    m.index = pd.to_datetime(m.index)
    return m


def build(load: pd.DataFrame, first_target="2017-02-01", last_target=None, col="ERCOT"):
    M = daily_slot_matrix(load, col)
    days = M.index
    tgt = load.copy()
    tgt["T"] = pd.to_datetime(tgt["op_date"])
    tgt = tgt[tgt["T"] >= pd.Timestamp(first_target)]
    if last_target:
        tgt = tgt[tgt["T"] <= pd.Timestamp(last_target)]
    tgt = tgt.rename(columns={col: "y"})
    tgt["slot"] = tgt["he"].astype(int)
    tgt["dst_dup"] = tgt["dst_tag"].astype(int)
    # issue time: 18:00 local on D = T-1
    issue_local = (tgt["T"] - pd.Timedelta(days=1) + pd.Timedelta(hours=18)).dt.tz_localize(TZ)
    tgt["issue_utc"] = issue_local.dt.tz_convert("UTC")
    tgt["horizon_h"] = (tgt["interval_end_utc"] - tgt["issue_utc"]).dt.total_seconds() / 3600
    T = tgt["T"].values
    slot = tgt["slot"].values

    def lookup(day_offset, slots=None):
        d = pd.DatetimeIndex(T) - pd.Timedelta(days=day_offset)
        r = days.get_indexer(d)
        s = (slot if slots is None else slots) - 1
        v = np.full(len(tgt), np.nan)
        ok = r >= 0
        v[ok] = M.values[r[ok], s[ok]]
        return v

    feats = {}
    for k in LAG_DAYS:
        feats[f"lag_d{k}"] = lookup(k)
    # neighbouring slots on T-2 and T-7 (shape information)
    for k in [2, 7]:
        feats[f"lag_d{k}_prev"] = lookup(k, np.clip(slot - 1, 1, 24))
        feats[f"lag_d{k}_next"] = lookup(k, np.clip(slot + 1, 1, 24))
    daily_mean = M.mean(axis=1)
    daily_max = M.max(axis=1)
    daily_min = M.min(axis=1)

    def day_stat(series, k):
        d = pd.DatetimeIndex(T) - pd.Timedelta(days=k)
        return series.reindex(d).values

    feats["d2_mean"] = day_stat(daily_mean, 2)
    feats["d2_max"] = day_stat(daily_max, 2)
    feats["d2_min"] = day_stat(daily_min, 2)
    feats["d2_last"] = day_stat(M[24], 2)
    feats["d3_mean"] = day_stat(daily_mean, 3)
    feats["d9_mean"] = day_stat(daily_mean, 9)  # same weekday as T-2, one week earlier
    roll7 = daily_mean.rolling(7, min_periods=7).mean()
    roll28 = daily_mean.rolling(28, min_periods=28).mean()
    feats["level7"] = day_stat(roll7, 2)          # mean of days T-8..T-2
    feats["level28"] = day_stat(roll28, 2)
    feats["level7_lastyear"] = day_stat(roll7, 2 + 364)
    F = pd.DataFrame(feats, index=tgt.index)
    tgt = pd.concat([tgt, F], axis=1)
    tgt["src_max_day"] = tgt["T"] - pd.Timedelta(days=2)

    # calendar (known in advance)
    hol = holiday_table(tgt["T"].min() - pd.Timedelta(days=400), tgt["T"].max() + pd.Timedelta(days=10))
    hol.index = pd.to_datetime(hol.index)
    tgt["holiday"] = hol["holiday"].reindex(tgt["T"]).values
    tgt["is_holiday"] = (tgt["holiday"] != "none").astype(int)
    tgt["xmas_week"] = hol["xmas_week"].reindex(tgt["T"]).values
    for k in [7, 2]:
        tgt[f"holiday_d{k}"] = (hol["holiday"].reindex(tgt["T"] - pd.Timedelta(days=k)).values != "none").astype(int)
    tgt["holiday_next"] = (hol["holiday"].reindex(tgt["T"] + pd.Timedelta(days=1)).values != "none").astype(int)
    tgt["holiday_prev"] = (hol["holiday"].reindex(tgt["T"] - pd.Timedelta(days=1)).values != "none").astype(int)
    tgt["dow"] = tgt["T"].dt.dayofweek
    tgt["month"] = tgt["T"].dt.month
    doy = tgt["T"].dt.dayofyear
    tgt["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    tgt["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    tgt["t_years"] = (tgt["T"] - pd.Timestamp("2017-01-01")).dt.days / 365.25
    tgt["is_dst"] = (tgt["utc_offset_h"] == -5).astype(int)
    return tgt.reset_index(drop=True)
