"""Tests: timestamp/DST handling, cutoff, metrics hand checks."""
import numpy as np
import pandas as pd
import pytest
from src.elf import features as F
from src.elf.metrics import point_metrics
from src.elf.ingest import day_intervals_utc

LOAD = "data/processed/load_hourly.parquet"


@pytest.fixture(scope="module")
def load():
    return pd.read_parquet(LOAD)


def test_dst_day_lengths():
    assert len(day_intervals_utc("2024-03-10")) == 23
    assert len(day_intervals_utc("2024-11-03")) == 25
    assert len(day_intervals_utc("2024-07-01")) == 24


def test_unique_contiguous_utc(load):
    s = load["interval_start_utc"]
    assert s.is_unique and s.is_monotonic_increasing
    assert (s.diff().dropna() == pd.Timedelta(hours=1)).all()


def test_fall_back_order(load):
    d = load[load["op_date"].astype(str) == "2024-11-03"]
    assert len(d) == 25
    he2 = d[d["he"] == 2]
    assert he2["dst_tag"].tolist() == [False, True]          # DST-tagged row is the second occurrence
    assert he2["utc_offset_h"].tolist() == [-5.0, -6.0]      # CDT first, then CST


def test_metrics_hand():
    df = pd.DataFrame({"y": [100.0, 200.0], "pred": [90.0, 240.0]})
    m = point_metrics(df)
    assert m["mae"] == pytest.approx(25.0)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 1600) / 2))
    assert m["mape"] == pytest.approx((10 / 100 + 40 / 200) / 2 * 100)
    assert m["bias"] == pytest.approx(15.0)


def test_cutoff_and_horizon(load):
    df = F.build(load, first_target="2024-11-01", last_target="2024-11-10")
    # all lag sources are at least two operating days before the target day
    assert (df["src_max_day"] <= df["T"] - pd.Timedelta(days=2)).all()
    # issue is 18:00 local of T-1, so HE01 horizon is 7h, HE24 is 30h on normal days
    normal = df[df["T"] == pd.Timestamp("2024-11-05")]
    assert normal["horizon_h"].min() == 7 and normal["horizon_h"].max() == 30
    # lag_d2 equals actual load on T-2 same slot
    r = df[(df["T"] == pd.Timestamp("2024-11-07")) & (df["slot"] == 15)].iloc[0]
    src = load[(load["op_date"].astype(str) == "2024-11-05") & (load["he"] == 15)]["ERCOT"].iloc[0]
    assert r["lag_d2"] == pytest.approx(src)


def test_injected_future_value_does_not_change_features(load):
    """Corrupting load on T-1 and T must not change any feature of target day T."""
    T = pd.Timestamp("2024-08-15")
    a = F.build(load, first_target=str(T.date()), last_target=str(T.date()))
    bad = load.copy()
    mask = pd.to_datetime(bad["op_date"]).isin([T, T - pd.Timedelta(days=1)])
    bad.loc[mask, "ERCOT"] = 1e9
    b = F.build(bad, first_target=str(T.date()), last_target=str(T.date()))
    feat = [c for c in a.columns if c.startswith(("lag_", "d2_", "d3_", "d9_", "level"))]
    pd.testing.assert_frame_equal(a[feat], b[feat])


def test_weather_vintage_never_after_issue(load):
    """For every target interval, the forecast run used (day1 if allowed else day2) was published
    (init + delay) no later than the 18:00 issue; day2 is always allowed."""
    from src.elf import weather as W
    df = F.build(load, first_target="2024-03-01", last_target="2024-11-30")   # covers both DST switches
    ends = pd.DatetimeIndex(df["interval_start_utc"]) + pd.Timedelta(hours=1)
    issue = pd.DatetimeIndex(df["issue_utc"])
    for model in ("gfs_global", "ecmwf_ifs025"):
        delay = pd.Timedelta(hours=W.PUB_DELAY_BY_MODEL[model])
        run1 = (ends - pd.Timedelta(hours=24)).floor("6h")
        run2 = (ends - pd.Timedelta(hours=48)).floor("6h")
        ok1 = run1 + delay <= issue
        used = np.where(ok1, run1, run2)
        assert (pd.DatetimeIndex(used) + delay <= issue).all()
        assert (run2 + delay <= issue).all()
    # the issue itself is 18:00 local the day before the target day
    loc = issue.tz_convert("America/Chicago")
    assert (loc.hour == 18).all()
    assert ((pd.DatetimeIndex(df["T"]) - pd.DatetimeIndex(loc.tz_localize(None).normalize())).days == 1).all()
