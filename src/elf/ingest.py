"""Ingest ERCOT Native Load yearly archives into an hourly UTC table.

Timestamp convention (verified 2026-10-13 against the files):
- First column is "Hour Ending" in local prevailing time (America/Chicago), "24:00" = midnight.
- Spring-forward days have 23 rows (HE03 absent); fall-back days have 25 rows with an extra
  "02:00 DST" row. Row order is chronological (smooth load sequence across years), so the
  "DST"-tagged row is the second (CST) occurrence of local 01:00-02:00.
We therefore assign each operating day's rows, in file order, to that day's real UTC intervals.
"""
import glob, hashlib, json, re, sys, zipfile
from pathlib import Path
import pandas as pd

TZ = "America/Chicago"
ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST", "ERCOT"]


def _parse_label(x: str):
    x = str(x).strip()
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4}) (\d{2}):00( DST)?$", x)
    if m:
        mo, d, y, h, dst = m.groups()
        return pd.Timestamp(int(y), int(mo), int(d)).date(), int(h), bool(dst)
    # one 2022 cell was stored by Excel as a datetime ("2022-12-01 01:00:00")
    t = pd.Timestamp(x)
    if t.minute or t.second:
        raise ValueError(f"unexpected timestamp {x!r}")
    if t.hour == 0:
        return (t - pd.Timedelta(days=1)).date(), 24, False
    return t.date(), t.hour, False


def day_intervals_utc(op_date):
    start = pd.Timestamp(op_date).tz_localize(TZ)
    end = (pd.Timestamp(op_date) + pd.Timedelta(days=1)).tz_localize(TZ)
    return pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")


def read_archive(path):
    with zipfile.ZipFile(path) as zf:
        df = pd.read_excel(zf.open(zf.namelist()[0]), dtype=str)
    df = df.rename(columns={df.columns[0]: "label"})
    parsed = df["label"].map(_parse_label)
    df["op_date"] = [p[0] for p in parsed]
    df["he"] = [p[1] for p in parsed]
    df["dst_tag"] = [p[2] for p in parsed]
    for z in ZONES:
        df[z] = df[z].astype(float)
    out = []
    for d, g in df.groupby("op_date", sort=True):
        iv = day_intervals_utc(d)
        if len(iv) != len(g):
            raise ValueError(f"{path}: {d} has {len(g)} rows but {len(iv)} real hours")
        g = g.copy()
        g["interval_start_utc"] = iv
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main(raw_dir="data/raw/ercot_native_load", out="data/processed/load_hourly.parquet"):
    files = sorted(f for f in glob.glob(f"{raw_dir}/*.zip") if "2016" not in f)
    parts, manifest = [], []
    for f in files:
        part = read_archive(f)
        part["source_file"] = Path(f).name
        parts.append(part)
        manifest.append({"file": Path(f).name, "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(),
                         "rows": len(part), "first_op_date": str(part.op_date.min()), "last_op_date": str(part.op_date.max())})
    df = pd.concat(parts, ignore_index=True).sort_values("interval_start_utc")
    dup = df.duplicated("interval_start_utc").sum()
    assert dup == 0, f"{dup} duplicate UTC intervals"
    full = pd.date_range(df.interval_start_utc.min(), df.interval_start_utc.max(), freq="h")
    missing = full.difference(pd.DatetimeIndex(df.interval_start_utc))
    df["interval_end_utc"] = df["interval_start_utc"] + pd.Timedelta(hours=1)
    loc = df["interval_start_utc"].dt.tz_convert(TZ)
    df["local_start"] = loc.dt.tz_localize(None)
    df["utc_offset_h"] = (loc.dt.tz_localize(None) - df["interval_start_utc"].dt.tz_localize(None)).dt.total_seconds() / 3600
    cols = ["interval_start_utc", "interval_end_utc", "op_date", "he", "dst_tag", "local_start", "utc_offset_h"] + ZONES + ["source_file"]
    df[cols].to_parquet(out, index=False)
    report = {"rows": len(df), "first_interval_utc": str(df.interval_start_utc.min()), "last_interval_utc": str(df.interval_start_utc.max()),
              "missing_utc_hours": [str(x) for x in missing], "duplicate_utc_intervals": int(dup),
              "nonpositive_ercot_rows": int((df.ERCOT <= 0).sum()),
              "zone_sum_minus_total_max_abs_mw": float((df[ZONES[:-1]].sum(axis=1) - df.ERCOT).abs().max()),
              "files": manifest}
    Path("data/processed/ingest_manifest.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
