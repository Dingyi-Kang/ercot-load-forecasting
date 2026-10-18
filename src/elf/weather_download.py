"""Download free Open-Meteo data (non-commercial use, CC BY 4.0) for Texas cities.

- Forecast vintages: previous-runs API, `<var>_previous_day1/2` (empirically: run initialised at
  floor6(valid - 24*N h); see reports/weather_semantics.md). Models: gfs_global (temperature from
  2021-03-25; other vars from 2024), ecmwf_ifs025 (from 2024).
- Observed proxy: ERA5 reanalysis via the archive API (used for training-on-actuals and the
  separately labeled observed-weather backcast only).
Raw JSON responses are saved unchanged under data/weather/raw/.
"""
import json, sys, time, urllib.request, urllib.parse
from pathlib import Path

CITIES = {  # name: (lat, lon, ERCOT weather zone)
    "houston": (29.76, -95.37, "COAST"), "dallas": (32.78, -96.80, "NCENT"), "fortworth": (32.75, -97.33, "NCENT"),
    "sanantonio": (29.42, -98.49, "SCENT"), "austin": (30.27, -97.74, "SCENT"), "corpus": (27.80, -97.40, "SOUTH"),
    "mcallen": (26.20, -98.23, "SOUTH"), "midland": (31.99, -102.08, "FWEST"), "abilene": (32.45, -99.73, "WEST"),
    "tyler": (32.35, -95.30, "EAST"), "wichitafalls": (33.91, -98.49, "NORTH"),
}
FC_VARS_ALL = ["temperature_2m", "dew_point_2m", "cloud_cover", "shortwave_radiation", "wind_speed_10m"]
OBS_VARS = FC_VARS_ALL + ["relative_humidity_2m", "precipitation"]
OUT = Path("data/weather/raw")


def get(url, tries=6):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:
            wait = 30 * (i + 1)
            print("retry", i, e, "sleep", wait, flush=True)
            time.sleep(wait)
    raise RuntimeError(url)


def fetch(kind, model, city, start, end, variables):
    lat, lon, _ = CITIES[city]
    f = OUT / kind / model / f"{city}_{start}_{end}.json"
    if f.exists():
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    if kind == "obs":
        base = "https://archive-api.open-meteo.com/v1/archive"
        q = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(variables), timezone="GMT", models=model)
    else:
        base = "https://previous-runs-api.open-meteo.com/v1/forecast"
        hv = [f"{v}_previous_day{n}" for v in variables for n in (1, 2)]
        q = dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=",".join(hv), timezone="GMT", models=model)
    j = get(base + "?" + urllib.parse.urlencode(q))
    if j.get("error"):
        raise RuntimeError(f"{city} {start}: {j}")
    f.write_text(json.dumps(j))
    print("saved", f, flush=True)
    time.sleep(1.0)


def main():
    jobs = []
    for city in CITIES:
        # GFS temperature 2021-03-25 .. 2023-12-31 (other vars not archived before 2024)
        for (s, e) in [("2021-03-25", "2021-12-31"), ("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")]:
            jobs.append(("fcst", "gfs_global", city, s, e, ["temperature_2m"]))
        for y in [2024, 2025, 2026]:
            e = f"{y}-12-31" if y < 2026 else "2026-10-01"
            jobs.append(("fcst", "gfs_global", city, f"{y}-01-01", e, FC_VARS_ALL))
            jobs.append(("fcst", "ecmwf_ifs025", city, f"{y}-01-01", e, FC_VARS_ALL))
        for y in range(2016, 2027):
            s = "2016-12-01" if y == 2016 else f"{y}-01-01"
            e = "2016-12-31" if y == 2016 else (f"{y}-12-31" if y < 2026 else "2026-10-01")
            jobs.append(("obs", "era5", city, s, e, OBS_VARS))
    for j in jobs:
        fetch(*j)
    print("DONE", len(jobs), flush=True)


if __name__ == "__main__":
    main()
