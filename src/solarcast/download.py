"""Download public Sydney data from Open-Meteo.  Usage: python -m solarcast.download

Two datasets are fetched:

1. Observed daily solar radiation (ERA5 reanalysis) from the Historical Weather
   API. This is the prediction target and the source of history features.
2. Archived weather forecasts from the Previous Runs API: what the weather
   model predicted 1 to 7 days ahead for each day. Using the forecast that was
   really available at the time (not the weather that later happened) keeps
   the experiment free of leakage.

Data licence: CC BY 4.0, Open-Meteo.com. Free for non-commercial use.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

SITE = {"name": "Sydney", "latitude": -33.87, "longitude": 151.21, "timezone": "Australia/Sydney"}
OBS_START = "2000-01-01"
FORECAST_START = "2024-01-01"          # Previous Runs archive begins here for most models
LEADS = range(1, 8)
NWP_MODELS = ["best_match", "gfs_seamless", "ecmwf_ifs025"]
OBS_DAILY = ["shortwave_radiation_sum", "sunshine_duration", "daylight_duration",
             "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
MIN_HOURS_PER_DAY = 20                 # a day with fewer forecast hours is treated as missing


def get_json(url: str, params: dict, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=90)
            if resp.status_code == 200:
                return resp.json()
            reason = resp.json().get("reason", resp.text[:200]) if resp.text else resp.status_code
            if resp.status_code == 400:
                raise ValueError(f"API rejected the request: {reason}")
            print(f"  HTTP {resp.status_code}: {reason}. Retrying...")
        except requests.RequestException as exc:
            print(f"  Network error: {exc}. Retrying...")
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {url}")


def base_params() -> dict:
    return {"latitude": SITE["latitude"], "longitude": SITE["longitude"], "timezone": SITE["timezone"]}


def download_observed(end: date) -> pd.DataFrame:
    params = base_params() | {"start_date": OBS_START, "end_date": end.isoformat(),
                              "daily": ",".join(OBS_DAILY)}
    data = get_json(ARCHIVE_URL, params)
    df = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def half_year_chunks(start: date, end: date):
    cur = start
    while cur <= end:
        nxt = min(date(cur.year + (cur.month > 6), 1 if cur.month > 6 else 7, 1) - timedelta(days=1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def download_forecasts(model: str, end: date) -> pd.DataFrame:
    """Hourly lead-time forecasts, aggregated to one row per valid (target) day."""
    hourly_vars = [f"{v}_previous_day{k}" for v in ("shortwave_radiation", "cloud_cover") for k in LEADS]
    frames = []
    for start_chunk, end_chunk in half_year_chunks(date.fromisoformat(FORECAST_START), end):
        params = base_params() | {"start_date": start_chunk.isoformat(), "end_date": end_chunk.isoformat(),
                                  "hourly": ",".join(hourly_vars)}
        if model != "best_match":
            params["models"] = model
        data = get_json(PREVIOUS_RUNS_URL, params)
        frames.append(pd.DataFrame(data["hourly"]))
        time.sleep(0.5)
    hourly = pd.concat(frames, ignore_index=True)
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.normalize()
    for col in hourly_vars:
        if col not in hourly:
            hourly[col] = float("nan")
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

    grouped = hourly.groupby("date")
    daily = pd.DataFrame(index=grouped.size().index)
    for k in LEADS:
        rad, cloud = f"shortwave_radiation_previous_day{k}", f"cloud_cover_previous_day{k}"
        enough_rad = grouped[rad].count() >= MIN_HOURS_PER_DAY
        enough_cloud = grouped[cloud].count() >= MIN_HOURS_PER_DAY
        # Hourly mean W/m2 summed over the day, converted to MJ/m2.
        daily[f"nwp_rad_lead{k}"] = (grouped[rad].sum() * 3600 / 1e6).where(enough_rad)
        daily[f"nwp_cloud_lead{k}"] = grouped[cloud].mean().where(enough_cloud)
    return daily.reset_index()


def main(out_dir: str = "data/open_meteo") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    end = date.today() - timedelta(days=7)      # reanalysis is published with a delay of about 5 days
    report = {"site": SITE, "downloaded_on": date.today().isoformat(), "end_date": end.isoformat(),
              "source": "Open-Meteo.com (CC BY 4.0)", "models": {}}

    print(f"Observed daily radiation {OBS_START} to {end} ...")
    obs = download_observed(end)
    obs.to_csv(out / "observed_daily.csv", index=False)
    report["observed"] = {"rows": len(obs),
                          "missing_radiation_days": int(obs["shortwave_radiation_sum"].isna().sum()),
                          "mean_radiation_MJ_m2": round(float(obs["shortwave_radiation_sum"].mean()), 2)}
    print(f"  {len(obs)} days, {report['observed']['missing_radiation_days']} missing")

    for model in NWP_MODELS:
        print(f"Archived forecasts ({model}) {FORECAST_START} to {end} ...")
        try:
            fc = download_forecasts(model, end)
        except (ValueError, RuntimeError) as exc:
            print(f"  skipped: {exc}")
            report["models"][model] = {"error": str(exc)}
            continue
        fc.to_csv(out / f"nwp_{model}_daily.csv", index=False)
        coverage = {f"lead{k}": round(float(fc[f"nwp_rad_lead{k}"].notna().mean()), 3) for k in LEADS}
        first_full = fc.dropna(subset=[f"nwp_rad_lead{k}" for k in LEADS])["date"].min()
        report["models"][model] = {"rows": len(fc), "radiation_coverage_by_lead": coverage,
                                   "first_day_with_all_leads": str(first_full)[:10]}
        print(f"  {len(fc)} days; share of days with a radiation forecast by lead: {coverage}")
        print(f"  first day with all seven leads: {str(first_full)[:10]}")

    (out / "download_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved to {out}/ . Summary in download_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/open_meteo")
    main(parser.parse_args().out)
