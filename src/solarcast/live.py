"""Fetch today's weather forecast and build the inputs for a live 7-day forecast.

The forecast origin is yesterday, the last complete day. Horizon 1 is today and
horizon 7 is six days from now. Recent "observed" radiation comes from the
weather model's own analysis of the past week, because reanalysis data is
published with a delay of about five days.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MIN_HOURS_PER_DAY = 20


def hourly_to_daily(hourly: dict) -> pd.DataFrame:
    """Aggregate hourly radiation (W/m2) and cloud cover (%) to daily values.

    Same rule as the training data: daily radiation is the hourly mean power
    summed over the day and converted to MJ/m2; days with too few hours are missing.
    """
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.normalize()
    for col in ("shortwave_radiation", "cloud_cover"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("date")
    daily = pd.DataFrame({
        "rad": (grouped["shortwave_radiation"].sum() * 3600 / 1e6)
        .where(grouped["shortwave_radiation"].count() >= MIN_HOURS_PER_DAY),
        "cloud": grouped["cloud_cover"].mean().where(grouped["cloud_cover"].count() >= MIN_HOURS_PER_DAY),
    })
    return daily


def fetch_hourly(latitude: float, longitude: float, timezone: str, horizons: int = 7) -> dict:
    params = {"latitude": latitude, "longitude": longitude, "timezone": timezone,
              "hourly": "shortwave_radiation,cloud_cover", "past_days": 8, "forecast_days": horizons + 1}
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["hourly"]


def build_live_row(daily: pd.DataFrame, today: pd.Timestamp, climatology, horizons: int = 7) -> pd.DataFrame:
    """One row of model inputs for forecast origin `today - 1 day`.

    `daily` has columns rad and cloud indexed by date, covering at least the
    last 7 days and the next `horizons - 1` days. `climatology` is a fitted
    `solarcast.baselines.Climatology`.
    """
    origin = today - pd.Timedelta(days=1)
    past = daily.loc[origin - pd.Timedelta(days=6): origin, "rad"]
    if len(past) < 7 or past.isna().any():
        raise ValueError("The last 7 days of radiation are incomplete.")
    targets = origin + pd.to_timedelta(np.arange(1, horizons + 1), unit="D")
    future = daily.reindex(targets)
    if future[["rad", "cloud"]].isna().any().any():
        raise ValueError("The weather forecast does not cover all target days.")

    days_in_year = 366 if origin.is_leap_year else 365
    row = {"solar_t": float(past.iloc[-1]), "solar_mean_7d": float(past.mean()),
           "day_sin": float(np.sin(2 * np.pi * origin.dayofyear / days_in_year)),
           "day_cos": float(np.cos(2 * np.pi * origin.dayofyear / days_in_year))}
    clim = climatology.predict(pd.DatetimeIndex([origin]), horizons)[0]
    for h in range(1, horizons + 1):
        row[f"nwp_rad_h{h}"] = float(future["rad"].iloc[h - 1])
        row[f"nwp_cloud_h{h}"] = float(future["cloud"].iloc[h - 1])
        row[f"clim_h{h}"] = float(clim[h - 1])
    return pd.DataFrame([row], index=pd.DatetimeIndex([origin], name="date"))


def live_inputs(site: dict, climatology, horizons: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Return (input row, recent daily radiation) for a forecast starting today."""
    hourly = fetch_hourly(site["latitude"], site["longitude"], site["timezone"], horizons)
    daily = hourly_to_daily(hourly)
    today = pd.Timestamp(pd.Timestamp.now(tz=site["timezone"]).date())
    row = build_live_row(daily, today, climatology, horizons)
    return row, daily.loc[: row.index[0], "rad"].dropna()


if __name__ == "__main__":   # quick manual check:  python -m solarcast.live
    from .config import load_config
    from .data import load_daily
    from .weather import WeatherRidge, fit_climatology

    cfg = load_config("configs/sydney.yaml")
    clim = fit_climatology(load_daily(cfg), cfg)
    row, recent = live_inputs(cfg["data"]["site_coords"], clim)
    model = WeatherRidge.load(f"{cfg['models_dir']}/weather_ridge.json")
    mean, lower, upper = (a[0] for a in model.predict_interval(row))
    dates = row.index[0] + pd.to_timedelta(np.arange(1, 8), unit="D")
    print(f"Forecast origin {row.index[0].date()} (run on {date.today()})")
    print(pd.DataFrame({"date": dates.date, "mean": mean, "lower": lower, "upper": upper,
                        "cloud_%": [row[f'nwp_cloud_h{h}'].iloc[0] for h in range(1, 8)]})
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
