"""Live-forecast input building, tested with a synthetic weather payload (no network)."""
import numpy as np
import pandas as pd
import pytest

from solarcast.baselines import Climatology
from solarcast.live import build_live_row, hourly_to_daily
from solarcast.weather import weather_inputs


def make_hourly(start="2026-09-20", days=16):
    times = pd.date_range(start, periods=days * 24, freq="h")
    # 1000 W/m2 for 6 hours a day = 21.6 MJ/m2; cloud cover equals the day number
    rad = np.where((times.hour >= 9) & (times.hour < 15), 1000.0, 0.0)
    cloud = (times.normalize() - times[0].normalize()).days.astype(float)
    return {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(), "shortwave_radiation": rad.tolist(),
            "cloud_cover": cloud.tolist()}


def make_climatology():
    idx = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    return Climatology(15).fit(pd.Series(15.0, index=idx))


def test_hourly_power_converts_to_daily_energy():
    daily = hourly_to_daily(make_hourly())
    assert daily["rad"].iloc[0] == pytest.approx(21.6)


def test_incomplete_day_is_missing():
    hourly = make_hourly(days=2)
    for key in hourly:
        hourly[key] = hourly[key][:30]          # second day has only 6 hours
    daily = hourly_to_daily(hourly)
    assert np.isnan(daily["rad"].iloc[1])


def test_live_row_uses_yesterday_as_origin_and_today_as_horizon_one():
    daily = hourly_to_daily(make_hourly())       # day numbers 0..15 from 2026-09-20
    today = pd.Timestamp("2026-09-28")           # day 8
    row = build_live_row(daily, today, make_climatology())
    assert row.index[0] == pd.Timestamp("2026-09-27")
    assert row["nwp_cloud_h1"].iloc[0] == 8.0    # today
    assert row["nwp_cloud_h7"].iloc[0] == 14.0   # six days from now
    assert row["solar_mean_7d"].iloc[0] == pytest.approx(21.6)
    for h in range(1, 8):
        assert set(weather_inputs(h)) <= set(row.columns)


def test_live_row_fails_loudly_when_forecast_is_too_short():
    daily = hourly_to_daily(make_hourly(days=10))
    with pytest.raises(ValueError):
        build_live_row(daily, pd.Timestamp("2026-09-28"), make_climatology())
