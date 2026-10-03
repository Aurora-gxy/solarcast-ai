"""Checks that weather-forecast inputs line up with the day they were available."""
import numpy as np
import pandas as pd

from solarcast.weather import WeatherRidge, align_to_origin, clim_columns, weather_inputs


def make_nwp(n=40, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="D", name="date")
    data = {}
    for k in range(1, 8):
        # Encode valid day and lead in the value so alignment can be checked exactly.
        data[f"nwp_rad_lead{k}"] = np.arange(n) * 10 + k
        data[f"nwp_cloud_lead{k}"] = np.arange(n) * 10 + k + 0.5
    return pd.DataFrame(data, index=dates)


def test_origin_row_holds_the_lead_h_forecast_for_day_t_plus_h():
    nwp = make_nwp()
    out = align_to_origin(nwp, nwp.index)
    origin = 5
    for h in range(1, 8):
        # forecast for valid day origin+h, issued h days earlier (on the origin day)
        assert out[f"nwp_rad_h{h}"].iloc[origin] == (origin + h) * 10 + h


def test_last_days_have_no_forecast_for_unknown_future():
    nwp = make_nwp(20)
    out = align_to_origin(nwp, nwp.index)
    assert np.isnan(out["nwp_rad_h7"].iloc[-1])


def make_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    f = pd.DataFrame(index=idx)
    f["solar_t"] = rng.uniform(5, 30, n)
    f["solar_mean_7d"] = rng.uniform(5, 30, n)
    f["day_sin"], f["day_cos"] = np.sin(np.arange(n) / 58), np.cos(np.arange(n) / 58)
    for h in range(1, 8):
        f[f"nwp_rad_h{h}"] = rng.uniform(5, 30, n)
        f[f"nwp_cloud_h{h}"] = rng.uniform(0, 100, n)
        f[f"clim_h{h}"] = 18.0
        f[f"solar_t_plus_{h}"] = 0.8 * f[f"nwp_rad_h{h}"] + 0.2 * 18 + rng.normal(0, 1, n)
    return f


def test_weather_ridge_learns_and_survives_save_and_load(tmp_path):
    f = make_frame()
    train, val, test = f.iloc[:180], f.iloc[180:240], f.iloc[240:]
    model = WeatherRidge().fit(train, val, [0.1, 1.0]).calibrate(val, 0.8)
    pred = model.predict(test)
    assert pred.shape == (60, 7)
    rmse = np.sqrt(np.mean((pred[:, 0] - test["solar_t_plus_1"]) ** 2))
    assert rmse < 1.5                      # noise level is 1.0
    model.save(tmp_path / "wr.json")
    again = WeatherRidge.load(tmp_path / "wr.json")
    np.testing.assert_allclose(pred, again.predict(test))
    mean, lower, upper = again.predict_interval(test)
    assert (lower <= mean).all() and (mean <= upper).all() and (lower >= 0).all()


def test_weather_inputs_use_only_the_matching_horizon():
    assert weather_inputs(3)[:3] == ["nwp_rad_h3", "nwp_cloud_h3", "clim_h3"]
    assert len(clim_columns()) == 7
