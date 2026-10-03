"""Build the modelling table and its time-ordered splits from a config file."""
from __future__ import annotations

import pandas as pd

from .data import load_daily
from .features import Splits, build_features, make_splits


def uses_weather(cfg: dict) -> bool:
    return "nwp" in cfg


def build_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, Splits]:
    """Return (daily table, feature/target frame, splits).

    When weather-forecast inputs are configured, validation and test rows
    without a complete set of forecasts are dropped, so every model, with or
    without weather inputs, is scored on exactly the same days.
    """
    f = cfg["features"]
    daily = load_daily(cfg)
    frame = build_features(daily, f["n_lags"], f["rolling_window"], f["horizons"])
    if uses_weather(cfg):
        from .weather import clim_columns, nwp_columns, weather_features

        frame = frame.join(weather_features(daily, cfg))
        needed = nwp_columns(f["horizons"]) + clim_columns(f["horizons"])
        complete = frame[needed].notna().all(axis=1)
        in_training_period = frame.index <= pd.Timestamp(cfg["split"]["train_end"])
        frame = frame[in_training_period | complete]
    return daily, frame, make_splits(frame, cfg)
