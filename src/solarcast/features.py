"""Feature and target construction on the calendar-aligned daily table.

Every feature for forecast origin day t uses information from day t or
earlier only. Targets are the trusted observations on days t+1 ... t+H.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def feature_names(n_lags: int = 7) -> list[str]:
    return (
        ["solar_t", "day_sin", "day_cos",
         "solar_mean_7d", "solar_std_7d", "solar_min_7d", "solar_max_7d"]
        + [f"solar_t_minus_{k}" for k in range(1, n_lags + 1)]
    )


def target_names(horizons: int = 7) -> list[str]:
    return [f"solar_t_plus_{h}" for h in range(1, horizons + 1)]


def build_features(daily: pd.DataFrame, n_lags: int = 7, window: int = 7,
                   horizons: int = 7, with_targets: bool = True) -> pd.DataFrame:
    """Return a frame indexed by forecast origin date with features and targets.

    Rows with any missing feature or target are dropped, so every model is
    trained and scored on exactly the same samples. Set `with_targets=False`
    when forecasting, where the future is not known yet.
    """
    x = daily["solar_input"]
    out = pd.DataFrame(index=daily.index)
    out["solar_t"] = x

    # Seasonal position on a circle, so 31 Dec sits next to 1 Jan.
    doy = daily.index.dayofyear
    days_in_year = np.where(daily.index.is_leap_year, 366, 365)
    out["day_sin"] = np.sin(2 * np.pi * doy / days_in_year)
    out["day_cos"] = np.cos(2 * np.pi * doy / days_in_year)

    # Trailing window: day t and the six days before it. Never centred.
    roll = x.rolling(window=window, min_periods=window)
    out["solar_mean_7d"] = roll.mean()
    out["solar_std_7d"] = roll.std(ddof=1)
    out["solar_min_7d"] = roll.min()
    out["solar_max_7d"] = roll.max()

    for k in range(1, n_lags + 1):
        out[f"solar_t_minus_{k}"] = x.shift(k)

    if not with_targets:
        return out[feature_names(n_lags)].dropna()

    # Targets come from trusted observations only, never from filled values.
    for h in range(1, horizons + 1):
        out[f"solar_t_plus_{h}"] = daily["solar_obs"].shift(-h)

    cols = feature_names(n_lags) + target_names(horizons)
    return out[cols].dropna()


def make_splits(frame: pd.DataFrame, cfg: dict) -> "Splits":
    """Split using the dates in the config file."""
    sp = cfg["split"]
    return temporal_split(frame, sp["train_end"], sp["val_end"],
                          cfg["features"]["horizons"], sp.get("test_start"))


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_split(frame: pd.DataFrame, train_end: str, val_end: str,
                   horizons: int = 7, test_start: str | None = None) -> Splits:
    """Split by forecast origin date, in time order.

    The last `horizons` days of train and val are removed so that no target
    of an earlier partition falls inside a later partition. `test_start` lets
    the test period begin later than the day after `val_end`.
    """
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    gap = pd.Timedelta(days=horizons)
    idx = frame.index
    train = frame[idx <= train_end - gap]
    val = frame[(idx > train_end) & (idx <= val_end - gap)]
    test = frame[idx >= pd.Timestamp(test_start)] if test_start else frame[idx > val_end]
    return Splits(train=train, val=val, test=test)
