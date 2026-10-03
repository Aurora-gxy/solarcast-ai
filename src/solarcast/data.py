"""Load raw yearly files and align them to a complete daily calendar.

Why this module exists
----------------------
Dropping bad-quality rows and re-numbering the table makes "previous row" no
longer mean "previous day", which silently corrupts lag and rolling features.
Here every calendar day keeps its own row; unusable days become NaN instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import validate_raw


def load_raw(raw_dir: str | Path, pattern: str = "daily_solar_*.csv") -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {raw_dir}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    validate_raw(df)
    return df


def to_daily_calendar(raw: pd.DataFrame, max_fill_gap: int = 2) -> pd.DataFrame:
    """Return one row per calendar day.

    Columns
    -------
    solar_obs   trusted observation; NaN if the day is missing or flagged bad.
                This is the only column ever used as a prediction target.
    solar_input model input series: `solar_obs` with gaps of at most
                `max_fill_gap` days forward-filled. Forward fill only uses past
                values, so it cannot leak future information (linear
                interpolation would, because it looks at the next valid day).
    was_filled  True where `solar_input` was filled rather than observed.
    missing_row True where the day was absent from the raw files.
    """
    df = raw.set_index("date")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    daily = df.reindex(full_index)

    daily["missing_row"] = daily["solar_t"].isna() & daily["bad_quality"].isna()
    daily["solar_obs"] = daily["solar_t"].where(daily["bad_quality"] == 0)
    if max_fill_gap > 0:
        daily["solar_input"] = daily["solar_obs"].ffill(limit=max_fill_gap)
    else:
        daily["solar_input"] = daily["solar_obs"]
    daily["was_filled"] = daily["solar_obs"].isna() & daily["solar_input"].notna()
    return daily[["solar_obs", "solar_input", "bad_quality", "missing_row", "was_filled"]]


def load_open_meteo(data_dir: str | Path) -> pd.DataFrame:
    """Observed daily radiation downloaded by `solarcast.download`, in the raw-table layout."""
    path = Path(data_dir) / "observed_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m solarcast.download")
    obs = pd.read_csv(path, parse_dates=["date"])
    raw = pd.DataFrame({"date": obs["date"], "solar_t": obs["shortwave_radiation_sum"]})
    raw["bad_quality"] = raw["solar_t"].isna().astype(int)
    raw = raw.sort_values("date").reset_index(drop=True)
    validate_raw(raw)
    return raw


def load_daily(cfg: dict) -> pd.DataFrame:
    dcfg = cfg["data"]
    if dcfg.get("source", "csv") == "open_meteo":
        raw = load_open_meteo(dcfg["dir"])
    else:
        raw = load_raw(dcfg["raw_dir"], dcfg["pattern"])
    return to_daily_calendar(raw, dcfg["max_fill_gap"])
