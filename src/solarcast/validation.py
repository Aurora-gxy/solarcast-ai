"""Data validation: fail loudly on bad input, and summarise data quality."""
from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    """Raised when the raw data breaks a rule the pipeline depends on."""


def validate_raw(df: pd.DataFrame) -> None:
    """Check the raw table (one row per recorded day) before any processing."""
    required = {"date", "solar_t", "bad_quality"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")
    if df["date"].isna().any():
        raise DataValidationError("Unparseable dates found.")
    if df["date"].duplicated().any():
        dup = df.loc[df["date"].duplicated(), "date"].dt.date.tolist()[:5]
        raise DataValidationError(f"Duplicate dates, e.g. {dup}")
    if not df["date"].is_monotonic_increasing:
        raise DataValidationError("Dates are not sorted in increasing order.")
    if not set(df["bad_quality"].dropna().unique()) <= {0, 1}:
        raise DataValidationError("bad_quality must be 0 or 1.")
    if (df["solar_t"].dropna() < 0).any():
        raise DataValidationError("Negative solar radiation values found.")


def _longest_run(mask: pd.Series) -> int:
    """Length of the longest run of True values."""
    if not mask.any():
        return 0
    groups = (mask != mask.shift()).cumsum()
    return int(mask.groupby(groups).sum().max())


def quality_report(daily: pd.DataFrame) -> dict:
    """Summarise the calendar-aligned table produced by `data.to_daily_calendar`."""
    unusable = daily["solar_obs"].isna()
    per_year = unusable.groupby(daily.index.year).mean().round(4)
    return {
        "start_date": str(daily.index.min().date()),
        "end_date": str(daily.index.max().date()),
        "calendar_days": int(len(daily)),
        "days_missing_from_files": int(daily["missing_row"].sum()),
        "days_flagged_bad_quality": int(daily["bad_quality"].eq(1).sum()),
        "days_unusable_total": int(unusable.sum()),
        "share_unusable": round(float(unusable.mean()), 4),
        "longest_unusable_run_days": _longest_run(unusable),
        "input_days_forward_filled": int(daily["was_filled"].sum()),
        "share_unusable_by_year": {int(k): float(v) for k, v in per_year.items()},
    }
