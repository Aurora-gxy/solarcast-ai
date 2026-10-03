"""Checks for the mistakes that never raise an error but quietly inflate scores."""
import numpy as np
import pandas as pd
import pytest

from solarcast.data import to_daily_calendar
from solarcast.features import build_features, feature_names, target_names, temporal_split
from solarcast.validation import DataValidationError, validate_raw


def make_raw(n=120, bad=(), start="2020-01-01", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    df = pd.DataFrame({"date": dates, "solar_t": rng.uniform(20, 300, n), "bad_quality": 0})
    df.loc[list(bad), "bad_quality"] = 1
    return df


def test_validation_rejects_duplicate_dates():
    raw = make_raw(10)
    raw.loc[3, "date"] = raw.loc[2, "date"]
    with pytest.raises(DataValidationError):
        validate_raw(raw)


def test_bad_day_keeps_its_calendar_slot():
    daily = to_daily_calendar(make_raw(30, bad=[10]), max_fill_gap=0)
    assert len(daily) == 30
    assert np.isnan(daily["solar_obs"].iloc[10])


def test_lag_is_previous_calendar_day_not_previous_valid_row():
    raw = make_raw(60, bad=[20])
    daily = to_daily_calendar(raw, max_fill_gap=0)
    lag1 = daily["solar_input"].shift(1)
    # The day after the bad day must see a missing "yesterday", not the day before it.
    assert np.isnan(lag1.iloc[21])


def test_gap_filling_uses_only_past_values():
    raw = make_raw(60, bad=[20])
    daily = to_daily_calendar(raw, max_fill_gap=2)
    assert daily["solar_input"].iloc[20] == daily["solar_obs"].iloc[19]


def test_targets_never_use_filled_values():
    raw = make_raw(120, bad=[50])
    daily = to_daily_calendar(raw, max_fill_gap=2)
    frame = build_features(daily)
    bad_date = daily.index[50]
    for h, col in enumerate(target_names(), start=1):
        assert (bad_date - pd.Timedelta(days=h)) not in frame.index, col


def test_features_do_not_depend_on_the_future():
    raw = make_raw(120)
    base = build_features(to_daily_calendar(raw))
    changed = raw.copy()
    changed.loc[80:, "solar_t"] = 9999.0
    other = build_features(to_daily_calendar(changed))
    cutoff = raw.loc[80, "date"]
    cols = feature_names()
    common = base.index[base.index < cutoff].intersection(other.index)
    pd.testing.assert_frame_equal(base.loc[common, cols], other.loc[common, cols])


def test_split_partitions_do_not_overlap_including_targets():
    frame = build_features(to_daily_calendar(make_raw(400)))
    s = temporal_split(frame, "2020-06-30", "2020-10-31", horizons=7)
    week = pd.Timedelta(days=7)
    assert s.train.index.max() + week <= pd.Timestamp("2020-06-30")
    assert s.val.index.min() > pd.Timestamp("2020-06-30")
    assert s.val.index.max() + week <= pd.Timestamp("2020-10-31")
    assert s.test.index.min() > pd.Timestamp("2020-10-31")


def test_feature_schema_is_fixed():
    frame = build_features(to_daily_calendar(make_raw(120)))
    assert list(frame.columns) == feature_names() + target_names()
    assert len(feature_names()) == 14


def test_test_start_leaves_a_gap_after_validation():
    frame = build_features(to_daily_calendar(make_raw(400)))
    s = temporal_split(frame, "2020-06-30", "2020-09-30", horizons=7, test_start="2020-11-01")
    assert s.val.index.max() < pd.Timestamp("2020-09-30")
    assert s.test.index.min() >= pd.Timestamp("2020-11-01")


def test_forecast_features_need_no_future_values():
    daily = to_daily_calendar(make_raw(60))
    feats = build_features(daily, with_targets=False)
    assert feats.index.max() == daily.index.max()
    assert list(feats.columns) == feature_names()
