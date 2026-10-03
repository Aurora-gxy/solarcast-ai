"""Run data checks and all baselines.  Usage: python -m solarcast.run_baselines"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baselines import Climatology, XGBMultiHorizon, fit_ridge, persistence_predict
from .config import load_config
from .data import load_daily
from .evaluate import add_skill, evaluate, summary_table
from .features import build_features, feature_names, make_splits, target_names
from .validation import quality_report


def main(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    reports = Path(cfg["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    # Step 1: data
    daily = load_daily(cfg)
    report = quality_report(daily)
    (reports / "data_quality.json").write_text(json.dumps(report, indent=2))

    # Step 2: features and time-ordered split
    fcfg = cfg["features"]
    frame = build_features(daily, fcfg["n_lags"], fcfg["rolling_window"], fcfg["horizons"])
    splits = make_splits(frame, cfg)
    print(f"Samples  train={len(splits.train)}  val={len(splits.val)}  test={len(splits.test)}")

    # Step 3: baselines, all scored on the same test samples
    y_cols = target_names(fcfg["horizons"])
    y_test = splits.test[y_cols].to_numpy()
    bcfg = cfg["baselines"]
    results = []

    results.append(evaluate(y_test, persistence_predict(splits.test), "Persistence"))

    train_obs = daily.loc[: cfg["split"]["train_end"], "solar_obs"]
    clim = Climatology(bcfg["climatology_smooth_days"]).fit(train_obs)
    results.append(evaluate(y_test, clim.predict(splits.test.index), "Climatology"))

    ridge, alpha = fit_ridge(splits.train, splits.val, bcfg["ridge_alphas"])
    print(f"Ridge alpha chosen on validation: {alpha}")
    results.append(evaluate(y_test, ridge.predict(splits.test[feature_names()]), "Ridge"))

    xgb = XGBMultiHorizon(bcfg["xgboost"], cfg["seed"]).fit(splits.train, splits.val)
    results.append(evaluate(y_test, xgb.predict(splits.test), "XGBoost"))

    out = add_skill(pd.concat(results, ignore_index=True))
    out.to_csv(reports / "results_baselines.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nRMSE by horizon")
    print(summary_table(out, "RMSE").round(2))
    print("\nSkill vs climatology (positive = better)")
    print(summary_table(out, "skill_vs_climatology").round(3))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    main(parser.parse_args().config)
