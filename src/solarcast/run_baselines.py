"""Run data checks and all baselines.  Usage: python -m solarcast.run_baselines"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .baselines import XGBMultiHorizon, fit_ridge, persistence_predict
from .config import load_config
from .dataset import build_dataset, uses_weather
from .evaluate import add_skill, evaluate, interval_metrics, summary_table
from .features import feature_names, target_names
from .validation import quality_report
from .weather import WeatherRidge, clim_columns, fit_climatology, nwp_columns, nwp_direct_predict


def main(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    reports = Path(cfg["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    # Steps 1-2: data, features and time-ordered split
    fcfg = cfg["features"]
    daily, frame, splits = build_dataset(cfg)
    report = quality_report(daily)
    (reports / "data_quality.json").write_text(json.dumps(report, indent=2))
    print(f"Samples  train={len(splits.train)}  val={len(splits.val)}  test={len(splits.test)}")

    # Step 3: baselines, all scored on the same test samples
    y_cols = target_names(fcfg["horizons"])
    y_test = splits.test[y_cols].to_numpy()
    bcfg = cfg["baselines"]
    results = []

    results.append(evaluate(y_test, persistence_predict(splits.test), "Persistence"))

    clim = fit_climatology(daily, cfg)
    results.append(evaluate(y_test, clim.predict(splits.test.index, fcfg["horizons"]), "Climatology"))

    ridge, alpha = fit_ridge(splits.train, splits.val, bcfg["ridge_alphas"])
    print(f"Ridge alpha chosen on validation: {alpha}")
    results.append(evaluate(y_test, ridge.predict(splits.test[feature_names()]), "Ridge"))

    xgb = XGBMultiHorizon(bcfg["xgboost"], cfg["seed"]).fit(splits.train, splits.val)
    results.append(evaluate(y_test, xgb.predict(splits.test), "XGBoost"))

    # Weather-aware models: only when archived weather forecasts are configured
    if uses_weather(cfg):
        results.append(evaluate(y_test, nwp_direct_predict(splits.test, fcfg["horizons"]), "NWP direct"))

        needed = nwp_columns(fcfg["horizons"]) + clim_columns(fcfg["horizons"])
        weather_train = splits.train.dropna(subset=needed)   # forecast archive starts in 2024
        print(f"Weather-aware training samples: {len(weather_train)}")
        wr = WeatherRidge(fcfg["horizons"]).fit(weather_train, splits.val, cfg["nwp"]["ridge_alphas"])
        wr.calibrate(splits.val, cfg["uncertainty"]["interval"])
        wr.meta.update({"nwp_model": cfg["nwp"]["model"], "split": cfg["split"],
                        "n_train": len(weather_train),
                        "trained_at": pd.Timestamp.now(tz="UTC").isoformat()})
        models_dir = Path(cfg["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        wr.save(models_dir / "weather_ridge.json")
        results.append(evaluate(y_test, wr.predict(splits.test), "Weather-aware Ridge"))
        _, lo, hi = wr.predict_interval(splits.test)
        interval_metrics(y_test, lo, hi, "weather_ridge:calibrated",
                         cfg["uncertainty"]["interval"]).to_csv(reports / "intervals_weather.csv", index=False)

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
