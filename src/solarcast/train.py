"""Train the neural models from the config file.  Usage: python -m solarcast.train

For each architecture, trains one model per random seed, scores every run on
the shared test set, and saves the run with the best validation loss as a
self-contained bundle (weights + scaling + interval calibration).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import load_config
from .data import load_daily
from .evaluate import add_skill, evaluate, interval_metrics, summary_table
from .features import build_features, feature_names, make_splits, target_names
from .models import build_model
from .predict import Forecaster


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(name: str, mcfg: dict, seed: int, Xtr, ytr, Xva, yva) -> tuple[dict, float, int]:
    """Train with early stopping. Returns (best weights, best val loss, epochs run)."""
    set_seed(seed)
    model = build_model(name, mcfg, Xtr.shape[1], ytr.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    loss_fn = nn.MSELoss()
    best_loss, best_state, wait, epoch = float("inf"), None, 0, 0
    n, bs = len(Xtr), mcfg["batch_size"]
    for epoch in range(1, mcfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(n)   # shuffles training batches only; time order of the split is untouched
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            optim.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, wait = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= mcfg["patience"]:
                break
    return best_state, best_loss, epoch


def main(config_path: str = "configs/config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    mcfg, fcfg, ucfg = cfg["model"], cfg["features"], cfg["uncertainty"]
    reports, models_dir = Path(cfg["reports_dir"]), Path(cfg["models_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    daily = load_daily(cfg)
    frame = build_features(daily, fcfg["n_lags"], fcfg["rolling_window"], fcfg["horizons"])
    splits = make_splits(frame, cfg)
    X_cols, y_cols = feature_names(fcfg["n_lags"]), target_names(fcfg["horizons"])

    # Scaling statistics come from the training set only.
    x_mean = splits.train[X_cols].mean().to_numpy()
    x_std = splits.train[X_cols].std(ddof=0).to_numpy()
    y_mean = splits.train[y_cols].mean().to_numpy()
    y_std = splits.train[y_cols].std(ddof=0).to_numpy()

    def to_x(df): return torch.tensor(((df[X_cols].to_numpy() - x_mean) / x_std), dtype=torch.float32)
    def to_y(df): return torch.tensor(((df[y_cols].to_numpy() - y_mean) / y_std), dtype=torch.float32)

    Xtr, ytr, Xva, yva = to_x(splits.train), to_y(splits.train), to_x(splits.val), to_y(splits.val)
    X_val_raw, y_val = splits.val[X_cols].to_numpy(), splits.val[y_cols].to_numpy()
    X_test_raw, y_test = splits.test[X_cols].to_numpy(), splits.test[y_cols].to_numpy()

    per_seed, summaries, intervals, best_overall = [], [], [], None
    for name in mcfg["architectures"]:
        best_for_arch = None
        for seed in mcfg["seeds"]:
            start = time.time()
            state, val_loss, epochs = train_one(name, mcfg, seed, Xtr, ytr, Xva, yva)
            seconds = time.time() - start
            bundle = {
                "architecture": name, "model_cfg": mcfg, "state_dict": state, "seed": seed,
                "feature_names": X_cols, "horizons": fcfg["horizons"],
                "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                "interval": ucfg["interval"], "val_loss": val_loss,
                "split": cfg["split"], "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            fc = Forecaster(bundle)
            # Interval half-width per horizon = the error size that covered the
            # nominal share of validation days (split conformal prediction).
            val_abs_err = np.abs(fc.predict(X_val_raw) - y_val)
            bundle["calibrated_half_width"] = np.quantile(val_abs_err, ucfg["interval"], axis=0)

            res = evaluate(y_test, fc.predict(X_test_raw), name)
            res["seed"], res["train_seconds"], res["epochs"] = seed, seconds, epochs
            per_seed.append(res)
            macro = res.loc[res["horizon"] == "macro", "RMSE"].item()
            print(f"{name:10s} seed={seed}  epochs={epochs:3d}  val_loss={val_loss:.4f}  "
                  f"test macro RMSE={macro:.2f}  ({seconds:.1f}s)")
            if best_for_arch is None or val_loss < best_for_arch["val_loss"]:
                best_for_arch = bundle

        # The saved model is picked by validation loss, never by test score.
        torch.save(best_for_arch, models_dir / f"{name}.pt")
        if best_overall is None or best_for_arch["val_loss"] < best_overall["val_loss"]:
            best_overall = best_for_arch

        fc = Forecaster(best_for_arch)
        for method in ("mc_dropout", "calibrated"):
            _, lo, hi = fc.predict_interval(X_test_raw, method, ucfg["mc_samples"])
            intervals.append(interval_metrics(y_test, lo, hi, f"{name}:{method}", ucfg["interval"]))

    torch.save(best_overall, models_dir / "best.pt")

    # Average each architecture over seeds; keep the spread of the macro RMSE.
    all_runs = pd.concat(per_seed, ignore_index=True)
    all_runs.to_csv(reports / "results_nn_per_seed.csv", index=False)
    label = {"mlp": "Multi-output MLP", "attention": "Attention MLP"}
    mean = (all_runs.groupby(["model", "horizon"], as_index=False)[["RMSE", "MAE", "MBE", "P95"]].mean())
    macro_runs = all_runs[all_runs["horizon"] == "macro"].groupby("model")
    seed_std = macro_runs["RMSE"].std(ddof=1).rename(index=label).to_dict()
    mean["model"] = mean["model"].map(label)

    baselines = pd.read_csv(reports / "results_baselines.csv")
    cols = ["model", "horizon", "RMSE", "MAE", "MBE", "P95"]
    combined = add_skill(pd.concat([baselines[cols], mean[cols]], ignore_index=True))
    combined["macro_rmse_seed_std"] = combined["model"].map(seed_std)
    combined.to_csv(reports / "results.csv", index=False)

    interval_table = pd.concat(intervals, ignore_index=True)
    interval_table.to_csv(reports / "intervals.csv", index=False)

    meta = {
        "best_model": best_overall["architecture"], "best_seed": best_overall["seed"],
        "n_train": len(splits.train), "n_val": len(splits.val), "n_test": len(splits.test),
        "test_period": [str(splits.test.index.min().date()), str(splits.test.index.max().date())],
        "macro_rmse_seed_std": seed_std,
        "mean_train_seconds": macro_runs["train_seconds"].mean().rename(index=label).round(2).to_dict(),
        "trained_at": best_overall["trained_at"],
    }
    (reports / "run_summary.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 200)
    print("\nRMSE by horizon (neural models averaged over seeds)")
    print(summary_table(combined, "RMSE").round(2))
    print("\nSkill vs climatology (positive = better)")
    print(summary_table(combined, "skill_vs_climatology").round(3))
    print("\nMacro RMSE spread across seeds:", {k: round(v, 3) for k, v in seed_std.items()})
    print(f"\n{int(ucfg['interval'] * 100)}% interval check on the test set")
    print(interval_table[interval_table["horizon"] == "macro"].round(3).to_string(index=False))
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    main(parser.parse_args().config)
