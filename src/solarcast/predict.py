"""Load a trained model bundle and produce forecasts with prediction intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import load_daily
from .features import build_features, feature_names
from .models import build_model


class Forecaster:
    """Wraps network weights, feature scaling and interval calibration together.

    Keeping them in one bundle prevents the classic bug of pairing a model
    with a scaler that was fitted on different data.
    """

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.features = bundle["feature_names"]
        self.model = build_model(bundle["architecture"], bundle["model_cfg"],
                                 len(self.features), bundle["horizons"])
        self.model.load_state_dict(bundle["state_dict"])
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path) -> "Forecaster":
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def _scale(self, X: np.ndarray) -> torch.Tensor:
        z = (X - self.bundle["x_mean"]) / self.bundle["x_std"]
        return torch.tensor(z, dtype=torch.float32)

    def _unscale(self, y: torch.Tensor) -> np.ndarray:
        return y.numpy() * self.bundle["y_std"] + self.bundle["y_mean"]

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point forecast with dropout switched off. Shape (n, horizons)."""
        self.model.eval()
        return self._unscale(self.model(self._scale(X)))

    @torch.no_grad()
    def mc_dropout(self, X: np.ndarray, n_samples: int = 100, seed: int = 0) -> np.ndarray:
        """Repeated forecasts with dropout left on. Shape (n_samples, n, horizons)."""
        torch.manual_seed(seed)
        self.model.train()
        z = self._scale(X)
        draws = np.stack([self._unscale(self.model(z)) for _ in range(n_samples)])
        self.model.eval()
        return draws

    def predict_interval(self, X: np.ndarray, method: str = "calibrated",
                         n_samples: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, lower, upper).

        method="mc_dropout": spread of the dropout samples (model uncertainty only).
        method="calibrated": point forecast +/- the error size seen on the
            validation set for each horizon, clipped at zero.
        """
        if method == "mc_dropout":
            draws = self.mc_dropout(X, n_samples)
            alpha = (1 - self.bundle["interval"]) / 2
            mean = draws.mean(axis=0)
            lower, upper = np.quantile(draws, [alpha, 1 - alpha], axis=0)
        else:
            mean = self.predict(X)
            half = self.bundle["calibrated_half_width"]
            lower, upper = mean - half, mean + half
        return mean, np.clip(lower, 0, None), upper


def forecast_latest(cfg: dict, model_path: str | Path, origin: str | None = None) -> pd.DataFrame:
    """Seven-day forecast from the last usable day in the data (or a chosen day)."""
    daily = load_daily(cfg)
    fcfg = cfg["features"]
    feats = build_features(daily, fcfg["n_lags"], fcfg["rolling_window"],
                           fcfg["horizons"], with_targets=False)
    if origin is not None:
        feats = feats.loc[:origin]
    if feats.empty:
        raise ValueError("No day with a complete feature window is available.")
    row = feats.iloc[[-1]]
    fc = Forecaster.load(model_path)
    mean, lower, upper = fc.predict_interval(row[feature_names()].to_numpy())
    dates = row.index[0] + pd.to_timedelta(np.arange(1, fcfg["horizons"] + 1), unit="D")
    return pd.DataFrame({"date": dates, "horizon": [f"t+{h}" for h in range(1, len(dates) + 1)],
                         "mean": mean[0], "lower": lower[0], "upper": upper[0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--origin", default=None, help="forecast origin date, e.g. 2019-06-30")
    args = parser.parse_args()
    table = forecast_latest(load_config(args.config), args.model, args.origin)
    table["date"] = table["date"].dt.date
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
