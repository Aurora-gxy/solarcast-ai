"""Weather-forecast inputs and the models that use them.

A history-only model cannot know whether clouds arrive in three days. A weather
forecast can. For forecast origin day t and horizon h, the inputs here are what
the weather model predicted for day t+h, issued about h days in advance - the
information that was really available on day t, never the weather that later
happened.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import Climatology
from .features import target_names


def nwp_columns(horizons: int = 7) -> list[str]:
    return [f"{kind}_h{h}" for h in range(1, horizons + 1) for kind in ("nwp_rad", "nwp_cloud")]


def clim_columns(horizons: int = 7) -> list[str]:
    return [f"clim_h{h}" for h in range(1, horizons + 1)]


def weather_inputs(h: int) -> list[str]:
    """Inputs of the weather-aware model for horizon h."""
    return [f"nwp_rad_h{h}", f"nwp_cloud_h{h}", f"clim_h{h}",
            "solar_t", "solar_mean_7d", "day_sin", "day_cos"]


def load_nwp(data_dir: str | Path, model: str) -> pd.DataFrame:
    """Archived forecasts, one row per valid (target) day, columns per lead time."""
    path = Path(data_dir) / f"nwp_{model}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m solarcast.download")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def align_to_origin(nwp: pd.DataFrame, index: pd.DatetimeIndex, horizons: int = 7) -> pd.DataFrame:
    """Re-index forecasts by the day they were made.

    The file stores, for valid day d, the forecast issued k days earlier
    (`lead k`). The row for origin day t needs the lead-h forecast for day t+h,
    so the lead-h column is shifted back by h days.
    """
    nwp = nwp.reindex(index)
    out = pd.DataFrame(index=index)
    for h in range(1, horizons + 1):
        out[f"nwp_rad_h{h}"] = nwp[f"nwp_rad_lead{h}"].shift(-h)
        out[f"nwp_cloud_h{h}"] = nwp[f"nwp_cloud_lead{h}"].shift(-h)
    return out


def fit_climatology(daily: pd.DataFrame, cfg: dict) -> Climatology:
    """Seasonal normal from observations up to the end of the training period only."""
    train_obs = daily.loc[: cfg["split"]["train_end"], "solar_obs"]
    return Climatology(cfg["baselines"]["climatology_smooth_days"]).fit(train_obs)


def weather_features(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """NWP forecast and seasonal-normal columns, indexed by forecast origin day."""
    horizons = cfg["features"]["horizons"]
    nwp = load_nwp(cfg["data"]["dir"], cfg["nwp"]["model"])
    out = align_to_origin(nwp, daily.index, horizons)
    clim = fit_climatology(daily, cfg).predict(daily.index, horizons)
    return out.join(pd.DataFrame(clim, index=daily.index, columns=clim_columns(horizons)))


def nwp_direct_predict(frame: pd.DataFrame, horizons: int = 7) -> np.ndarray:
    """Use the weather model's own radiation forecast with no correction."""
    return frame[[f"nwp_rad_h{h}" for h in range(1, horizons + 1)]].to_numpy()


class WeatherRidge:
    """One small linear model per horizon that corrects the weather forecast.

    It learns how much to trust the weather model versus the seasonal normal:
    close to the forecast for tomorrow, closer to the normal for day 7.
    Parameters are stored as plain numbers, so predicting needs only NumPy.
    """

    def __init__(self, horizons: int = 7):
        self.horizons = horizons
        self.params: list[dict] = []
        self.half_width: np.ndarray | None = None
        self.meta: dict = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, alphas: list[float]) -> "WeatherRidge":
        from sklearn.linear_model import Ridge

        self.params = []
        for h in range(1, self.horizons + 1):
            cols, y_col = weather_inputs(h), f"solar_t_plus_{h}"
            mean = train[cols].mean().to_numpy()
            std = train[cols].std(ddof=0).to_numpy()
            std = np.where(std == 0, 1.0, std)   # a constant input must not divide by zero
            Xtr, Xva = (train[cols].to_numpy() - mean) / std, (val[cols].to_numpy() - mean) / std
            best = None
            for alpha in alphas:   # alpha is chosen on the validation set
                model = Ridge(alpha=alpha).fit(Xtr, train[y_col])
                rmse = float(np.sqrt(np.mean((model.predict(Xva) - val[y_col].to_numpy()) ** 2)))
                if best is None or rmse < best[0]:
                    best = (rmse, alpha, model)
            _, alpha, model = best
            self.params.append({"horizon": h, "inputs": cols, "alpha": alpha, "mean": mean.tolist(),
                                "std": std.tolist(), "coef": model.coef_.tolist(),
                                "intercept": float(model.intercept_)})
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for p in self.params:
            z = (frame[p["inputs"]].to_numpy() - np.array(p["mean"])) / np.array(p["std"])
            cols.append(z @ np.array(p["coef"]) + p["intercept"])
        return np.clip(np.column_stack(cols), 0, None)

    def calibrate(self, val: pd.DataFrame, interval: float) -> "WeatherRidge":
        """Size the prediction interval from validation errors.

        Errors are larger in summer, when radiation is high, than in winter. So
        the half-width is stored as a share of the seasonal normal for the
        target day, not as a fixed number: it widens and narrows with the season.
        """
        err = np.abs(self.predict(val) - val[target_names(self.horizons)].to_numpy())
        relative = err / val[clim_columns(self.horizons)].to_numpy()
        self.half_width = np.quantile(relative, interval, axis=0)
        self.meta["interval"] = interval
        return self

    def predict_interval(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = self.predict(frame)
        half = self.half_width * frame[clim_columns(self.horizons)].to_numpy()
        return mean, np.clip(mean - half, 0, None), mean + half

    def save(self, path: str | Path) -> None:
        payload = {"horizons": self.horizons, "params": self.params,
                   "half_width": None if self.half_width is None else self.half_width.tolist(),
                   "meta": self.meta}
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "WeatherRidge":
        payload = json.loads(Path(path).read_text())
        model = cls(payload["horizons"])
        model.params, model.meta = payload["params"], payload.get("meta", {})
        if payload.get("half_width") is not None:
            model.half_width = np.array(payload["half_width"])
        return model
