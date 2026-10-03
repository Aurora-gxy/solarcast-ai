"""Simple reference models. A complex model only earns its place by beating these."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import feature_names, target_names

# scikit-learn and XGBoost are imported inside the functions that use them.
# On macOS, loading XGBoost and PyTorch in one process can crash (both bundle
# their own OpenMP runtime), and the dashboard only needs Climatology from here.


def persistence_predict(frame: pd.DataFrame, horizons: int = 7) -> np.ndarray:
    """Every future day equals today."""
    return np.repeat(frame[["solar_t"]].to_numpy(), horizons, axis=1)


class Climatology:
    """Mean of the same calendar day across the training years, smoothed.

    Uses only the calendar, so its error barely grows with the horizon.
    """

    def __init__(self, smooth_days: int = 15):
        self.smooth_days = smooth_days
        self.table_: pd.Series | None = None

    def fit(self, train_obs: pd.Series) -> "Climatology":
        obs = train_obs.dropna()
        by_day = obs.groupby(obs.index.dayofyear).mean().reindex(range(1, 367))
        by_day = by_day.interpolate(limit_direction="both")
        # Circular smoothing so late December blends into early January.
        pad = self.smooth_days
        wrapped = pd.concat([by_day.iloc[-pad:], by_day, by_day.iloc[:pad]])
        smooth = wrapped.rolling(self.smooth_days, center=True, min_periods=1).mean()
        self.table_ = pd.Series(smooth.to_numpy()[pad:-pad], index=by_day.index)
        return self

    def predict(self, origin_dates: pd.DatetimeIndex, horizons: int = 7) -> np.ndarray:
        cols = []
        for h in range(1, horizons + 1):
            target_doy = (origin_dates + pd.Timedelta(days=h)).dayofyear
            cols.append(self.table_.reindex(target_doy).to_numpy())
        return np.column_stack(cols)


def fit_ridge(train: pd.DataFrame, val: pd.DataFrame, alphas: list[float]):
    """Linear model on the same features; alpha chosen on the validation set."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_cols, y_cols = feature_names(), target_names()
    best, best_rmse, best_alpha = None, np.inf, None
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(train[X_cols], train[y_cols])
        err = model.predict(val[X_cols]) - val[y_cols].to_numpy()
        rmse = float(np.sqrt((err ** 2).mean()))
        if rmse < best_rmse:
            best, best_rmse, best_alpha = model, rmse, alpha
    return best, best_alpha


class XGBMultiHorizon:
    """One gradient-boosted tree model per horizon, early-stopped on validation."""

    def __init__(self, params: dict, seed: int = 42):
        self.params = dict(params)
        self.seed = seed
        self.models_: list = []

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "XGBMultiHorizon":
        from xgboost import XGBRegressor

        X_cols = feature_names()
        self.models_ = []
        for y_col in target_names():
            model = XGBRegressor(objective="reg:squarederror", random_state=self.seed,
                                 n_jobs=4, **self.params)
            model.fit(train[X_cols], train[y_col],
                      eval_set=[(val[X_cols], val[y_col])], verbose=False)
            self.models_.append(model)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame[feature_names()]
        return np.column_stack([m.predict(X) for m in self.models_])
