"""One evaluation function for every model, so all numbers are comparable."""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model: str) -> pd.DataFrame:
    """Per-horizon RMSE, MAE, MBE and P95 absolute error, plus the macro average."""
    rows = []
    for h in range(y_true.shape[1]):
        err = y_pred[:, h] - y_true[:, h]
        rows.append({
            "model": model,
            "horizon": f"t+{h + 1}",
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "MBE": float(np.mean(err)),
            "P95": float(np.percentile(np.abs(err), 95)),
        })
    table = pd.DataFrame(rows)
    macro = table[["RMSE", "MAE", "MBE", "P95"]].mean().to_dict()
    macro.update({"model": model, "horizon": "macro"})
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)


def add_skill(results: pd.DataFrame, reference: str = "Climatology") -> pd.DataFrame:
    """Skill = 1 - RMSE_model / RMSE_reference. Positive means better than the reference."""
    ref = results[results["model"] == reference].set_index("horizon")["RMSE"]
    out = results.copy()
    out[f"skill_vs_{reference.lower()}"] = 1 - out["RMSE"] / out["horizon"].map(ref)
    return out


def summary_table(results: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    """Models as rows, horizons as columns."""
    wide = results.pivot(index="model", columns="horizon", values=metric)
    order = [c for c in wide.columns if c != "macro"] + ["macro"]
    return wide[order].sort_values("macro")


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     method: str, nominal: float) -> pd.DataFrame:
    """Coverage (share of true values inside the interval) and mean width, per horizon.

    A trustworthy interval has coverage close to its nominal level.
    """
    inside = (y_true >= lower) & (y_true <= upper)
    rows = [{"method": method, "horizon": f"t+{h + 1}", "nominal": nominal,
             "coverage": float(inside[:, h].mean()),
             "mean_width": float((upper[:, h] - lower[:, h]).mean())}
            for h in range(y_true.shape[1])]
    rows.append({"method": method, "horizon": "macro", "nominal": nominal,
                 "coverage": float(inside.mean()), "mean_width": float((upper - lower).mean())})
    return pd.DataFrame(rows)
