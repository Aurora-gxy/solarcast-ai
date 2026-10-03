"""SolarCast dashboard.  Run from the project root:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from solarcast.baselines import Climatology  # noqa: E402
from solarcast.config import load_config  # noqa: E402
from solarcast.data import load_daily  # noqa: E402
from solarcast.features import build_features, feature_names  # noqa: E402
from solarcast.predict import Forecaster  # noqa: E402

# One fixed colour per model, so a model keeps its colour on every chart.
MODEL_COLOURS = {
    "Climatology": "#2a78d6",
    "Attention MLP": "#eb6834",
    "Multi-output MLP": "#1baf7a",
    "XGBoost": "#eda100",
    "Ridge": "#e87ba4",
    "Persistence": "#008300",
}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
HORIZONS = [f"t+{h}" for h in range(1, 8)]

st.set_page_config(page_title="SolarCast AI", page_icon="☀️", layout="wide")


def style(fig: go.Figure, y_title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fcfcfb",
        font=dict(color=INK, size=13), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", tickfont=dict(color=MUTED))
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED))
    return fig


@st.cache_data
def get_data():
    cfg = load_config()
    daily = load_daily(cfg)
    f = cfg["features"]
    feats = build_features(daily, f["n_lags"], f["rolling_window"], f["horizons"], with_targets=False)
    return cfg, daily, feats


@st.cache_resource
def get_forecaster(path: str, mtime: float) -> Forecaster:
    return Forecaster.load(path)


@st.cache_data
def get_climatology(train_end: str, smooth: int) -> pd.Series:
    _, daily, _ = get_data()
    return Climatology(smooth).fit(daily.loc[:train_end, "solar_obs"]).table_


def read_report(name: str):
    path = ROOT / "reports" / name
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


cfg, daily, feats = get_data()
results = read_report("results.csv")
intervals = read_report("intervals.csv")
quality = read_report("data_quality.json")
summary = read_report("run_summary.json")

st.title("SolarCast AI")
st.caption("Seven-day solar radiation forecast with honest baselines and checked uncertainty")

tab_forecast, tab_models, tab_data = st.tabs(["Forecast", "Model comparison", "Data & reliability"])

# --------------------------------------------------------------------------- Forecast
with tab_forecast:
    model_files = {"Best on validation": "best.pt", "Multi-output MLP": "mlp.pt", "Attention MLP": "attention.pt"}
    available = {k: ROOT / "models" / v for k, v in model_files.items() if (ROOT / "models" / v).exists()}
    if not available:
        st.warning("No trained model found. Run `python -m solarcast.train` first.")
    else:
        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
        origin = pd.Timestamp(c1.date_input(
            "Forecast made on", value=feats.index.max().date(),
            min_value=feats.index.min().date(), max_value=feats.index.max().date(),
            help="Pick a past date to replay the forecast and compare it with what happened."))
        choice = c2.selectbox("Model", list(available))
        method_label = c3.radio("80% interval", ["Calibrated", "MC Dropout"], horizontal=True,
                                help="Calibrated intervals are sized from validation errors. "
                                     "MC Dropout intervals are shown for comparison and are too narrow.")

        usable = feats.loc[:origin]
        if usable.empty:
            st.error("Not enough history before this date.")
            st.stop()
        used_origin = usable.index[-1]
        if used_origin != origin:
            st.info(f"{origin.date()} has no complete 7-day input window (missing or low-quality data). "
                    f"Showing the forecast from {used_origin.date()} instead.")

        path = available[choice]
        fc = get_forecaster(str(path), path.stat().st_mtime)
        X = usable.iloc[[-1]][feature_names()].to_numpy()
        method = "calibrated" if method_label == "Calibrated" else "mc_dropout"
        mean, lower, upper = (a[0] for a in fc.predict_interval(X, method, cfg["uncertainty"]["mc_samples"]))
        dates = used_origin + pd.to_timedelta(np.arange(1, 8), unit="D")
        clim_table = get_climatology(cfg["split"]["train_end"], cfg["baselines"]["climatology_smooth_days"])
        clim = clim_table.reindex(dates.dayofyear).to_numpy()
        actual = daily["solar_obs"].reindex(dates).to_numpy()
        today = float(usable.iloc[-1]["solar_t"])

        # Plain-language headline first, chart second.
        change = (mean.mean() - today) / today * 100
        trend = "stay close to" if abs(change) < 10 else ("rise above" if change > 0 else "fall below")
        low_day = int(np.argmin(mean))
        vs_normal = (mean.mean() - clim.mean()) / clim.mean() * 100
        st.info(
            f"**Next 7 days:** radiation is expected to {trend} today's level "
            f"({today:.0f}), averaging **{mean.mean():.0f}**, which is {abs(vs_normal):.0f}% "
            f"{'above' if vs_normal >= 0 else 'below'} the seasonal normal. "
            f"Lowest expected day: **{dates[low_day].strftime('%a %d %b')}** ({mean[low_day]:.0f}). "
            f"Treat days 2–7 as a seasonal estimate: the model has little skill beyond one day."
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Today (observed input)", f"{today:.0f}")
        m2.metric("7-day mean forecast", f"{mean.mean():.0f}", f"{change:+.0f}% vs today")
        m3.metric("Lowest expected day", f"{dates[low_day].strftime('%a %d %b')} · {mean[low_day]:.0f}")
        m4.metric("Typical interval width", f"±{(upper - lower).mean() / 2:.0f}")

        hist = daily["solar_obs"].loc[used_origin - pd.Timedelta(days=20): used_origin]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=upper, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dates, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(42,120,214,0.18)", name="80% interval",
                                 hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Observed history",
                                 line=dict(color=MUTED, width=2), marker=dict(size=5),
                                 hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=dates, y=clim, mode="lines", name="Seasonal normal",
                                 line=dict(color=MUTED, width=2, dash="dot"), hovertemplate="%{y:.0f}"))
        fig.add_trace(go.Scatter(x=dates, y=mean, mode="lines+markers", name="Forecast",
                                 line=dict(color=BLUE, width=2), marker=dict(size=8),
                                 hovertemplate="%{y:.0f}"))
        if np.isfinite(actual).any():
            fig.add_trace(go.Scatter(x=dates, y=actual, mode="markers", name="What actually happened",
                                     marker=dict(color=ORANGE, size=9, line=dict(color="#fcfcfb", width=2)),
                                     hovertemplate="%{y:.0f}"))
        fig.add_vline(x=used_origin, line_color="#c3c2b7", line_width=1)
        st.plotly_chart(style(fig, "Daily solar radiation"), width="stretch")

        table = pd.DataFrame({"Date": dates.strftime("%a %d %b %Y"), "Horizon": HORIZONS,
                              "Forecast": mean, "Lower": lower, "Upper": upper,
                              "Seasonal normal": clim, "Actual": actual})
        with st.expander("Forecast table"):
            st.dataframe(table.round(1), hide_index=True, width="stretch")
        st.caption(f"Model: {fc.bundle['architecture']} (seed {fc.bundle['seed']}), "
                   f"trained {fc.bundle['trained_at'][:10]} on data up to {cfg['split']['train_end']}. "
                   "Not for trading or safety-critical decisions.")

# --------------------------------------------------------------------------- Models
with tab_models:
    if results is None:
        st.warning("No results found. Run `python -m solarcast.run_baselines` and `python -m solarcast.train`.")
    else:
        macro = results[results["horizon"] == "macro"].sort_values("RMSE").reset_index(drop=True)
        t1 = results[results["horizon"] == "t+1"].set_index("model")
        learned = macro[~macro["model"].isin(["Climatology", "Persistence"])]
        best_learned = learned.iloc[0]
        clim_macro = macro.set_index("model").loc["Climatology", "RMSE"]
        st.info(
            f"**No model clearly beats the seasonal-average baseline.** The best learned model "
            f"({best_learned['model']}) scores {best_learned['RMSE']:.2f} against {clim_macro:.2f} for climatology. "
            f"Learned models win by about {t1.loc[best_learned['model'], 'skill_vs_climatology'] * 100:.0f}% "
            f"on day 1 and lose slightly from day 2. Past radiation alone cannot anticipate cloud cover."
        )

        left, right = st.columns([1, 1.4])
        with left:
            st.subheader("Average error across 7 days")
            show = macro[macro["model"] != "Persistence"].iloc[::-1]
            # Dots, not bars: the axis is zoomed in, and bars must start at zero.
            fig = go.Figure(go.Scatter(
                x=show["RMSE"], y=show["model"], mode="markers+text",
                marker=dict(color=[MODEL_COLOURS[m] for m in show["model"]], size=14,
                            line=dict(color="#fcfcfb", width=2)),
                text=[f"{v:.2f}" for v in show["RMSE"]], textposition="middle right",
                textfont=dict(color=INK), hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
            fig = style(fig, "", 300)
            fig.update_layout(hovermode="closest", showlegend=False)
            lo = show["RMSE"].min()
            fig.update_xaxes(title="Macro RMSE (lower is better)", range=[lo - 0.15, show["RMSE"].max() + 0.15],
                             showgrid=True, gridcolor=GRID, title_font=dict(color=MUTED))
            fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
            st.plotly_chart(fig, width="stretch")
            pers = macro.set_index("model").loc["Persistence", "RMSE"]
            st.caption(f"Axis is zoomed in: the models differ by less than 1%. "
                       f"Persistence ({pers:.1f}) is far off this scale and is left out.")
        with right:
            st.subheader("Error by forecast day")
            with_pers = st.checkbox("Include Persistence", value=False)
            by_h = results[results["horizon"] != "macro"]
            fig = go.Figure()
            for model in macro["model"]:
                if model == "Persistence" and not with_pers:
                    continue
                d = by_h[by_h["model"] == model].set_index("horizon").reindex(HORIZONS)
                fig.add_trace(go.Scatter(
                    x=HORIZONS, y=d["RMSE"], mode="lines+markers", name=model,
                    line=dict(color=MODEL_COLOURS[model], width=2,
                              dash="dot" if model == "Climatology" else "solid"),
                    marker=dict(size=8), hovertemplate="%{y:.2f}"))
            st.plotly_chart(style(fig, "RMSE", 340), width="stretch")

        st.subheader("Full table")
        wide = results.pivot(index="model", columns="horizon", values="RMSE")[HORIZONS + ["macro"]]
        tbl = wide.join(macro.set_index("model")[["MAE", "MBE", "P95", "skill_vs_climatology", "macro_rmse_seed_std"]])
        tbl = tbl.sort_values("macro").rename(columns={
            "macro": "Macro RMSE", "skill_vs_climatology": "Skill vs climatology",
            "macro_rmse_seed_std": "Seed std"})
        fmt = {c: "{:.2f}" for c in tbl.columns}
        fmt["Skill vs climatology"] = "{:+.1%}"
        st.dataframe(tbl.style.format(fmt, na_rep="–"), width="stretch")
        with st.expander("How to read this"):
            st.markdown(
                "- **RMSE**: typical error size; large misses are penalised more.\n"
                "- **MBE**: average bias. Positive means the model over-predicts.\n"
                "- **P95**: 95% of daily errors are smaller than this.\n"
                "- **Skill vs climatology**: share of error removed relative to the seasonal average. "
                "Zero means no better than looking at the calendar.\n"
                "- **Seed std**: how much the score moves when only the random seed changes. "
                "Differences smaller than this are noise.")
        if summary:
            st.caption(f"Test period {summary['test_period'][0]} to {summary['test_period'][1]}, "
                       f"{summary['n_test']:,} forecasts. Train {summary['n_train']:,}, "
                       f"validation {summary['n_val']:,} samples. Neural models averaged over 5 seeds.")

# --------------------------------------------------------------------------- Data
with tab_data:
    if quality is None:
        st.warning("No data-quality report found. Run `python -m solarcast.run_baselines`.")
    else:
        st.info(f"**{quality['share_unusable']:.0%} of days are unusable** (flagged low quality). "
                f"They keep their place in the calendar as gaps instead of being deleted, "
                f"so 'yesterday' always means the previous calendar day.")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Period", f"{quality['start_date'][:4]}–{quality['end_date'][:4]}")
        q2.metric("Calendar days", f"{quality['calendar_days']:,}")
        q3.metric("Unusable days", f"{quality['days_unusable_total']:,}")
        q4.metric("Longest gap", f"{quality['longest_unusable_run_days']} days")

        st.subheader("Share of unusable days per year")
        by_year = pd.Series(quality["share_unusable_by_year"])
        fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values * 100,
                               marker=dict(color=BLUE, cornerradius=4),
                               hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
        fig = style(fig, "% of days", 280)
        fig.update_layout(showlegend=False, bargap=0.25)
        st.plotly_chart(fig, width="stretch")
        st.caption("2010–2013 are mostly unusable, so 2010 is excluded and the test set effectively starts in 2013.")

    if intervals is not None:
        st.subheader("Can the prediction interval be trusted?")
        mac = intervals[intervals["horizon"] == "macro"].copy()
        mac[["Model", "Method"]] = mac["method"].str.split(":", expand=True)
        mac["Model"] = mac["Model"].map({"mlp": "Multi-output MLP", "attention": "Attention MLP"})
        mac["Method"] = mac["Method"].map({"mc_dropout": "MC Dropout", "calibrated": "Calibrated"})
        nominal = float(mac["nominal"].iloc[0])
        worst = mac[mac["Method"] == "MC Dropout"]["coverage"].mean()
        st.markdown(
            f"An {nominal:.0%} interval should contain the true value on about {nominal:.0%} of days. "
            f"**MC Dropout intervals contain it on only {worst:.0%} of days**: they capture the model's own "
            f"uncertainty but not day-to-day weather noise. Calibrated intervals are sized from "
            f"validation-set errors and reach the target, at the cost of being much wider.")
        out = mac[["Model", "Method", "coverage", "mean_width"]].rename(
            columns={"coverage": "Coverage on test", "mean_width": "Mean width"})
        st.dataframe(out.style.format({"Coverage on test": "{:.0%}", "Mean width": "{:.0f}"}),
                     hide_index=True, width="stretch")
