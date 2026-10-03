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

from solarcast.config import load_config  # noqa: E402
from solarcast.data import load_daily  # noqa: E402
from solarcast.features import build_features, feature_names  # noqa: E402
from solarcast.weather import (WeatherRidge, clim_columns, fit_climatology,  # noqa: E402
                               nwp_columns, weather_features)

# PyTorch is imported only when a neural model is selected, so the dashboard
# can be deployed with the lightweight weather-aware model alone.

# One fixed colour per model, so a model keeps its colour on every chart.
MODEL_COLOURS = {
    "Climatology": "#2a78d6",
    "Attention MLP": "#eb6834",
    "Multi-output MLP": "#1baf7a",
    "XGBoost": "#eda100",
    "Ridge": "#e87ba4",
    "Persistence": "#008300",
    "NWP direct": "#4a3aa7",
    "Weather-aware Ridge": "#e34948",
}
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9"
HORIZONS = [f"t+{h}" for h in range(1, 8)]
DATASETS = {
    "Sydney (with weather forecasts)": ("configs/sydney.yaml", "data/open_meteo/observed_daily.csv"),
    "Course benchmark (history only)": ("configs/config.yaml", "data/raw"),
}
NN_FILES = {"Multi-output MLP": "mlp.pt", "Attention MLP": "attention.pt"}

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


def has_data(path: str) -> bool:
    p = ROOT / path
    return p.is_file() or (p.is_dir() and any(p.glob("*.csv")))


@st.cache_data
def get_data(config_path: str):
    """Daily table plus forecast-time features (no targets) for every usable origin day."""
    cfg = load_config(config_path)
    daily = load_daily(cfg)
    f = cfg["features"]
    feats = build_features(daily, f["n_lags"], f["rolling_window"], f["horizons"], with_targets=False)
    clim = fit_climatology(daily, cfg).predict(daily.index, f["horizons"])
    feats = feats.join(pd.DataFrame(clim, index=daily.index, columns=clim_columns(f["horizons"])))
    if "nwp" in cfg:
        feats = feats.join(weather_features(daily, cfg)[nwp_columns(f["horizons"])])
    return cfg, daily, feats


@st.cache_resource
def get_nn(path: str, mtime: float):
    from solarcast.predict import Forecaster
    return Forecaster.load(path)


@st.cache_resource
def get_weather_model(path: str, mtime: float) -> WeatherRidge:
    return WeatherRidge.load(path)


@st.cache_data(ttl=3600, show_spinner="Fetching the latest weather forecast...")
def get_live(config_path: str):
    """Inputs for a forecast starting today. Refreshed at most once an hour."""
    from solarcast.live import live_inputs
    cfg, daily, _ = get_data(config_path)
    return live_inputs(cfg["data"]["site_coords"], fit_climatology(daily, cfg), cfg["features"]["horizons"])


def read_report(reports_dir: Path, name: str):
    path = reports_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


# ----------------------------------------------------------------------------- setup
available_sets = {k: v[0] for k, v in DATASETS.items() if has_data(v[1])}
if not available_sets:
    st.error("No data found. Run `python -m solarcast.download` to fetch the Sydney dataset.")
    st.stop()

st.title("SolarCast AI")
st.caption("Seven-day solar radiation forecast with honest baselines and checked uncertainty")
dataset = st.sidebar.radio("Dataset", list(available_sets))
cfg, daily, feats = get_data(available_sets[dataset])
unit = cfg["data"].get("unit", "")
weather_on = "nwp" in cfg
reports_dir, models_dir = ROOT / cfg["reports_dir"], ROOT / cfg["models_dir"]
results = read_report(reports_dir, "results.csv")
if results is None:
    results = read_report(reports_dir, "results_baselines.csv")
quality = read_report(reports_dir, "data_quality.json")
summary = read_report(reports_dir, "run_summary.json")
interval_parts = [read_report(reports_dir, n) for n in ("intervals_weather.csv", "intervals.csv")]
interval_parts = [p for p in interval_parts if p is not None]
intervals = pd.concat(interval_parts, ignore_index=True) if interval_parts else None
st.sidebar.caption(
    f"**{cfg['data'].get('site', '')}** · daily radiation in {unit}\n\n"
    + ("Inputs: past radiation, calendar, and the weather forecast issued on the day."
       if weather_on else "Inputs: past radiation and calendar only."))
if weather_on:
    st.sidebar.caption("Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).")

tab_forecast, tab_models, tab_data = st.tabs(["Forecast", "Model comparison", "Data & reliability"])

# --------------------------------------------------------------------------- Forecast
with tab_forecast:
    choices = {}
    if weather_on and (models_dir / "weather_ridge.json").exists():
        choices["Weather-aware Ridge"] = models_dir / "weather_ridge.json"
    choices.update({k: models_dir / v for k, v in NN_FILES.items() if (models_dir / v).exists()})
    if not choices:
        st.warning("No trained model found. Run `python -m solarcast.run_baselines` and "
                   "`python -m solarcast.train` with this dataset's config.")
    else:
        live_possible = "Weather-aware Ridge" in choices and "site_coords" in cfg["data"]
        mode = "Replay a past date"
        if live_possible:
            mode = st.radio("Mode", ["Live: today onwards", "Replay a past date"], horizontal=True,
                            help="Live uses the weather forecast published today. Replay shows what the "
                                 "model would have said on a past date, next to what actually happened.")
        live_row = recent = None
        if mode.startswith("Live"):
            try:
                live_row, recent = get_live(available_sets[dataset])
            except Exception as exc:   # network or API problem: fall back instead of failing
                st.warning(f"Live weather data is unavailable right now ({exc}). Showing replay mode instead.")
                mode = "Replay a past date"
        is_live = live_row is not None

        c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
        choice = c2.selectbox("Model", ["Weather-aware Ridge"] if is_live else list(choices), disabled=is_live)
        is_weather = choice == "Weather-aware Ridge"
        if is_live:
            row = live_row
            used_origin = row.index[0]
            c1.markdown(f"**Forecast for**  \n{(used_origin + pd.Timedelta(days=1)).strftime('%a %d %b %Y')} "
                        f"and the following 6 days")
        else:
            needed = feature_names() + clim_columns() + (nwp_columns() if is_weather else [])
            usable_all = feats.dropna(subset=needed)
            origin = pd.Timestamp(c1.date_input(
                "Forecast made on", value=usable_all.index.max().date(),
                min_value=usable_all.index.min().date(), max_value=usable_all.index.max().date(),
                help="Pick a past date to replay the forecast and compare it with what happened."))
            usable = usable_all.loc[:origin]
            if usable.empty:
                st.error("Not enough history before this date.")
                st.stop()
            row = usable.iloc[[-1]]
            used_origin = row.index[0]
            if used_origin != origin:
                st.info(f"{origin.date()} has incomplete inputs. Showing the forecast from {used_origin.date()}.")
        if is_weather:
            method_label = "Calibrated"
            c3.markdown("**80% interval**  \nSized from validation errors, scaled by season")
        else:
            method_label = c3.radio("80% interval", ["Calibrated", "MC Dropout"], horizontal=True,
                                    help="Calibrated intervals are sized from validation errors. "
                                         "MC Dropout intervals are shown for comparison and are too narrow.")

        path = choices[choice]
        if is_weather:
            model = get_weather_model(str(path), path.stat().st_mtime)
            mean, lower, upper = (a[0] for a in model.predict_interval(row))
            trained_note = (f"Weather-aware Ridge, trained {model.meta.get('trained_at', '')[:10]} on "
                            f"{model.meta.get('n_train', '?')} days of archived forecasts.")
        else:
            model = get_nn(str(path), path.stat().st_mtime)
            method = "calibrated" if method_label == "Calibrated" else "mc_dropout"
            mean, lower, upper = (a[0] for a in model.predict_interval(
                row[feature_names()].to_numpy(), method, cfg["uncertainty"]["mc_samples"]))
            trained_note = (f"{choice} (seed {model.bundle['seed']}), trained "
                            f"{model.bundle['trained_at'][:10]}. Inputs: past radiation only.")

        dates = used_origin + pd.to_timedelta(np.arange(1, 8), unit="D")
        clim = row[clim_columns()].to_numpy()[0]
        actual = np.full(7, np.nan) if is_live else daily["solar_obs"].reindex(dates).to_numpy()
        today = float(row["solar_t"].iloc[0])

        # Plain-language headline first, chart second.
        change = (mean.mean() - today) / today * 100
        trend = "stay close to" if abs(change) < 10 else ("rise above" if change > 0 else "fall below")
        low_day = int(np.argmin(mean))
        vs_normal = (mean.mean() - clim.mean()) / clim.mean() * 100
        skill = None
        if results is not None and "skill_vs_climatology" in results:
            sk = results[(results["model"] == choice) & (results["horizon"] != "macro")]
            skill = sk.set_index("horizon")["skill_vs_climatology"].reindex(HORIZONS)
        if skill is not None and skill.notna().all() and (skill > 0.10).any():
            last_good = int(np.where(skill.to_numpy() > 0.10)[0].max()) + 1
            reliability = (f"On past data this model beat the seasonal normal by {skill.iloc[0]:.0%} "
                           f"on day 1 and {skill.iloc[-1]:.0%} on day 7"
                           + ("." if last_good == 7 else f"; treat days after day {last_good} as a seasonal estimate."))
        else:
            reliability = "Treat days 2–7 as a seasonal estimate: this model has little skill beyond one day."
        st.info(
            f"**{'Today and the next 6 days' if is_live else 'Next 7 days'}:** radiation is expected to {trend} "
            f"{'yesterday' if is_live else 'today'}'s level ({today:.1f} {unit}), "
            f"averaging **{mean.mean():.1f}**, which is {abs(vs_normal):.0f}% "
            f"{'above' if vs_normal >= 0 else 'below'} the seasonal normal. "
            f"Lowest expected day: **{dates[low_day].strftime('%a %d %b')}** ({mean[low_day]:.1f}). {reliability}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Yesterday (model analysis)" if is_live else "Today (observed input)", f"{today:.1f}")
        m2.metric("7-day mean forecast", f"{mean.mean():.1f}",
                  f"{change:+.0f}% vs {'yesterday' if is_live else 'today'}")
        m3.metric(f"Lowest expected day ({mean[low_day]:.1f})", dates[low_day].strftime("%a %d %b"))
        m4.metric("Interval half-width, day 1 → day 7",
                  f"±{(upper[0] - lower[0]) / 2:.1f} → ±{(upper[-1] - lower[-1]) / 2:.1f}")

        hist = recent if is_live else daily["solar_obs"].loc[used_origin - pd.Timedelta(days=20): used_origin]
        fmt = "%{y:.1f}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=upper, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dates, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(42,120,214,0.18)", name="80% interval", hovertemplate=fmt))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Recent days (weather-model analysis)" if is_live else "Observed history",
                                 line=dict(color=MUTED, width=2), marker=dict(size=5), hovertemplate=fmt))
        fig.add_trace(go.Scatter(x=dates, y=clim, mode="lines", name="Seasonal normal",
                                 line=dict(color=MUTED, width=2, dash="dot"), hovertemplate=fmt))
        fig.add_trace(go.Scatter(x=dates, y=mean, mode="lines+markers", name="Forecast",
                                 line=dict(color=BLUE, width=2), marker=dict(size=8), hovertemplate=fmt))
        if np.isfinite(actual).any():
            fig.add_trace(go.Scatter(x=dates, y=actual, mode="markers", name="What actually happened",
                                     marker=dict(color=ORANGE, size=9, line=dict(color="#fcfcfb", width=2)),
                                     hovertemplate=fmt))
        fig.add_vline(x=used_origin, line_color="#c3c2b7", line_width=1)
        st.plotly_chart(style(fig, f"Daily solar radiation ({unit})"), width="stretch")

        table = pd.DataFrame({"Date": dates.strftime("%a %d %b %Y"), "Horizon": HORIZONS,
                              "Forecast": mean, "Lower": lower, "Upper": upper, "Seasonal normal": clim})
        if is_weather:
            table["Raw weather-model forecast"] = row[[f"nwp_rad_h{h}" for h in range(1, 8)]].to_numpy()[0]
            table["Forecast cloud cover (%)"] = row[[f"nwp_cloud_h{h}" for h in range(1, 8)]].to_numpy()[0]
        table["Actual"] = actual
        with st.expander("Forecast table"):
            st.dataframe(table.round(1), hide_index=True, width="stretch")
        if is_live:
            st.caption("Live mode uses today's weather forecast from Open-Meteo and the weather model's own "
                       "analysis for the past week, because reanalysis data arrives about five days late. "
                       "Refreshed hourly.")
        st.caption(f"{trained_note} Not for trading or safety-critical decisions.")

# --------------------------------------------------------------------------- Models
with tab_models:
    if results is None:
        st.warning("No results found. Run `python -m solarcast.run_baselines` and `python -m solarcast.train`.")
    else:
        macro = results[results["horizon"] == "macro"].sort_values("RMSE").reset_index(drop=True)
        by_model = macro.set_index("model")
        clim_macro = by_model.loc["Climatology", "RMSE"]
        best = macro.iloc[0] if macro.iloc[0]["model"] != "Climatology" else macro.iloc[1]
        t1 = results[results["horizon"] == "t+1"].set_index("model")["skill_vs_climatology"]
        history_only = [m for m in ("Ridge", "XGBoost", "Multi-output MLP", "Attention MLP") if m in by_model.index]
        if best["skill_vs_climatology"] > 0.05:
            hist_range = by_model.loc[history_only, "skill_vs_climatology"]
            st.info(
                f"**{best['model']} cuts forecast error by {best['skill_vs_climatology']:.0%} versus the "
                f"seasonal-average baseline** ({best['RMSE']:.2f} vs {clim_macro:.2f} {unit}), and by "
                f"{t1[best['model']]:.0%} on day 1. Models that only see past radiation stay within "
                f"{hist_range.abs().max():.0%} of that baseline, whatever their architecture: "
                f"the gain comes from the weather-forecast inputs, not from model complexity.")
        else:
            st.info(
                f"**No model clearly beats the seasonal-average baseline.** The best learned model "
                f"({best['model']}) scores {best['RMSE']:.2f} against {clim_macro:.2f} for climatology. "
                f"Learned models win by about {t1[best['model']]:.0%} on day 1 and lose slightly from day 2. "
                f"Past radiation alone cannot anticipate cloud cover.")

        left, right = st.columns([1, 1.4])
        pers = by_model.loc["Persistence", "RMSE"]
        with left:
            st.subheader("Average error across 7 days")
            show = macro[macro["model"] != "Persistence"].iloc[::-1]
            # Dots, not bars: the axis does not start at zero.
            fig = go.Figure(go.Scatter(
                x=show["RMSE"], y=show["model"], mode="markers+text",
                marker=dict(color=[MODEL_COLOURS[m] for m in show["model"]], size=14,
                            line=dict(color="#fcfcfb", width=2)),
                text=[f"{v:.2f}" for v in show["RMSE"]], textposition="middle right",
                textfont=dict(color=INK), hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
            fig = style(fig, "", 60 + 46 * len(show))
            fig.update_layout(hovermode="closest", showlegend=False)
            span = show["RMSE"].max() - show["RMSE"].min()
            fig.update_xaxes(title=f"Macro RMSE in {unit} (lower is better)",
                             range=[show["RMSE"].min() - 0.12 * span, show["RMSE"].max() + 0.22 * span],
                             showgrid=True, gridcolor=GRID, title_font=dict(color=MUTED))
            fig.update_yaxes(showgrid=False, tickfont=dict(color=INK))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"The axis does not start at zero. Persistence ({pers:.2f}) is off this scale and is left out.")
        with right:
            st.subheader("Error by forecast day")
            with_pers = st.checkbox("Include Persistence", value=False)
            by_h = results[results["horizon"] != "macro"]
            fig = go.Figure()
            for name in macro["model"]:
                if name == "Persistence" and not with_pers:
                    continue
                d = by_h[by_h["model"] == name].set_index("horizon").reindex(HORIZONS)
                fig.add_trace(go.Scatter(
                    x=HORIZONS, y=d["RMSE"], mode="lines+markers", name=name,
                    line=dict(color=MODEL_COLOURS[name], width=2,
                              dash="dot" if name == "Climatology" else "solid"),
                    marker=dict(size=8), hovertemplate="%{y:.2f}"))
            st.plotly_chart(style(fig, f"RMSE ({unit})", 360), width="stretch")

        st.subheader("Full table")
        wide = results.pivot(index="model", columns="horizon", values="RMSE")[HORIZONS + ["macro"]]
        extra = [c for c in ("MAE", "MBE", "P95", "skill_vs_climatology", "macro_rmse_seed_std") if c in by_model]
        tbl = wide.join(by_model[extra]).sort_values("macro").rename(columns={
            "macro": "Macro RMSE", "skill_vs_climatology": "Skill vs climatology",
            "macro_rmse_seed_std": "Seed std"})
        fmt_map = {c: "{:.2f}" for c in tbl.columns}
        fmt_map["Skill vs climatology"] = "{:+.1%}"
        st.dataframe(tbl.style.format(fmt_map, na_rep="–"), width="stretch")
        with st.expander("How to read this"):
            st.markdown(
                "- **RMSE**: typical error size; large misses are penalised more.\n"
                "- **MBE**: average bias. Positive means the model over-predicts.\n"
                "- **P95**: 95% of daily errors are smaller than this.\n"
                "- **Skill vs climatology**: share of error removed relative to the seasonal average. "
                "Zero means no better than looking at the calendar.\n"
                "- **Seed std**: how much the score moves when only the random seed changes. "
                "Differences smaller than this are noise.\n"
                "- **NWP direct**: the weather model's own radiation forecast, used as is.\n"
                "- **Weather-aware Ridge**: a small linear model that blends the weather forecast "
                "with the seasonal normal, one per forecast day.")
        if summary:
            st.caption(f"Test period {summary['test_period'][0]} to {summary['test_period'][1]}, "
                       f"{summary['n_test']:,} forecasts, identical for every model. "
                       f"Neural models are averaged over 5 seeds.")

# --------------------------------------------------------------------------- Data
with tab_data:
    if quality is not None:
        if quality["days_unusable_total"] > 0:
            st.info(f"**{quality['share_unusable']:.0%} of days are unusable** (flagged low quality). "
                    f"They keep their place in the calendar as gaps instead of being deleted, "
                    f"so 'yesterday' always means the previous calendar day.")
        else:
            st.info("**Every calendar day has a usable value.** The target is ERA5 reanalysis radiation "
                    "from Open-Meteo: a gridded estimate built from satellite and ground observations, "
                    "not a reading from a single ground instrument.")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Period", f"{quality['start_date'][:4]}–{quality['end_date'][:4]}")
        q2.metric("Calendar days", f"{quality['calendar_days']:,}")
        q3.metric("Unusable days", f"{quality['days_unusable_total']:,}")
        q4.metric("Longest gap", f"{quality['longest_unusable_run_days']} days")

        by_year = pd.Series(quality["share_unusable_by_year"])
        if by_year.max() > 0:
            st.subheader("Share of unusable days per year")
            fig = go.Figure(go.Bar(x=by_year.index, y=by_year.values * 100,
                                   marker=dict(color=BLUE, cornerradius=4),
                                   hovertemplate="%{x}: %{y:.0f}%<extra></extra>"))
            fig = style(fig, "% of days", 280)
            fig.update_layout(showlegend=False, bargap=0.25)
            st.plotly_chart(fig, width="stretch")

    if weather_on:
        nwp_ok = feats[nwp_columns()].notna().all(axis=1)
        first = nwp_ok[nwp_ok].index.min()
        share = nwp_ok.loc[first:].mean()
        st.subheader("Weather-forecast inputs")
        st.markdown(
            f"Archived forecasts (Open-Meteo Previous Runs API, model `{cfg['nwp']['model']}`) start on "
            f"**{first.date()}** and cover **{share:.0%}** of days since then. For each day the model uses "
            f"the forecast issued that day for 1 to 7 days ahead, never the weather that later happened. "
            f"Because this archive is short, weather-aware models train on about 14 months of data.")

    if intervals is not None:
        st.subheader("Can the prediction interval be trusted?")
        mac = intervals[intervals["horizon"] == "macro"].copy()
        mac[["Model", "Method"]] = mac["method"].str.split(":", expand=True)
        mac["Model"] = mac["Model"].map({"mlp": "Multi-output MLP", "attention": "Attention MLP",
                                         "weather_ridge": "Weather-aware Ridge"})
        mac["Method"] = mac["Method"].map({"mc_dropout": "MC Dropout", "calibrated": "Calibrated"})
        nominal = float(mac["nominal"].iloc[0])
        text = (f"An {nominal:.0%} interval should contain the true value on about {nominal:.0%} of days. ")
        mc = mac[mac["Method"] == "MC Dropout"]
        if len(mc):
            text += (f"**MC Dropout intervals contain it on only {mc['coverage'].mean():.0%} of days**: they "
                     f"capture the model's own uncertainty but not day-to-day weather noise. ")
        text += "Calibrated intervals are sized from validation-set errors."
        st.markdown(text)
        out = mac[["Model", "Method", "coverage", "mean_width"]].rename(
            columns={"coverage": "Coverage on test", "mean_width": f"Mean width ({unit})"})
        st.dataframe(out.style.format({"Coverage on test": "{:.0%}", f"Mean width ({unit})": "{:.1f}"}),
                     hide_index=True, width="stretch")
        if weather_on:
            st.caption("The neural models' calibrated intervals under-cover here because the validation "
                       "period (April to September) is mostly winter, when errors are smaller. The "
                       "weather-aware model scales its interval with the seasonal normal to correct for this.")
