# SolarCast AI

Seven-day solar radiation forecasting for Sydney, with honest baselines and checked uncertainty.

> Status: work in progress. Data pipeline, baselines, neural models, weather-aware
> model, interval calibration and a local dashboard are done. Live "next 7 days"
> forecasts and online deployment are not yet included.

## Key result

On one full year of unseen Sydney data (Oct 2025 – Sep 2026, 353 forecasts):

- Models that only see past radiation (Ridge, XGBoost, MLP, Attention MLP) stay within
  about 1% of a seasonal-average baseline, whatever their architecture.
- Adding the weather forecast that was available on the day cuts 7-day error by **35%**
  versus that baseline, and by **55%** for the next day.

The gain comes from better inputs, not from a more complex model.

| Model | Inputs | t+1 | t+3 | t+7 | Macro RMSE | Skill vs climatology |
|---|---|---|---|---|---|---|
| Weather-aware Ridge | history + weather forecast | 2.06 | 2.65 | 3.88 | **2.97** | **+35.4%** |
| NWP direct | weather forecast only | 1.89 | 2.70 | 3.91 | 3.10 | +32.5% |
| Attention MLP | history | 4.29 | 4.57 | 4.68 | 4.54 ± 0.02 | +1.1% |
| XGBoost | history | 4.25 | 4.62 | 4.72 | 4.58 | +0.3% |
| Climatology | calendar | 4.54 | 4.60 | 4.65 | 4.59 | 0.0% |
| Multi-output MLP | history | 4.31 | 4.64 | 4.73 | 4.61 ± 0.04 | −0.3% |
| Ridge | history | 4.32 | 4.67 | 4.72 | 4.61 | −0.4% |
| Persistence | today's value | 5.26 | 6.62 | 6.61 | 6.23 | −35.6% |

RMSE in MJ/m² per day. Every model is scored on the same test days. Neural models are
averaged over 5 random seeds. Regenerate with the commands below; do not edit by hand.

**Reading the table.** Most of the gain is already in the raw weather-model forecast
(NWP direct). The learned correction adds about 3 points of skill, mainly at days 6–7,
where it leans back towards the seasonal normal.

## Quick start

```bash
pip install -e ".[all]"
python -m solarcast.download                                   # public Sydney data (Open-Meteo)
python -m solarcast.run_baselines --config configs/sydney.yaml # baselines + weather-aware model
python -m solarcast.train --config configs/sydney.yaml         # neural models, 5 seeds each
streamlit run app/streamlit_app.py                             # dashboard at http://localhost:8501
pytest -q
```

All settings live in `configs/`. Outputs are written to `reports/` and `models/`.

## How leakage is avoided

- Bad or missing days keep their calendar slot as gaps, so "yesterday" is always the previous day.
- Splits are in time order; scaling and the seasonal normal use the training period only.
- Weather inputs are **archived forecasts**: for a forecast made on day t for day t+h, the
  model sees what the weather model predicted h days ahead, never the weather that later happened.
- Model choices (Ridge strength, early stopping, which seed to save) use the validation set only.
- Unit tests cover each of these (`tests/`).

## Uncertainty

| 80% interval | Coverage on test | Mean width (MJ/m²) |
|---|---|---|
| Weather-aware Ridge, calibrated and scaled by season | 86% | 7.8 |
| MLP, calibrated (fixed width) | 63% | 7.3 |
| MLP, MC Dropout | 18% | 1.8 |

MC Dropout reflects model uncertainty only and is far too narrow. A fixed-width interval
calibrated on a winter-heavy validation period under-covers in summer, so the weather-aware
model stores its interval as a share of the seasonal normal.

## Data

- **Sydney** (`configs/sydney.yaml`): daily shortwave radiation 2000–2026 from the Open-Meteo
  Historical Weather API (ERA5 reanalysis), and archived forecasts from the Open-Meteo
  Previous Runs API (from January 2024). Weather data by [Open-Meteo.com](https://open-meteo.com/), CC BY 4.0.
- **Course benchmark** (`configs/config.yaml`): 24 years of daily station data with 13%
  low-quality days, history-only inputs. Not redistributed here. On this dataset no model
  beats climatology (macro RMSE 58.17 for both the best neural model and climatology).

## Known limitations

- The target is reanalysis radiation, a gridded estimate, not a ground instrument reading.
- The forecast archive starts in 2024, so weather-aware models train on about 14 months
  (431 days) and are tested on a single year.
- Skill drops after day 5; days 6–7 are only about 17% better than the seasonal normal.
- Train, validation and test periods cover different seasons, which affects interval calibration.
