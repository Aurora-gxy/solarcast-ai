# SolarCast AI

Seven-day solar radiation forecasting with honest baselines and checked uncertainty.

> Status: work in progress. Data pipeline, baselines, neural models, interval
> calibration and a local dashboard are done. Online deployment is not yet included.

## Quick start

```bash
pip install -e ".[all]"
python -m solarcast.run_baselines      # data checks + four baselines
python -m solarcast.train              # neural models, 5 seeds each (~1 min on CPU)
python -m solarcast.predict            # 7-day forecast with an 80% interval
streamlit run app/streamlit_app.py     # dashboard at http://localhost:8501
pytest -q
```

Place the yearly files (`daily_solar_YYYY.csv`) in `data/raw/` first.
All settings live in `configs/config.yaml`; outputs are written to `reports/`.

## Results

Test period 2013–2019, 1,600 forecast origins. Train 1996–2006, validation 2007–2009.
Neural models are averaged over 5 random seeds. RMSE in the units of the source data.

| Model | t+1 | t+3 | t+7 | Macro RMSE | Skill vs climatology |
|---|---|---|---|---|---|
| Climatology | 58.86 | 58.14 | 58.35 | 58.17 | 0.0% |
| Attention MLP | 56.87 | 58.56 | 58.62 | 58.17 ± 0.12 | 0.0% |
| Multi-output MLP | 56.83 | 58.68 | 58.86 | 58.33 ± 0.07 | −0.3% |
| XGBoost | 57.15 | 58.68 | 58.93 | 58.38 | −0.4% |
| Ridge | 56.93 | 59.02 | 58.73 | 58.49 | −0.6% |
| Persistence | 71.17 | 84.13 | 82.69 | 81.49 | −40.1% |

Regenerate with the commands above; do not edit by hand. Full tables: `reports/results.csv`.

**Reading the table.** Every learned model beats climatology by about 3% at t+1
and loses slightly from t+2 onwards. With past radiation as the only input,
the models learn the seasonal cycle and one day of persistence, and nothing more.

## Uncertainty

| 80% interval | Coverage on test | Mean width |
|---|---|---|
| MC Dropout (100 samples) | 20% | 26 |
| Calibrated on validation errors | 80% | 136 |

MC Dropout intervals are far too narrow: they reflect model uncertainty only,
not day-to-day weather noise. The calibrated interval uses the error size
observed on the validation set for each horizon.

## Known limitations

- History-only inputs cannot anticipate cloud cover, so no model beats
  climatology beyond one day. Weather-forecast features are the next step.
- 2010–2013 have 40–82% unusable days. 2010 is excluded and the test set
  effectively starts in 2013.
- Calibrated intervals have the same width for every day at a given horizon.
