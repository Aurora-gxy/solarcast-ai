# Data

## `open_meteo/` (committed)

Public data for Sydney (33.87°S, 151.21°E), downloaded with `python -m solarcast.download`.

| File | Content |
|---|---|
| `observed_daily.csv` | Daily shortwave radiation sum (MJ/m²) and basic weather, 2000 onwards. ERA5 reanalysis via the Open-Meteo Historical Weather API. |
| `nwp_<model>_daily.csv` | Archived weather-model forecasts of radiation and cloud cover at lead times of 1 to 7 days, from January 2024. Open-Meteo Previous Runs API. One row per valid day. |
| `download_report.json` | Download date and coverage summary. |

Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## `raw/` (not committed)

Yearly files `daily_solar_YYYY.csv` with columns `date, solar_t, bad_quality`, used by
`configs/config.yaml`. This dataset is not redistributed.
