# TrafficAnomalyDetection

Anomaly detection on hourly vehicle counts from Bologna's inductive loop detectors (spire), covering September 2023 – September 2025. The project combines traffic flow data with weather observations and event calendars to detect three families of structural anomalies.

## Problem

Standard traffic monitoring flags instantaneous outliers (a single noisy reading). This project targets a harder class of anomalies: **persistent, structural deviations** from expected behaviour — changes in daily profile shape, long-term trend shifts, and abnormal responses to recurring events.

## Data

| File | Description |
|---|---|
| `data/raw/spire_*.parquet` | Hourly vehicle counts per sensor and direction |
| `data/raw/meteo_*.parquet` | Hourly weather (temperature, precipitation, wind) from Open-Meteo |
| `data/raw/accuratezza_*.parquet` | Per-sensor data quality scores |
| `data/processed/dataset_finale_*.parquet` | Merged and enriched dataset ready for modelling |

Data is sourced from the [Bologna Open Data portal](https://opendata.comune.bologna.it) and the Open-Meteo API. Download and merge logic is in `notebooks/00_merge_dataset.ipynb`.

### Processed dataset

**345 232 rows × 33 columns** — 19 sensors, September 2023 – September 2025.

| Column | Type | Description |
|---|---|---|
| `data` | datetime | Timestamp (hourly) |
| `chiave` | int | Sensor ID |
| `id_uni` | int | Detector station ID (can group multiple `chiave` directions) |
| `nome_via` | str | Street name |
| `direzione` | str | Traffic direction |
| `longitudine` / `latitudine` | float | Sensor coordinates |
| `conteggio_veicoli` | int | Hourly vehicle count (target) |
| `ora` | int | Hour of day (0–23), raw column parsed from the source data |
| `ora_del_giorno` | int | Hour of day (0–23) |
| `timestamp` | datetime | Combined date + hour timestamp |
| `fascia_oraria` | str | Time slot (morning / afternoon / …) |
| `giorno_settimana` | str | Day of week |
| `tipo_giorno` | str | Day type: feriale / weekend / festivo |
| `weekend` / `festivo_nazionale` / `festa_locale` | bool | Calendar flags |
| `nome_festivita` | str | Holiday name (if applicable) |
| `temperature_2m` | float | Air temperature (°C) |
| `precipitation` | float | Precipitation (mm/h) |
| `rain` | float | Rain amount (mm/h), raw Open-Meteo field |
| `wind_speed_10m` | float | Wind speed (km/h) |
| `weather_code` | int | Raw WMO weather code |
| `pioggia` | bool | Rain flag |
| `tempo` | str | Weather condition label (from `weather_code`) |
| `accuratezza` | float | Sensor data quality score (%) |
| `evento_traffico` | str | Event name (if applicable) |
| `tipo_evento` | str | Event type (concert / sport / fair / …) |
| `impatto_evento` | str | Expected traffic impact level |
| `score_evento` | int | Numeric impact score derived from `impatto_evento` |
| `eventi_sovrapposti` | int | Number of simultaneous events |
| `data_giorno` | date | Calendar date (date part of `timestamp`) |
| `data_str` | str | Date string (used for event filtering) |

## Anomaly types

### 1 — Profile shape anomaly (`detect_profile_shape_anomaly`)
Detects sensors whose daily hourly profile has structurally changed compared to a reference period. Uses cosine similarity between the normalised hourly distribution of the current month and the mean of the first N reference months. A drop in peak-hour share is also penalised.

### 2 — Trend anomaly (`detect_trend_anomaly`)
Detects persistent deviations from the seasonal baseline. For each sensor and day type (weekday / weekend / holiday), it computes a z-score against the same calendar month in the reference period. A sensor is flagged when the deviation is statistically significant for at least `min_consecutive_months` consecutive months.

### 3 — Event response anomaly (`detect_event_response_anomaly`)
Detects sensors that respond anomalously to recurring events (concerts, fairs, sporting events). For each event family and sensor, it computes the impact ratio (traffic during the event vs same-month baseline) and flags editions whose impact is a statistical outlier relative to previous editions of the same event.

## Project structure

```
TrafficAnomalyDetection/
├── data/
│   ├── raw/                        # Source parquet files
│   └── processed/                  # Merged dataset
├── docs/
│   └── problem_framing.md
├── notebooks/
│   ├── 00_merge_dataset.ipynb      # Data download and merging
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_baselines.ipynb          # Baseline models (exploration)
│   ├── 03_final.ipynb              # Final pipeline (uses src modules)
│   └── 04_evaluation.ipynb         # Evaluation
├── src/
│   ├── anomaly_detection.py        # Three anomaly detectors
│   └── baselines.py                # Feature engineering, ML baselines
├── requirements.txt
└── README.md
```


## Usage

Run the notebooks in order:

```
00_merge_dataset.ipynb   → downloads and merges raw data
01_eda.ipynb             → exploratory analysis
02_baselines.ipynb       → baseline model exploration
03_final.ipynb           → full pipeline with anomaly detection + ML baselines
```

Or import the detectors directly:

```python
import pandas as pd
from src.anomaly_detection import (
    detect_profile_shape_anomaly,
    detect_trend_anomaly,
    detect_event_response_anomaly,
)

df = pd.read_parquet("data/processed/dataset_finale_2023-09-01_2025-09-30.parquet")

r1 = detect_profile_shape_anomaly(df)
r2 = detect_trend_anomaly(df)
r3 = detect_event_response_anomaly(df)
```

## ML baselines

`src/baselines.py` provides three additional anomaly detection approaches for comparison:

* **Isolation Forest** — unsupervised, partition-based outlier detection
* **Local Outlier Factor** — density-based, sensitive to local structure
* **LightGBM regressor** — forecasting-based: residuals between predicted and actual counts are used as anomaly scores; hyperparameters are tuned via `lgb_grid_search`

Feature engineering (`feat_engineering`) adds temporal lags, rolling means, and encodes categorical columns. Lags and rolling windows are computed per sensor (`groupby("chiave")`) to avoid data leakage across sensors.

## Predictive analysis and results

`notebooks/04_evaluation.ipynb` goes beyond the three retrospective detectors above and explores forecasting-based early warning.

**Methodology**
* **Interval forecasting**: three LightGBM quantile regressors (5th, 50th, 95th percentile) trained on lag, rolling-mean, and weather features; an hourly count is flagged as anomalous when it falls outside the [q05, q95] band.
* **Evaluation**: point-forecast accuracy (RMSE, MAPE, R²) on the median model, plus precision/recall and average lead time obtained by aggregating the hourly flags to a monthly anomaly call and comparing against the retrospective trend detector.
* Two further approaches are sketched for future work: an early-warning classifier on monthly trend features, and an event-impact regressor predicting the impact ratio of recurring events before they happen.

**Results**
* The median model explains traffic volume well (R² ≈ 0.94), but residuals show a systematic rush-hour bias (under-prediction at 7–9 and 16–18, over-prediction at night).
* Feature importance is dominated by short-term lags and rolling means; weather features contribute comparatively little.
* Aggregated monthly and compared against the retrospective detector, the method is conservative: precision ≈ 0.85 but recall ≈ 0.28, with an average lead time of ~27 days ahead of the retrospective heuristic.
* Conclusion: quantile-interval forecasting is a useful complement to the retrospective detectors for genuine early warning, but recall is currently too low to replace them — see the notebook's conclusions section for details and next steps.

