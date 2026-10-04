# Predicting Ozone Nonattainment in US Counties

Predict daily ground-level ozone from weather and precursor pollutants (NO2, VOC), then roll predictions up to
**county nonattainment status and severity tier** under the EPA 0.070 ppm standard.
Extends the CS229 (2019) ozone-regression report and fixes its limitations (see "Improvements over the paper").

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python train.py --source synthetic --fast   # ~1 min smoke run, no credentials needed
python train.py --source synthetic          # full offline run (~6 min on a laptop)
python train.py --source bigquery           # REAL EPA data (see below)
python train.py --source csv --csv my.csv   # your own daily site data
python make_diagrams.py && python make_report.py    # architecture figures + 2-page PDF from outputs/metrics.json
python demo.py --list                       # county status table
python demo.py --state 01 --county 012 --temp-delta-f 4   # what-if: +4 F
pytest -q                                   # unit tests
```

### Real data (BigQuery)
1. Create/choose a Google Cloud project, enable BigQuery, run `gcloud auth application-default login`.
2. `python train.py --source bigquery` (public dataset `bigquery-public-data.epa_historical_air_quality`).
   Table names/parameter codes are in `config.py` and `src/data_bigquery.py`; confirm them in the BigQuery console if a query errors.
3. Re-run `python make_report.py` so the PDF reflects real-data numbers.

CSV schema: `site_id, state_code, county_code, latitude, longitude, date, ozone (ppm), temp (F), pressure, wind, rh, no2, voc`
(missing sensor values may be blank).

## Layout
| Path | Purpose |
|---|---|
| `config.py` | thresholds, split years, BigQuery table names |
| `src/data_bigquery.py`, `src/data_synthetic.py` | data loading (real / offline demo) |
| `src/features.py` | cleaning, leakage-safe features, temporal split |
| `src/models.py` | baselines, Ridge, SVR (tuned), MLP, HistGradientBoosting, classifier |
| `src/county.py` | 4th-high, 3-yr design value, tiers, expected exceedance days |
| `src/evaluate.py` | metrics and plots |
| `train.py` / `demo.py` | full pipeline / what-if demo |
| `make_diagrams.py`, `make_report.py` | architecture + flow figures, PDF write-up |
| `outputs/` | metrics.json, plots, county tables, saved model, write-up PDF |

## Improvements over the paper
| Paper limitation | Our fix |
|---|---|
| Train 1980-2016, validate Jan-Jun, test Jul-Dec (seasonal mismatch) | Whole-year split: train <=2014, val 2015, test 2016-17 |
| R2 = 1 - SSE/sum(y^2) (non-standard, inflated) | Standard sklearn R2 plus RMSE/MAE and baselines |
| SVR epsilon far too large -> R2 of -2.36 | Target scaled, C/epsilon grid-searched |
| Numeric site ID as a feature | Latitude/longitude |
| Inner joins shrink data (VOC/NO2 sparse) | LEFT JOIN + imputation / native NaN handling |
| Regression only, no regulatory output | County design value, nonattainment flag, severity tiers, exceedance classifier |
| No feature analysis | Permutation importance + no-lag ablation |

## Honest limitations
* Results in `outputs/` were produced on the real EPA BigQuery data (2000-2017). The offline synthetic generator is for demos/tests only.
* Target is the daily 1-hour max (available in the daily table), not the regulatory 8-hour average.
* No monitor-completeness rules, exceptional-event exclusions, or source-proximity features.
