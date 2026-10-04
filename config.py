"""Central configuration for the ozone nonattainment project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DOCS_DIR = ROOT / "docs"
CLEAN_CSV = DATA_DIR / "clean_daily.csv.gz"

SEED = 42

# ---- Regulatory threshold -------------------------------------------------
# EPA 2015 ozone NAAQS: 0.070 ppm. A county is in nonattainment when its
# design value (3-yr average of the annual 4th-highest daily maximum, highest
# monitor in the county) exceeds this value.
NAAQS_PPM = 0.070
# Simplified severity tiers (upper bounds in ppm) used for the classifier extension.
TIER_BINS = [0.0, 0.070, 0.081, 0.093, 0.105, 0.111, 10.0]
TIER_NAMES = ["Attainment", "Marginal", "Moderate", "Serious", "Severe", "Extreme"]

# ---- Temporal split (fixes the paper's Jan-Jun / Jul-Dec seasonal split) ----
TRAIN_END_YEAR = 2014      # train: <= 2014
VAL_YEAR = 2015            # validation: full calendar year 2015
TEST_YEARS = (2016, 2017)  # test: two full, unseen calendar years

# ---- BigQuery (public EPA dataset) ----------------------------------------
BQ_DATASET = "bigquery-public-data.epa_historical_air_quality"
BQ_TABLES = {
    "ozone": "o3_daily_summary",
    "temp": "temperature_daily_summary",
    "pressure": "pressure_daily_summary",
    "wind": "wind_daily_summary",
    "rh": "rh_and_dp_daily_summary",
    "no2": "no2_daily_summary",
    "voc": "voc_daily_summary",
}
BQ_START_YEAR = 2000
BQ_END_YEAR = 2017
