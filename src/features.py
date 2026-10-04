"""Feature engineering and temporal splits."""
import numpy as np
import pandas as pd
import config as C

BASE = ["temp", "pressure", "wind", "rh", "no2", "voc"]
CALENDAR = ["month_sin", "month_cos", "doy_sin", "doy_cos", "year_trend"]
GEO = ["latitude", "longitude"]               # replaces the paper's meaningless numeric site id
DERIVED = ["temp_roll3", "temp_lag1", "temp_max_anomaly"]
LAG = ["ozone_lag1"]                          # yesterday's observed ozone (persistence signal)

FEATURE_SETS = {
    "full": BASE + CALENDAR + GEO + DERIVED + LAG,
    "no_lag": BASE + CALENDAR + GEO + DERIVED,        # meteorology + precursors only
}
TARGET = "ozone"


def load_clean(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[TARGET])
    df = df[(df[TARGET] >= 0) & (df[TARGET] < 0.5)]
    df = df.drop_duplicates(["site_id", "date"])
    return df.sort_values(["site_id", "date"]).reset_index(drop=True)


def add_features(df):
    df = df.sort_values(["site_id", "date"]).copy()
    d = df["date"]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    doy = d.dt.dayofyear
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["year_trend"] = df["year"] - C.TRAIN_END_YEAR
    g = df.groupby("site_id", sort=False)
    gap = g["date"].diff().dt.days
    df["ozone_lag1"] = g[TARGET].shift(1).where(gap == 1)
    df["temp_lag1"] = g["temp"].shift(1).where(gap == 1)
    df["temp_roll3"] = g["temp"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    site_mean = g["temp"].transform("mean")
    df["temp_max_anomaly"] = df["temp"] - site_mean
    return df


def split(df):
    tr = df[df.year <= C.TRAIN_END_YEAR]
    va = df[df.year == C.VAL_YEAR]
    te = df[df.year.isin(C.TEST_YEARS)]
    return tr, va, te
