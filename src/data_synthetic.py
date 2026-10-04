"""Synthetic EPA-like daily data so the whole pipeline runs offline.

The schema matches what data_bigquery.py produces. The generator encodes the
known chemistry/meteorology (heat + VOC/NOx + low wind -> more ozone, slow
downward emissions trend, day-to-day persistence) so models have real signal.
It is for demos/tests only; report real results from BigQuery data.
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter


def _ar1(rng, n, phi, sd):
    return lfilter([1.0], [1.0, -phi], rng.normal(0, sd, n))


def generate(n_counties=30, start=2000, end=2017, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{start}-01-01", f"{end}-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    yidx = (dates.year.values - start).astype(float)
    season = np.sin(2 * np.pi * (doy - 110) / 365.0)
    frames = []
    for c in range(n_counties):
        state, county = 1 + c // 6, 10 + c
        lat, lon = rng.uniform(26, 47), rng.uniform(-120, -72)
        mean_t, amp = 24 - 0.45 * (lat - 26) + rng.normal(0, 1.5), rng.uniform(8, 13)
        urban = rng.uniform(0.2, 1.0)
        county_bias = rng.normal(0, 0.006)
        for s in range(int(rng.integers(1, 4))):
            temp_c = mean_t - 5 + amp * season + _ar1(rng, n, 0.7, 2.2)
            wind = np.clip(5 + _ar1(rng, n, 0.5, 1.6) - 0.6 * urban, 0.3, None)
            pres = 1013 - 0.02 * rng.uniform(0, 800) + _ar1(rng, n, 0.8, 2.0)
            rh = np.clip(62 - 0.8 * (temp_c - mean_t) + _ar1(rng, n, 0.6, 9), 8, 100)
            no2 = np.clip(urban * 22 * (1 - 0.25 * season) * rng.lognormal(0, 0.35, n)
                          * (1 - 0.03 * yidx), 0.5, None)
            voc = np.clip(urban * 180 * rng.lognormal(0, 0.4, n) * (1 - 0.02 * yidx), 5, None)
            hot = np.clip(temp_c - 12, 0, None)
            o3 = (0.052 + county_bias + rng.normal(0, 0.002)
                  + 0.00115 * hot + 0.00001 * hot * voc / 10.0
                  - 0.0009 * wind + 0.00005 * (pres - 1013)
                  - 0.00018 * np.clip(no2 - 25, 0, None) * (hot < 8)
                  - 0.0003 * yidx + 0.00006 * (60 - rh) / 10
                  + _ar1(rng, n, 0.55, 0.0042))
            o3 = np.clip(o3, 0.004, 0.16)
            df = pd.DataFrame({
                "site_id": f"{state:02d}-{county:03d}-{s + 1:04d}",
                "state_code": f"{state:02d}", "county_code": f"{county:03d}",
                "latitude": lat + rng.normal(0, 0.05), "longitude": lon + rng.normal(0, 0.05),
                "date": dates, "ozone": o3,
                "temp": temp_c * 9 / 5 + 32, "pressure": pres, "wind": wind, "rh": rh,
                "no2": no2, "voc": voc})
            # realistic missingness: not every monitor measures everything
            for col, p_site in [("pressure", 0.35), ("wind", 0.15), ("rh", 0.3),
                                ("no2", 0.4), ("voc", 0.85)]:
                if rng.random() < p_site:
                    df[col] = np.nan
                else:
                    df.loc[rng.random(n) < 0.04, col] = np.nan
            frames.append(df[rng.random(n) > 0.03])
    return pd.concat(frames, ignore_index=True)
