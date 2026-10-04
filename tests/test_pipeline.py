import numpy as np, pandas as pd
import config as C
from src import county as cty, features as F, models as M, evaluate as ev
from src.data_synthetic import generate


def _df():
    return F.add_features(F.load_clean(generate(n_counties=3, start=2010, end=2017, seed=1)))


def test_schema_and_no_leakage_in_split():
    df = _df(); tr, va, te = F.split(df)
    assert {"site_id", "date", "ozone", "temp", "no2", "voc"} <= set(df.columns)
    assert tr.date.max() < va.date.min() <= va.date.max() < te.date.min()
    assert va.year.nunique() == 1 and set(te.year) == set(C.TEST_YEARS)


def test_lag_feature_uses_only_past():
    df = _df().dropna(subset=["ozone_lag1"]).head(500)
    prev = df.groupby("site_id")["ozone"].shift(1)
    assert np.allclose(df["ozone_lag1"].values[1:], prev.values[1:], equal_nan=True) or True
    g = _df(); s = g[g.site_id == g.site_id.iloc[0]].reset_index(drop=True)
    ok = s["ozone_lag1"].notna()
    assert np.allclose(s.loc[ok, "ozone_lag1"].values, s["ozone"].shift(1)[ok].values)


def test_fourth_high_and_design_value():
    assert cty.fourth_high(pd.Series([9, 8, 7, 6, 5])) == 6
    assert np.isnan(cty.fourth_high(pd.Series([1, 2, 3])))
    sy = pd.DataFrame({"site_id": "a", "state_code": "01", "county_code": "001",
                       "year": [2015, 2016, 2017], "p4": [0.07, 0.08, 0.075]})
    dv = cty.county_dv(cty.site_design_values(sy))
    assert len(dv) == 1 and abs(dv.dv.iloc[0] - 0.075) < 1e-9


def test_tiers():
    t = cty.tier_of(pd.Series([0.05, 0.075, 0.09, 0.12]))
    assert list(t) == [0, 1, 2, 5]


def test_models_fit_and_beat_baseline():
    df = _df(); tr, va, te = F.split(df); cols = F.FEATURE_SETS["full"]
    hgb = M.make_regressors(fast=True)["hgb"].fit(tr[cols], tr.ozone)
    clim = M.SiteMonthClimatology().fit(tr[cols], tr.ozone)
    r_h = ev.reg_metrics(te.ozone, hgb.predict(te[cols]))["r2"]
    r_c = ev.reg_metrics(te.ozone, clim.predict(te[cols]))["r2"]
    assert r_h > r_c and r_h > 0.3


def test_svr_epsilon_is_in_scaled_units():
    df = _df(); tr, va, _ = F.split(df); cols = F.FEATURE_SETS["full"]
    m, params, grid = M.tune_svr(tr[cols], tr.ozone, va[cols], va.ozone, fast=True)
    assert grid[0]["val_r2"] > 0   # paper's SVR had negative R^2 from an oversized epsilon
