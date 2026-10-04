"""Live demo: load the trained model and ask 'what if it were X degrees hotter?'

python demo.py --list
python demo.py --state 01 --county 012
python demo.py --state 01 --county 012 --temp-delta-f 4
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import config as C
from src import county as cty, features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state"); ap.add_argument("--county"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--temp-delta-f", type=float, default=0.0, help="add this many deg F to every day's temperature")
    ap.add_argument("--year", type=int, default=max(C.TEST_YEARS))
    a = ap.parse_args()

    status = pd.read_csv(C.OUT_DIR / "county_status_2017.csv", dtype={"state_code": str, "county_code": str})
    if a.list or not (a.state and a.county):
        print(status[["state_code", "county_code", "dv_obs", "dv_pred", "tier_obs_name", "tier_pred_name"]]
              .round(4).to_string(index=False)); return

    bundle = joblib.load(C.OUT_DIR / "model.joblib")
    model, cols, (ca, cb) = bundle["model"], bundle["features"], bundle["calibration"]
    raw = F.load_clean(pd.read_csv(C.CLEAN_CSV, dtype={"state_code": str, "county_code": str}))
    raw = raw[(raw.state_code == a.state) & (raw.county_code == a.county)]
    if raw.empty:
        raise SystemExit("county not found; run `python demo.py --list`")
    obs = F.add_features(raw)
    mod = raw.copy(); mod["temp"] = mod["temp"] + a.temp_delta_f
    mod = F.add_features(mod); mod["ozone_lag1"] = obs["ozone_lag1"].values   # keep observed persistence signal
    rows = []
    for label, d in [("baseline", obs), (f"+{a.temp_delta_f}F", mod)]:
        d = d[d.year >= a.year - 2].copy(); d["pred"] = model.predict(d[cols])
        sy = cty.site_year_p4(d, "pred"); sy["p4"] = ca * sy["p4"] + cb
        dv = cty.county_dv(cty.site_design_values(sy)); dv = dv[dv.year == a.year]
        v = float(dv.dv.iloc[0]) if len(dv) else float("nan")
        rows.append((label, v, "NONATTAINMENT" if v > C.NAAQS_PPM else "attainment",
                     C.TIER_NAMES[int(cty.tier_of(pd.Series([v])).iloc[0])] if v == v else "n/a"))
    print(f"County {a.state}-{a.county}, design value year {a.year} (standard {C.NAAQS_PPM} ppm)")
    for r in rows:
        print(f"  {r[0]:>10}: DV={r[1]:.4f} ppm -> {r[2]} ({r[3]})")


if __name__ == "__main__":
    main()
