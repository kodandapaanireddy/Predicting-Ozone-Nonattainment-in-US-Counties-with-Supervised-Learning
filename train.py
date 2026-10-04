"""End-to-end pipeline: data -> features -> models -> county-level nonattainment -> artifacts.

python train.py --source synthetic            # offline demo data
python train.py --source csv --csv path.csv   # your own file (see README for schema)
python train.py --source bigquery             # real EPA data (needs GCP credentials)
Add --fast for a quick smoke run.
"""
import argparse, json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config as C
from src import county as cty, evaluate as ev, features as F, models as M


def get_data(args):
    if args.source == "synthetic":
        from src.data_synthetic import generate
        df = generate(n_counties=12 if args.fast else 30, start=2008 if args.fast else 2000, seed=C.SEED)
    elif args.source == "csv":
        df = pd.read_csv(args.csv)
    else:
        from src.data_bigquery import download
        df = download()
    df = F.load_clean(df)
    C.DATA_DIR.mkdir(exist_ok=True); df.to_csv(C.CLEAN_CSV, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "csv", "bigquery"], default="synthetic")
    ap.add_argument("--csv"); ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    t0 = time.time(); C.OUT_DIR.mkdir(exist_ok=True)
    R = {"source": args.source, "fast": args.fast}

    df = F.add_features(get_data(args))
    tr, va, te = F.split(df)
    R["rows"] = {"train": len(tr), "val": len(va), "test": len(te), "sites": int(df.site_id.nunique()),
                 "counties": int(df.groupby(["state_code", "county_code"]).ngroups)}
    print("rows", R["rows"])
    cols = F.FEATURE_SETS["full"]
    Xtr, ytr, Xva, yva, Xte, yte = tr[cols], tr[F.TARGET], va[cols], va[F.TARGET], te[cols], te[F.TARGET]

    # ---- 1. regression benchmark -------------------------------------------------
    fitted, table = {}, {}
    for name, m in {"persistence": M.Persistence(), "climatology": M.SiteMonthClimatology()}.items():
        m.fit(Xtr, ytr); fitted[name] = m
    for name, m in M.make_regressors(args.fast, C.SEED).items():
        print("fitting", name); m.fit(Xtr, ytr); fitted[name] = m
    print("tuning svr"); svr, svr_params, svr_grid = M.tune_svr(Xtr, ytr, Xva, yva, fast=args.fast)
    fitted["svr"] = svr; R["svr_best_params"], R["svr_grid"] = svr_params, svr_grid
    for name, m in fitted.items():
        table[name] = {"val": ev.reg_metrics(yva, m.predict(Xva)), "test": ev.reg_metrics(yte, m.predict(Xte))}
    best = min((n for n in table if n not in ("persistence", "climatology")), key=lambda n: table[n]["val"]["rmse"])
    R["regression"], R["best_model"] = table, best
    print("best (by val RMSE):", best, table[best]["test"])
    ev.plot_model_comparison(table, C.OUT_DIR / "model_comparison.png")
    pte = fitted[best].predict(Xte)
    ev.plot_pred_vs_actual(yte.values, pte, C.OUT_DIR / "pred_vs_actual.png", f"{best} - test years")

    # ---- 2. ablation: do we still predict without yesterday's ozone? ---------------
    nl = M.make_regressors(args.fast, C.SEED)["hgb"].fit(tr[F.FEATURE_SETS["no_lag"]], ytr)
    R["ablation_no_lag"] = ev.reg_metrics(yte, nl.predict(te[F.FEATURE_SETS["no_lag"]]))

    # ---- 3. feature importance (paper's future work) --------------------------------
    sub = va.sample(min(4000, len(va)), random_state=C.SEED)
    pi = permutation_importance(fitted[best], sub[cols], sub[F.TARGET], n_repeats=3, random_state=C.SEED)
    R["feature_importance"] = dict(sorted(zip(cols, map(float, pi.importances_mean)), key=lambda kv: -kv[1]))
    ev.plot_importance(cols, pi.importances_mean, C.OUT_DIR / "feature_importance.png")

    # ---- 4. daily exceedance classifier -----------------------------------------------
    clf = M.make_classifier(C.SEED, args.fast).fit(Xtr, (ytr > C.NAAQS_PPM).astype(int))
    pva, pte_p = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    yv_c, yt_c = (yva > C.NAAQS_PPM).astype(int), (yte > C.NAAQS_PPM).astype(int)
    thr = ev.best_f1_threshold(yv_c, pva)
    R["daily_exceedance"] = {"threshold": thr, "base_rate_test": float(yt_c.mean()),
                             "classifier": ev.cls_metrics(yt_c, pte_p >= thr, pte_p),
                             "from_regression": ev.cls_metrics(yt_c, pte >= C.NAAQS_PPM)}

    # ---- 5. county-level nonattainment ---------------------------------------------------
    ev_df = pd.concat([va, te]).copy()           # 2015-2017: none of it seen in training
    ev_df["pred_reg"] = fitted[best].predict(ev_df[cols]); ev_df["pred_prob"] = clf.predict_proba(ev_df[cols])[:, 1]
    obs_sy = cty.site_year_p4(ev_df, F.TARGET); prd_sy = cty.site_year_p4(ev_df, "pred_reg")
    sy = obs_sy.merge(prd_sy, on=["state_code", "county_code", "site_id", "year"], suffixes=("_obs", "_pred"))
    cal_rows = sy[(sy.year == C.VAL_YEAR)].dropna()
    a, b = np.polyfit(cal_rows.p4_pred, cal_rows.p4_obs, 1)       # fix regression-to-the-mean (fit on val year only)
    sy["p4_cal"] = a * sy.p4_pred + b
    # (a) annual county status via expected exceedance days; (b) via calibrated regression p4
    obs_c = sy.groupby(["state_code", "county_code", "year"])["p4_obs"].max().rename("obs_p4").reset_index()
    reg_c = sy.groupby(["state_code", "county_code", "year"])["p4_cal"].max().rename("pred_p4").reset_index()
    exp_c = cty.expected_exceedances(ev_df, "pred_prob")
    cy = obs_c.merge(reg_c).merge(exp_c)
    cy["obs_na"] = (cy.obs_p4 > C.NAAQS_PPM).astype(int)
    cy["pred_na_reg"] = (cy.pred_p4 > C.NAAQS_PPM).astype(int)
    cy["pred_na_cnt"] = (cy.exp_days >= 4).astype(int)
    R["county_year_counts"] = {"n": len(cy), "nonattainment_rate": float(cy.obs_na.mean())}
    R["county_year_regression"] = ev.cls_metrics(cy.obs_na, cy.pred_na_reg)
    R["county_year_exceedance_count"] = ev.cls_metrics(cy.obs_na, cy.pred_na_cnt, cy.exp_days)
    # (c) official-style 3-year design value for the last test year, with severity tiers
    o_dv = cty.county_dv(cty.site_design_values(sy.rename(columns={"p4_obs": "p4"}), "p4"))
    p_dv = cty.county_dv(cty.site_design_values(sy.rename(columns={"p4_cal": "p4"}), "p4"))
    dv = o_dv.merge(p_dv, on=["state_code", "county_code", "year"], suffixes=("_obs", "_pred"))
    dv = dv[dv.year == max(C.TEST_YEARS)].copy()
    if len(dv):
        dv["tier_obs"], dv["tier_pred"] = cty.tier_of(dv.dv_obs), cty.tier_of(dv.dv_pred)
        dv["na_obs"], dv["na_pred"] = (dv.dv_obs > C.NAAQS_PPM).astype(int), (dv.dv_pred > C.NAAQS_PPM).astype(int)
        R["county_dv_2017"] = {"n": len(dv), **ev.cls_metrics(dv.na_obs, dv.na_pred),
                               "dv_mae_ppm": float((dv.dv_obs - dv.dv_pred).abs().mean()),
                               "tier_exact": float((dv.tier_obs == dv.tier_pred).mean()),
                               "tier_within_1": float(((dv.tier_obs - dv.tier_pred).abs() <= 1).mean())}
        ev.plot_county_scatter(dv.dv_obs, dv.dv_pred, C.NAAQS_PPM, C.OUT_DIR / "county_dv_scatter.png")
        dv["tier_obs_name"] = dv.tier_obs.map(dict(enumerate(C.TIER_NAMES)))
        dv["tier_pred_name"] = dv.tier_pred.map(dict(enumerate(C.TIER_NAMES)))
        dv.to_csv(C.OUT_DIR / "county_status_2017.csv", index=False)
    cy.to_csv(C.OUT_DIR / "county_year_status.csv", index=False)

    # ---- 6. save -----------------------------------------------------------------------------
    joblib.dump({"model": fitted[best], "name": best, "features": cols, "classifier": clf,
                 "calibration": (float(a), float(b))}, C.OUT_DIR / "model.joblib")
    R["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(R, open(C.OUT_DIR / "metrics.json", "w"), indent=2)
    print(json.dumps({k: R[k] for k in ("best_model", "ablation_no_lag", "daily_exceedance", "county_year_regression",
                                       "county_year_exceedance_count", "county_dv_2017") if k in R}, indent=1))


if __name__ == "__main__":
    main()
