"""Site -> county aggregation following the EPA design-value logic."""
import numpy as np
import pandas as pd
import config as C


def fourth_high(s):
    s = s.dropna().sort_values(ascending=False)
    return s.iloc[3] if len(s) >= 4 else np.nan


def site_year_p4(df, col):
    """Annual 4th-highest daily value per site."""
    return (df.groupby(["state_code", "county_code", "site_id", "year"])[col]
              .apply(fourth_high).rename("p4").reset_index())


def site_design_values(sy, value="p4"):
    """3-year average per site; requires 3 consecutive years of data."""
    out = []
    for site, g in sy.sort_values("year").groupby("site_id"):
        g = g.set_index("year")[value]
        for y in g.index:
            w = [y - 2, y - 1, y]
            if all(v in g.index for v in w) and g.loc[w].notna().all():
                out.append((site, y, g.loc[w].mean()))
    d = pd.DataFrame(out, columns=["site_id", "year", "site_dv"])
    meta = sy[["site_id", "state_code", "county_code"]].drop_duplicates("site_id")
    return d.merge(meta, on="site_id")


def county_dv(site_dv):
    """County DV = highest site DV in the county."""
    return (site_dv.groupby(["state_code", "county_code", "year"])["site_dv"].max()
                   .rename("dv").reset_index())


def tier_of(dv):
    return pd.cut(dv, bins=C.TIER_BINS, labels=range(len(C.TIER_NAMES)),
                  right=True, include_lowest=True).astype("Int64")


def expected_exceedances(df, prob_col):
    """Sum of P(exceed) per site-year -> expected exceedance days; >=4 <=> 4th-high above threshold."""
    se = df.groupby(["state_code", "county_code", "site_id", "year"])[prob_col].sum().rename("exp_days")
    return se.reset_index().groupby(["state_code", "county_code", "year"])["exp_days"].max().reset_index()
