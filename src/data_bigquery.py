"""Download + merge EPA Historical Air Quality daily tables from BigQuery.

Needs: pip install google-cloud-bigquery db-dtypes, and `gcloud auth application-default login`
(or GOOGLE_APPLICATION_CREDENTIALS) with any GCP project that has BigQuery enabled.
Table names live in config.BQ_TABLES - verify them in the BigQuery console if a query fails.
Unlike the paper, we LEFT JOIN on ozone so sites missing NO2/VOC are kept (imputed later)
instead of being thrown away by an inner join.
"""
import pandas as pd
import config as C

_AGG = """
  SELECT state_code, county_code, site_num, date_local, AVG(arithmetic_mean) AS {name}
  FROM `{ds}.{tbl}`
  WHERE EXTRACT(YEAR FROM date_local) BETWEEN {y0} AND {y1} {extra}
  GROUP BY state_code, county_code, site_num, date_local"""

_PARAM = {"temp": "AND parameter_code = 62101", "pressure": "AND parameter_code = 64101",
          "wind": "AND parameter_code IN (61101, 61103)", "rh": "AND parameter_code = 62201",
          "no2": "AND parameter_code = 42602", "voc": ""}


def _q(name, extra=""):
    return _AGG.format(name=name, ds=C.BQ_DATASET, tbl=C.BQ_TABLES[name],
                       y0=C.BQ_START_YEAR, y1=C.BQ_END_YEAR, extra=extra)


def build_query():
    o3 = f"""
  SELECT state_code, county_code, site_num, date_local,
         MAX(latitude) AS latitude, MAX(longitude) AS longitude,
         MAX(first_max_value) AS ozone
  FROM `{C.BQ_DATASET}.{C.BQ_TABLES['ozone']}`
  WHERE EXTRACT(YEAR FROM date_local) BETWEEN {C.BQ_START_YEAR} AND {C.BQ_END_YEAR}
  GROUP BY state_code, county_code, site_num, date_local"""
    parts = [f"WITH o3 AS ({o3})"]
    for k, extra in _PARAM.items():
        parts.append(f", {k} AS ({_q(k, extra)})")
    keys = "USING (state_code, county_code, site_num, date_local)"
    joins = " ".join(f"LEFT JOIN {k} {keys}" for k in _PARAM)
    parts.append(f"""
SELECT o3.*, temp.temp, pressure.pressure, wind.wind, rh.rh, no2.no2, voc.voc
FROM o3 {joins}""")
    return "\n".join(parts)


def download():
    from google.cloud import bigquery
    df = bigquery.Client().query(build_query()).to_dataframe()
    df["site_id"] = (df.state_code.astype(str).str.zfill(2) + "-" +
                     df.county_code.astype(str).str.zfill(3) + "-" +
                     df.site_num.astype(str).str.zfill(4))
    df["date"] = pd.to_datetime(df["date_local"])
    return df.drop(columns=["site_num", "date_local"])
