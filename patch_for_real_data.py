"""Run once:  python patch_for_real_data.py
Makes make_report.py, make_deck.js and README.md describe REAL-data results correctly
(removes the hard-coded 'synthetic' wording and the 'non-linear beat SVR' claim)."""
import pathlib, sys


def sub(path, pairs):
    p = pathlib.Path(path); s = p.read_text(encoding="utf-8")
    if "PATCHED_FOR_REAL_DATA" in s:
        print(path, "already patched"); return
    for old, new in pairs:
        if old not in s:
            sys.exit(f"Could not find text in {path}:\n{old[:90]}...")
        s = s.replace(old, new, 1)
    p.write_text(s.rstrip("\n") + ("\n# PATCHED_FOR_REAL_DATA\n" if path.endswith(".py") else "\n// PATCHED_FOR_REAL_DATA\n"), encoding="utf-8")
    print("patched", path)


# ---------------- make_report.py ----------------
sub("make_report.py", [
 ('data_note = (',
  'MLP_STOPPED_EARLY = True   # set False if you retrain the MLP to completion\n'
  'data_note = ('),
 ('''               "The synthetic generator mirrors the BigQuery schema (including missing sensors) so the pipeline runs offline; "
               "<font face='Courier'>python train.py --source bigquery</font> reproduces everything on real data.", B),''',
  '''               "" + (("The synthetic generator mirrors the BigQuery schema so the pipeline runs offline; <font face='Courier'>python train.py --source bigquery</font> reproduces everything on real data.")
                     if R["source"] == "synthetic" else
                     ("Sites missing a sensor (NO<sub>2</sub>, VOC, wind, ...) are kept and imputed rather than dropped. Reproduce with <font face='Courier'>python train.py --source bigquery</font>.")), B),'''),
 ('doc = SimpleDocTemplate(',
  '''best_ = R["best_model"]; bt_, rt_, pt_ = reg[best_]["test"], reg["ridge"]["test"], reg["persistence"]["test"]
CONCLUSION = (
    f"{best_.upper()} was the best model on validation RMSE (test R<super>2</super> {bt_['r2']:.3f}, RMSE {bt_['rmse']:.4f} ppm), compared with Ridge "
    f"({rt_['r2']:.3f}) and the persistence baseline ({pt_['r2']:.3f}). Rolling daily predictions into county design values flags nonattainment with "
    f"{dv.get('accuracy', 0):.0%} accuracy and places counties within one severity tier {dv.get('tier_within_1', 0):.0%} of the time, but nonattainment recall is "
    f"{dv.get('recall', 0):.0%}, so some violating counties are missed. Without yesterday's ozone the model drops to R<super>2</super> {R['ablation_no_lag']['r2']:.2f}, "
    "showing that weather and NO<sub>2</sub>/VOC alone explain only part of daily ozone. <b>Limitations:</b> the daily target is the 1-hour maximum rather than the regulatory "
    "8-hour average, so 'nonattainment' here is a proxy; the SVR was trained on a 15,000-row subsample because it scales quadratically; "
    + ("the MLP was stopped early to save time on the large training set, so its score is a lower bound; " if (MLP_STOPPED_EARLY and R["source"] != "synthetic") else "")
    + ("results are on synthetic data and must be re-run on BigQuery data; " if R["source"] == "synthetic" else "")
    + "completeness rules and exceptional-event exclusions are not modelled; tier cut-offs are simplified; emission-source features (factories, highways, census data) "
    "and spatial/temporal deep models are future work.")
doc = SimpleDocTemplate('''),
 ('''     Paragraph("Flexible non-linear models (MLP, boosting) beat linear and SVR baselines, and rolling daily predictions into county design values gives accurate nonattainment "
               "flags and tiers. Limitations: results above are on synthetic data and must be re-run on BigQuery data before drawing real-world conclusions; the daily target is the "
               "1-hour maximum rather than the regulatory 8-hour average; completeness rules and exceptional-event exclusions are not modelled; tier cut-offs are simplified; "
               "emission-source features (factories, highways, census data) and spatial/temporal deep models are future work.", B)]''',
  '''     Paragraph(CONCLUSION, B)]'''),
])

# ---------------- make_deck.js ----------------
sub("make_deck.js", [
 ('const f = (x, d = 3) => Number(x).toFixed(d);',
  'const f = (x, d = 3) => Number(x).toFixed(d);\nconst MLP_STOPPED_EARLY = true;  // set false if you retrain the MLP to completion'),
 ('"Offline synthetic generator (same schema) used for the demo numbers shown here"',
  '(R.source === "synthetic" ? "Offline synthetic generator (same schema) used for the numbers shown here" : `${R.rows.sites} monitoring sites, ${R.rows.counties} counties; sites missing a sensor are kept and imputed`)'),
 ('"Reported numbers use synthetic data; rerun on BigQuery before claiming real-world accuracy"',
  '(R.source === "synthetic" ? "Reported numbers use synthetic data; rerun on BigQuery before claiming real-world accuracy" : `Weather + precursors alone explain only part of daily ozone (no-lag R2 ${f(R.ablation_no_lag.r2, 2)}); nonattainment recall ${f(dv.recall, 2)}`)'),
 ('"Target is 1-hour max, not the regulatory 8-hour average"',
  '"Target is 1-hour max, not the regulatory 8-hour average" + (MLP_STOPPED_EARLY && R.source !== "synthetic" ? "; MLP stopped early and SVR trained on a 15k subsample for time" : "")'),
 ('"Non-linear models beat linear and SVR baselines on a fair, leakage-free split"',
  '`${R.best_model.toUpperCase()} is the best model (test R2 ${f(reg[R.best_model].test.r2, 2)}) on a fair, leakage-free split`'),
 ('"Daily predictions roll up into accurate county nonattainment flags and tiers"',
  '`County design values: ${f(dv.accuracy, 2)} accuracy, tiers within one level ${f(dv.tier_within_1, 2)}`'),
 ('valAxisMinVal: 0,',
  'valAxisMinVal: Math.floor(Math.min(0, ...names.map(n => reg[n].test.r2)) * 10) / 10,'),
])

# ---------------- README.md ----------------
p = pathlib.Path("README.md"); s = p.read_text(encoding="utf-8")
old = "* Bundled numbers come from **synthetic** data; real-data accuracy will be lower. Report your BigQuery run."
if old in s:
    p.write_text(s.replace(old, "* Results in `outputs/` were produced on the real EPA BigQuery data (2000-2017). The offline synthetic generator is for demos/tests only."), encoding="utf-8")
    print("patched README.md")
