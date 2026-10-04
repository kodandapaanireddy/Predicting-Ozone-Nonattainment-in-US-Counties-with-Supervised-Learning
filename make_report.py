"""Builds the 2-page PDF write-up from outputs/metrics.json (re-run after any new training run)."""
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import config as C

R = json.load(open(C.OUT_DIR / "metrics.json"))
ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.6, leading=10.6, spaceAfter=3)
H = ParagraphStyle("h", parent=ss["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#264653"))
T = ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=2)
reg, de = R["regression"], R["daily_exceedance"]
MLP_STOPPED_EARLY = True   # set False if you retrain the MLP to completion
data_note = ("synthetic demo data (offline mode)" if R["source"] == "synthetic" else "EPA Historical Air Quality data")

rows = [["Model", "Val R2", "Test R2", "Test RMSE (ppm)", "Test MAE (ppm)"]]
for n in ["persistence", "climatology", "ridge", "svr", "mlp", "hgb"]:
    rows.append([n, f"{reg[n]['val']['r2']:.3f}", f"{reg[n]['test']['r2']:.3f}",
                 f"{reg[n]['test']['rmse']:.4f}", f"{reg[n]['test']['mae']:.4f}"])
tbl = Table(rows, colWidths=[1.3 * inch] + [1.15 * inch] * 4)
tbl.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#264653")),
                         ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                         ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
cy, cc, dv = R["county_year_regression"], R["county_year_exceedance_count"], R.get("county_dv_2017", {})
best_ = R["best_model"]; bt_, rt_, pt_ = reg[best_]["test"], reg["ridge"]["test"], reg["persistence"]["test"]
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
doc = SimpleDocTemplate(str(C.OUT_DIR / "writeup.pdf"), pagesize=letter, leftMargin=.6 * inch, rightMargin=.6 * inch,
                        topMargin=.5 * inch, bottomMargin=.45 * inch)
S = [Paragraph("Predicting Ozone Nonattainment in US Counties with Supervised Learning", T),
     Paragraph("Team No : 25 | 1. Name : ATHARVA SIDRAMAPPA GOUNI | SRN : PES2UG24CS094 |2. Name : B. Kodandapaani Reddy | SRN : PES2UG24CS108 | Machine Learning project write-up", ParagraphStyle("c", parent=B, alignment=1)),
     Paragraph("1. Problem statement", H),
     Paragraph("Ground-level ozone is a regulated pollutant. A US county is designated <b>nonattainment</b> when its ozone design value "
               "(3-year average of the annual 4th-highest daily maximum at the worst monitor) exceeds 0.070 ppm, which triggers costly mandated "
               "controls. We predict daily ozone from weather and precursor pollutants, then roll predictions up to <b>county nonattainment "
               "status and severity tier</b>, so regions can act before designation. Our work extends the CS229 (2019) ozone-regression report, "
               "fixing the weaknesses it listed.", B),
     Paragraph("2. Dataset", H),
     Paragraph(f"Source: EPA Historical Air Quality daily summaries (Google BigQuery public data, 2000-2017): ozone, temperature, pressure, wind, relative humidity, "
               f"NO<sub>2</sub>, VOC, joined by site and day with LEFT JOINs. The numbers below were produced on <b>{data_note}</b>: {R['rows']['train']:,} train / "
               f"{R['rows']['val']:,} validation / {R['rows']['test']:,} test site-days, {R['rows']['sites']} sites, {R['rows']['counties']} counties. "
               "" + (("The synthetic generator mirrors the BigQuery schema so the pipeline runs offline; <font face='Courier'>python train.py --source bigquery</font> reproduces everything on real data.")
                     if R["source"] == "synthetic" else
                     ("Sites missing a sensor (NO<sub>2</sub>, VOC, wind, ...) are kept and imputed rather than dropped. Reproduce with <font face='Courier'>python train.py --source bigquery</font>.")), B),
     Paragraph("3. Approach", H),
     Paragraph("<b>Fixes over the paper:</b> (i) time-based split by whole calendar years (train &le;2014, validation 2015, test 2016-17) instead of a "
               "Jan-Jun / Jul-Dec seasonal split; (ii) standard R<super>2</super> = 1 - SSE/SST (the paper divided by &Sigma;y<super>2</super>); "
               "(iii) latitude/longitude instead of a numeric site ID; (iv) missing sensors kept via median imputation plus missing-indicators or native NaN "
               "handling, instead of dropping rows; (v) SVR target scaled so &epsilon; is meaningful and tuned (paper's &epsilon; was too large, giving negative R<super>2</super>); "
               "(vi) naive baselines and a no-persistence ablation; (vii) permutation feature importance; (viii) a county-level classifier with severity tiers.", B),
     Paragraph("<b>Models:</b> Ridge regression, RBF support vector regression, MLP (128-128-64, ReLU, adaptive learning rate, early stopping), histogram gradient "
               "boosting, plus persistence and site-month climatology baselines. The best model on validation RMSE (" + R["best_model"] + ") is used downstream. "
               "A gradient-boosting classifier gives P(daily ozone &gt; 0.070 ppm); summing probabilities per site-year gives expected exceedance days, and "
               "&ge;4 means the 4th-highest day exceeds the standard. Regression outputs are rolled into annual 4th-high values, bias-calibrated on the validation year, "
               "averaged over 3 years into design values, and binned into Marginal-Extreme tiers.", B),
     Image(str(C.DOCS_DIR / "architecture.png"), width=7.2 * inch, height=3.1 * inch),
     Paragraph("4. Implementation overview", H),
     Paragraph("Python (pandas, scikit-learn, matplotlib). <font face='Courier'>src/data_bigquery.py</font> and <font face='Courier'>data_synthetic.py</font> load data; "
               "<font face='Courier'>features.py</font> builds leakage-safe features (lags use only past days); <font face='Courier'>models.py</font> holds the model zoo; "
               "<font face='Courier'>county.py</font> implements EPA design-value logic; <font face='Courier'>train.py</font> runs the full pipeline and writes metrics, plots and a saved model; "
               "<font face='Courier'>demo.py</font> answers what-if questions (e.g. +4&deg;F) per county; 6 pytest tests cover splits, lag leakage, design values and model sanity.", B),
     Paragraph("5. Results (test years 2016-2017)", H), tbl, Spacer(1, 4),
     Paragraph(f"Without yesterday's ozone the gradient-boosting model still reaches R<super>2</super> = {R['ablation_no_lag']['r2']:.3f} using meteorology and precursors alone. "
               f"Daily exceedance classifier: ROC-AUC {de['classifier']['roc_auc']:.3f}, PR-AUC {de['classifier']['pr_auc']:.3f}, F1 {de['classifier']['f1']:.3f} "
               f"(exceedance base rate {de['base_rate_test']:.1%}). County-year nonattainment (n={R['county_year_counts']['n']}): regression-based accuracy {cy['accuracy']:.3f}, "
               f"F1 {cy['f1']:.3f}; exceedance-count method accuracy {cc['accuracy']:.3f}, F1 {cc['f1']:.3f}. 2017 design values: accuracy {dv.get('accuracy', 0):.3f}, "
               f"MAE {dv.get('dv_mae_ppm', 0):.4f} ppm, severity tier exact {dv.get('tier_exact', 0):.2f} / within one tier {dv.get('tier_within_1', 0):.2f}.", B),
     Paragraph("6. Conclusions and limitations", H),
     Paragraph(CONCLUSION, B)]
doc.build(S)
print("wrote", C.OUT_DIR / "writeup.pdf")
# PATCHED_FOR_REAL_DATA
