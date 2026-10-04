// Builds outputs/presentation.pptx from outputs/metrics.json  (node make_deck.js)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const R = JSON.parse(fs.readFileSync("outputs/metrics.json"));
const p = new pptxgen(); p.layout = "LAYOUT_16x9"; p.title = "Predicting Ozone Nonattainment in US Counties";
const NAVY = "1D3557", TEAL = "2A9D8F", LIGHT = "F4F8F8", INK = "22313F";
const f = (x, d = 3) => Number(x).toFixed(d);
const MLP_STOPPED_EARLY = true;  // set false if you retrain the MLP to completion
const reg = R.regression, de = R.daily_exceedance, cy = R.county_year_regression, cc = R.county_year_exceedance_count, dv = R.county_dv_2017 || {};

function slide(title, section) {
  const s = p.addSlide(); s.background = { color: "FFFFFF" };
  s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: 10, h: 0.9, fill: { color: NAVY }, line: { color: NAVY } });
  s.addText(title, { x: 0.5, y: 0.1, w: 9, h: 0.7, fontSize: 24, bold: true, color: "FFFFFF", fontFace: "Calibri", isTextBox: true, margin: 0 });
  return s;
}
const bullets = (s, items, o = {}) => s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
  Object.assign({ x: 0.6, y: 1.2, w: 8.8, h: 4, fontSize: 16, color: INK, paraSpaceAfter: 8, valign: "top", fontFace: "Calibri", isTextBox: true }, o));

let s = p.addSlide(); s.background = { color: NAVY };
s.addText("Predicting Ozone Nonattainment in US Counties", { x: 0.6, y: 1.5, w: 8.8, h: 1.2, fontSize: 36, bold: true, color: "FFFFFF", fontFace: "Calibri", isTextBox: true });
s.addText("with Supervised Learning  | Team No : 25 | 1. Name : ATHARVA SIDRAMAPPA GOUNI | SRN : PES2UG24CS094 |2. Name : B. Kodandapaani Reddy | SRN : PES2UG24CS108 |  Machine Learning project", { x: 0.6, y: 3.0, w: 8.8, h: 0.5, fontSize: 18, color: "A8DADC", fontFace: "Calibri", isTextBox: true });

s = slide("Introduction & problem statement");
bullets(s, ["Ozone near the ground harms health; EPA limit = 0.070 ppm (design value)",
  "Nonattainment counties face costly, mandatory controls",
  "Goal: predict daily ozone, then flag county nonattainment + severity tier early",
  "Base: CS229 (2019) ozone regression paper; we fix its drawbacks and add county-level output"]);

s = slide("Paper drawbacks and our fixes");
bullets(s, ["Seasonal split (val Jan-Jun, test Jul-Dec) -> whole-year temporal split",
  "Non-standard R2 and oversized SVR epsilon (R2 = -2.36) -> standard metrics, tuned SVR",
  "Numeric site ID feature -> latitude/longitude; inner joins -> LEFT JOIN + imputation",
  "Regression only -> exceedance classifier, design value, severity tiers",
  "No feature analysis -> permutation importance and ablation"], { fontSize: 15 });

s = slide("Dataset");
bullets(s, ["EPA Historical Air Quality daily summaries, Google BigQuery public data (2000-2017)",
  "Ozone target + temperature, pressure, wind, humidity, NO2, VOC per site-day",
  "Engineered: calendar cycles, lags, rolling temperature, lat/lon, trend",
  `Split: train <=2014 (${R.rows.train.toLocaleString()}), val 2015 (${R.rows.val.toLocaleString()}), test 2016-17 (${R.rows.test.toLocaleString()})`,
  (R.source === "synthetic" ? "Offline synthetic generator (same schema) used for the numbers shown here" : `${R.rows.sites} monitoring sites, ${R.rows.counties} counties; sites missing a sensor are kept and imputed`)], { fontSize: 15 });

s = slide("System architecture");
s.addImage({ path: "docs/architecture.png", x: 0.4, y: 1.1, w: 9.2, h: 3.98 });

s = slide("Pipeline flow");
s.addImage({ path: "docs/flow.png", x: 0.3, y: 1.4, w: 9.4, h: 2.67 });
s.addText("python train.py runs steps 1-8 end to end and writes metrics, plots, saved model", { x: 0.6, y: 4.4, w: 8.8, h: 0.5, fontSize: 14, color: INK, isTextBox: true });

s = slide("Algorithms used");
bullets(s, ["Ridge: linear baseline, y = theta.x with L2 penalty",
  "SVR (RBF): epsilon-insensitive loss, scaled target, grid-tuned C/epsilon",
  "MLP: 128-128-64, ReLU, adaptive learning rate, early stopping",
  "Gradient boosting: tree ensemble, handles missing sensors natively; also the exceedance classifier",
  "Baselines: persistence and site-month climatology",
  "County: 4th-high -> 3-yr design value -> tier; expected exceedance days = sum of P(exceed)"], { fontSize: 14 });

s = slide("Implementation");
bullets(s, ["Python: pandas, scikit-learn, matplotlib; modular src/ package",
  "Leakage safe: lags use only past days, normalisation fitted on train only, time-based splits",
  "Model chosen on validation RMSE; test years touched once",
  "demo.py: per-county what-if (e.g. +4 F) for the live demonstration",
  "6 pytest unit tests; README with setup and run steps; private GitHub repo shared with faculty/TAs"], { fontSize: 15 });

s = slide("Results: daily ozone regression (test 2016-17)");
const names = ["persistence", "climatology", "ridge", "svr", "mlp", "hgb"];
s.addChart(p.charts.BAR, [{ name: "Test R2", labels: names, values: names.map(n => +f(reg[n].test.r2, 3)) }],
  { x: 0.4, y: 1.1, w: 5.6, h: 3.9, barDir: "col", chartColors: [TEAL], showValue: true, dataLabelFormatCode: "0.00", showLegend: false,
    showTitle: true, title: "Test R2 (higher is better)", valAxisMinVal: Math.floor(Math.min(0, ...names.map(n => reg[n].test.r2)) * 10) / 10, valGridLine: { color: "E0E0E0", size: 0.5 }, catGridLine: { style: "none" } });
s.addText([{ text: `Best: ${R.best_model.toUpperCase()}`, options: { bold: true, breakLine: true } },
  { text: `RMSE ${f(reg[R.best_model].test.rmse, 4)} ppm`, options: { breakLine: true } },
  { text: `No-lag ablation R2 ${f(R.ablation_no_lag.r2, 2)}`, options: { breakLine: true } },
  { text: `Exceedance ROC-AUC ${f(de.classifier.roc_auc, 2)}`, options: {} }],
  { x: 6.3, y: 1.6, w: 3.4, h: 2.5, fontSize: 15, color: INK, isTextBox: true, valign: "top" });

s = slide("Results: county nonattainment");
s.addImage({ path: "outputs/county_dv_scatter.png", x: 0.5, y: 1.1, w: 3.9, h: 3.9 });
bullets(s, [`County-year accuracy ${f(cy.accuracy, 2)}, F1 ${f(cy.f1, 2)}`,
  `Exceedance-count method accuracy ${f(cc.accuracy, 2)}`,
  `2017 design value MAE ${f(dv.dv_mae_ppm, 4)} ppm`,
  `Severity tier exact ${f(dv.tier_exact, 2)}, within one ${f(dv.tier_within_1, 2)}`],
  { x: 4.7, y: 1.4, w: 4.9, h: 3.4, fontSize: 15 });

s = slide("Limitations and future work");
bullets(s, [(R.source === "synthetic" ? "Reported numbers use synthetic data; rerun on BigQuery before claiming real-world accuracy" : `Weather + precursors alone explain only part of daily ozone (no-lag R2 ${f(R.ablation_no_lag.r2, 2)}); nonattainment recall ${f(dv.recall, 2)}`),
  "Target is 1-hour max, not the regulatory 8-hour average" + (MLP_STOPPED_EARLY && R.source !== "synthetic" ? "; MLP stopped early and SVR trained on a 15k subsample for time" : ""),
  "No completeness rules, exceptional events, or emission-source features (factories, highways, census)",
  "Future: 8-hour metrics, spatial/graph models, uncertainty intervals, other pollutants"], { fontSize: 15 });

s = slide("Conclusion");
bullets(s, [`${R.best_model.toUpperCase()} is the best model (test R2 ${f(reg[R.best_model].test.r2, 2)}) on a fair, leakage-free split`,
  `County design values: ${f(dv.accuracy, 2)} accuracy, tiers within one level ${f(dv.tier_within_1, 2)}`,
  "Every paper drawback addressed; pipeline reproducible with one command",
  "Live demo: python demo.py --state 01 --county 012 --temp-delta-f 4"], { fontSize: 16 });

p.writeFile({ fileName: "outputs/presentation.pptx" }).then(() => console.log("deck written"));
// PATCHED_FOR_REAL_DATA
