"""Draws docs/architecture.png and docs/flow.png (used in the report and slides)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import config as C

C.DOCS_DIR.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc="#e9f5f3", ec="#2a9d8f", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#264653", lw=1.5))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 5.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 0.2, 3.6, 2.6, 1.2, "EPA Historical Air Quality\n(BigQuery) or synthetic\ndaily site data", "#fff3e0", "#e76f51")
    box(ax, 3.3, 3.6, 2.6, 1.2, "Data layer\nLEFT JOIN O3 + temp, pressure,\nwind, RH, NO2, VOC")
    box(ax, 6.4, 3.6, 2.6, 1.2, "Feature layer\ncalendar, lags, rolling temp,\nlat/lon, anomalies")
    box(ax, 9.3, 3.6, 2.5, 1.2, "Temporal split\ntrain <=2014 | val 2015\ntest 2016-17")
    box(ax, 0.2, 1.6, 2.6, 1.2, "Regressors\nRidge, SVR, MLP,\nGradient Boosting", "#e3eefc", "#457b9d")
    box(ax, 3.3, 1.6, 2.6, 1.2, "Exceedance classifier\nP(ozone > 0.070 ppm)", "#e3eefc", "#457b9d")
    box(ax, 6.4, 1.6, 2.6, 1.2, "County aggregation\n4th-high, 3-yr design value,\nexpected exceedance days", "#f1e6fa", "#7b2cbf")
    box(ax, 9.3, 1.6, 2.5, 1.2, "Outputs\nNonattainment flag +\nseverity tier per county", "#f1e6fa", "#7b2cbf")
    box(ax, 3.3, 0.1, 5.7, 0.85, "Evaluation: R2/RMSE vs baselines, ablation, permutation importance, county precision/recall",
        "#f7f7f7", "#888888", 9)
    for x in (2.8, 5.9, 9.0): arrow(ax, x, 4.2, x + 0.5, 4.2)
    arrow(ax, 10.5, 3.6, 10.5, 3.2); arrow(ax, 10.5, 3.2, 1.5, 3.2); arrow(ax, 1.5, 3.2, 1.5, 2.8)
    arrow(ax, 4.6, 3.2, 4.6, 2.8); arrow(ax, 2.8, 2.2, 3.3, 2.2); arrow(ax, 5.9, 2.2, 6.4, 2.2)
    arrow(ax, 9.0, 2.2, 9.3, 2.2); arrow(ax, 7.7, 1.6, 7.7, 0.95)
    ax.set_title("System architecture", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(C.DOCS_DIR / "architecture.png", dpi=170); plt.close(fig)


def flow():
    fig, ax = plt.subplots(figsize=(12, 3.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    steps = ["1 Load\nBigQuery / CSV", "2 Clean\ndedupe, range\ncheck", "3 Features\nlags, calendar,\nrolling temp", "4 Split\nby year\n(no leakage)",
             "5 Train &\ntune\n5 models", "6 Select\nbest on\nvalidation", "7 Daily\nexceedance\nprobability", "8 County\ndesign value\n& tier"]
    w = 1.3
    for i, s in enumerate(steps):
        x = 0.1 + i * 1.47
        box(ax, x, 1.0, w, 1.4, s, "#e9f5f3" if i < 4 else "#e3eefc" if i < 6 else "#f1e6fa", "#2a9d8f" if i < 4 else "#457b9d" if i < 6 else "#7b2cbf", 8.5)
        if i < len(steps) - 1: arrow(ax, x + w, 1.7, x + 1.47, 1.7)
    ax.set_title("Pipeline flow (python train.py)", fontsize=13, weight="bold")
    fig.tight_layout(); fig.savefig(C.DOCS_DIR / "flow.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    architecture(); flow(); print("saved diagrams")
