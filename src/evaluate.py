"""Metrics and plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)


def reg_metrics(y, p):
    # NB: sklearn's R^2 = 1 - SSE/SST (about the mean). The paper divided by sum(y^2),
    # which is not the standard definition and inflates R^2.
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))), "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p))}


def cls_metrics(y, pred, prob=None):
    m = {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
         "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0))}
    if prob is not None and len(set(y)) > 1:
        m["roc_auc"] = float(roc_auc_score(y, prob)); m["pr_auc"] = float(average_precision_score(y, prob))
    return m


def best_f1_threshold(y, prob):
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[int(np.argmax([f1_score(y, prob >= t, zero_division=0) for t in ts]))])


def plot_pred_vs_actual(y, p, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = ax.hexbin(y, p, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    lim = [0, max(y.max(), p.max())]
    ax.plot(lim, lim, "r--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed daily max ozone (ppm)"); ax.set_ylabel("Predicted (ppm)"); ax.set_title(title)
    fig.colorbar(hb, label="log10(count)"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_model_comparison(table, path):
    names = list(table); r2 = [table[n]["test"]["r2"] for n in names]; rm = [table[n]["test"]["rmse"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(names, r2, color="#2a9d8f"); ax[0].set_title("Test R$^2$ (higher = better)"); ax[0].axhline(0, c="k", lw=.6)
    ax[1].bar(names, rm, color="#e76f51"); ax[1].set_title("Test RMSE, ppm (lower = better)")
    for a in ax: a.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(names, imp, path):
    o = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 4.5)); ax.barh(np.array(names)[o], np.array(imp)[o], color="#264653")
    ax.set_xlabel("Permutation importance (R$^2$ drop)"); ax.set_title("Feature importance (validation year)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_county_scatter(obs, pred, thr, path):
    fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(obs, pred, s=18, alpha=.7, c="#264653")
    lim = [min(obs.min(), pred.min()) * .95, max(obs.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1); ax.axvline(thr, c="gray", ls=":"); ax.axhline(thr, c="gray", ls=":")
    ax.set_xlabel("Observed county design value (ppm)"); ax.set_ylabel("Predicted (ppm)")
    ax.set_title("County design values (dotted = 0.070 ppm standard)"); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
