"""Model zoo: baselines, linear, SVR (properly tuned epsilon), MLP, gradient boosting."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def _prep():
    return [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]


def _scaled_y(reg):
    # Scaling y matters: SVR's epsilon is in target units. Ozone is ~0.05 ppm, so the
    # default epsilon=0.1 swallows every residual (the paper's SVR bug).
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())


def make_regressors(fast=False, seed=42):
    return {
        "ridge": Pipeline(_prep() + [("m", Ridge(alpha=1.0))]),
        "mlp": _scaled_y(Pipeline(_prep() + [("m", MLPRegressor(
            hidden_layer_sizes=(128, 128, 64), learning_rate="adaptive", learning_rate_init=1e-3,
            early_stopping=True, n_iter_no_change=8, max_iter=40 if fast else 200,
            random_state=seed))])),
        "hgb": HistGradientBoostingRegressor(
            max_iter=120 if fast else 400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed),
    }


def tune_svr(Xtr, ytr, Xva, yva, n_sub=15000, fast=False, seed=42):
    """Small grid over C/epsilon on a subsample (SVR is O(n^2)). Returns (fitted_model, params, grid)."""
    rng = np.random.default_rng(seed)
    n_sub = 3000 if fast else n_sub
    idx = rng.choice(len(Xtr), min(n_sub, len(Xtr)), replace=False)
    vidx = rng.choice(len(Xva), min(5000 if not fast else 1000, len(Xva)), replace=False)
    grid, best = [], (-np.inf, None, None)
    for C in ([1.0] if fast else [1.0, 10.0]):
        for eps in ([0.1] if fast else [0.02, 0.1, 0.5]):   # in std-units of y
            m = _scaled_y(Pipeline(_prep() + [("m", SVR(kernel="rbf", C=C, epsilon=eps))]))
            m.fit(Xtr.iloc[idx], ytr.iloc[idx])
            r2 = r2_score(yva.iloc[vidx], m.predict(Xva.iloc[vidx]))
            grid.append({"C": C, "epsilon": eps, "val_r2": float(r2)})
            if r2 > best[0]:
                best = (r2, m, {"C": C, "epsilon": eps})
    return best[1], best[2], grid


def make_classifier(seed=42, fast=False):
    # Unweighted on purpose: we sum predicted probabilities into expected exceedance counts,
    # which needs calibrated probabilities.
    return HistGradientBoostingClassifier(max_iter=120 if fast else 300, learning_rate=0.06,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=seed)


class Persistence:
    """Baseline: tomorrow = today (falls back to the training mean)."""
    def fit(self, X, y):
        self.mean_ = float(y.mean()); return self
    def predict(self, X):
        return X["ozone_lag1"].fillna(self.mean_).values


class SiteMonthClimatology:
    """Baseline: mean ozone per (latitude, longitude, month) seen in training."""
    def fit(self, X, y):
        self.table_ = y.groupby(self._k(X).values).mean(); self.mean_ = float(y.mean())
        return self
    @staticmethod
    def _k(X):
        return X["latitude"].round(3).astype(str) + "|" + X["longitude"].round(3).astype(str) + "|" + \
               np.round(np.arctan2(X["month_sin"], X["month_cos"]) * 6 / np.pi).astype(int).mod(12).astype(str)
    def predict(self, X):
        return self._k(X).map(self.table_).fillna(self.mean_).values
