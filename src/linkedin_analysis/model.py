"""Models that predict `lift` (author-relative performance) from Jev features.

Three models, all fitted on train authors with the same author-grouped CV folds:
  controls  ridge on non-content controls only (media type, length, posting time)
  linear    lasso on Jev P(yes) features + controls
  rulefit   RuleFit: rules from shallow boosted trees + linear terms, then lasso

`lift` already removes each author's level, so raw P(yes) features can be
used directly, which keeps the models usable on a brand-new post.
Lasso inputs are standardized and the penalty is the one with the lowest
author-grouped CV error. (The sparser one-standard-error rule was tried first
and removed every term, including controls that clearly carry signal.)
Coefficients are reported back in raw units, so a coefficient is the change in
lift when P(yes) goes from 0 to 1.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV, Ridge

MEDIA = ["image", "images", "video", "document"]  # "text" is the baseline
HOUR_BUCKETS = {"night": (0, 5), "afternoon": (12, 17), "evening": (18, 23)}  # morning is the baseline


# Design matrix


def controls_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Non-content controls. Needs media, words, hour_pt, weekday."""
    out = pd.DataFrame(index=df.index)
    for m in MEDIA:
        out[f"ctl.media_{m}"] = (df["media"] == m).astype(float)
    out["ctl.log_words"] = np.log1p(df["words"].astype(float))
    out["ctl.weekend"] = (df["weekday"] >= 5).astype(float)
    for name, (lo, hi) in HOUR_BUCKETS.items():
        out[f"ctl.hour_{name}"] = df["hour_pt"].between(lo, hi).astype(float)
    return out


def word_count(text: str) -> int:
    return len(text.split())


# Folds and the one-standard-error rule


def cv_splits(folds: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in np.unique(folds)]


class ScaledLasso:
    """Lasso fitted on standardized inputs, exposing raw-unit coef_ and intercept_."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledLasso":
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        m = Lasso(alpha=self.alpha, max_iter=20000).fit((X - self.mean_) / self.std_, y)
        self.coef_ = m.coef_ / self.std_
        self.intercept_ = float(m.intercept_ - (self.coef_ * self.mean_).sum())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


def lasso_cv(X: np.ndarray, y: np.ndarray, folds: np.ndarray, rule: str = "min",
             standardize: bool = True) -> tuple["ScaledLasso | Lasso", dict]:
    """Choose the lasso penalty with author-grouped CV, then refit on all of X.

    rule="min" takes the lowest CV error; rule="1se" the sparsest model within
    one standard error of it.
    """
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    Xs = (X - mean) / std if standardize else X
    cv = LassoCV(cv=cv_splits(folds), alphas=40, max_iter=20000, n_jobs=-1).fit(Xs, y)
    mse = cv.mse_path_.mean(axis=1)
    se = cv.mse_path_.std(axis=1) / np.sqrt(cv.mse_path_.shape[1])
    best = int(mse.argmin())
    alpha_1se = float(cv.alphas_[mse <= mse[best] + se[best]].max())
    alpha = float(cv.alphas_[best]) if rule == "min" else alpha_1se
    model = ScaledLasso(alpha).fit(X, y) if standardize else Lasso(alpha=alpha, max_iter=20000).fit(X, y)
    return model, {"alpha": alpha, "rule": rule, "alpha_min": float(cv.alphas_[best]),
                   "alpha_1se": alpha_1se, "cv_mse": float(mse[best]), "nonzero": int((model.coef_ != 0).sum())}


lasso_1se = lasso_cv  # backwards-compatible name


# RuleFit


@dataclass
class Rule:
    conditions: list[tuple[str, str, float]]  # (column, "<=" or ">", threshold)

    def mask(self, X: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(X), dtype=bool)
        for col, op, t in self.conditions:
            m &= (X[col].to_numpy() <= t) if op == "<=" else (X[col].to_numpy() > t)
        return m

    def text(self) -> str:
        return " AND ".join(f"{c} {op} {t:.2f}" for c, op, t in self.conditions)


def extract_rules(gbm: GradientBoostingRegressor, columns: list[str]) -> list[Rule]:
    """Every non-root node of every tree becomes a rule (the path to it)."""
    rules, seen = [], set()
    for est in gbm.estimators_.ravel():
        tree = est.tree_

        def walk(node: int, path: list[tuple[str, str, float]]) -> None:
            if path:
                key = tuple(sorted((c, op, round(t, 3)) for c, op, t in path))
                if key not in seen:
                    seen.add(key)
                    rules.append(Rule(list(path)))
            if tree.children_left[node] == -1:
                return
            col, t = columns[tree.feature[node]], float(tree.threshold[node])
            walk(tree.children_left[node], path + [(col, "<=", t)])
            walk(tree.children_right[node], path + [(col, ">", t)])

        walk(0, [])
    return rules


@dataclass
class RuleFit:
    n_trees: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    min_support: float = 0.02
    rule: str = "min"
    random_state: int = 0
    rules_: list[Rule] = field(default_factory=list)
    linear_cols_: list[str] = field(default_factory=list)
    scale_: dict = field(default_factory=dict)
    lasso_: Lasso | None = None
    info_: dict = field(default_factory=dict)

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        rule_cols = [r.mask(X).astype(float) for r in self.rules_]
        lin = [(X[c].to_numpy() - self.scale_[c][0]) / self.scale_[c][1] for c in self.linear_cols_]
        return np.column_stack(rule_cols + lin)

    def fit(self, X: pd.DataFrame, y: np.ndarray, folds: np.ndarray, alpha: float | None = None) -> "RuleFit":
        """Fit rules and the lasso. With `alpha`, skip the CV search (used for CV refits)."""
        gbm = GradientBoostingRegressor(n_estimators=self.n_trees, max_depth=self.max_depth,
                                        learning_rate=self.learning_rate, subsample=0.5,
                                        random_state=self.random_state).fit(X, y)
        rules = extract_rules(gbm, list(X.columns))
        support = np.array([r.mask(X).mean() for r in rules])
        self.rules_ = [r for r, s in zip(rules, support) if self.min_support <= s <= 1 - self.min_support]
        # Linear terms scaled to a rule-like spread (Friedman & Popescu use 0.4 / std).
        self.linear_cols_ = list(X.columns)
        self.scale_ = {c: (float(X[c].mean()), float(X[c].std() / 0.4) or 1.0) for c in self.linear_cols_}
        D = self._design(X)
        if alpha is None:
            # Rules are 0/1 and linear terms are pre-scaled (Friedman & Popescu), so no standardizing.
            self.lasso_, self.info_ = lasso_cv(D, y, folds, rule=self.rule, standardize=False)
        else:
            self.lasso_, self.info_ = Lasso(alpha=alpha, max_iter=20000).fit(D, y), {"alpha": alpha}
        self.info_["n_candidate_rules"] = len(self.rules_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lasso_.predict(self._design(X))

    def terms(self, X: pd.DataFrame) -> pd.DataFrame:
        """Non-zero terms with importance |coef| x std of the term on X."""
        D = self._design(X)
        names = [r.text() for r in self.rules_] + [f"linear: {c}" for c in self.linear_cols_]
        kinds = ["rule"] * len(self.rules_) + ["linear"] * len(self.linear_cols_)
        coef = self.lasso_.coef_
        df = pd.DataFrame({"term": names, "kind": kinds, "coef": coef,
                           "support": D.mean(axis=0), "importance": np.abs(coef) * D.std(axis=0)})
        return df[df["coef"] != 0].sort_values("importance", ascending=False).reset_index(drop=True)


# Evaluation


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def pairwise_by_author(y: np.ndarray, pred: np.ndarray, authors: np.ndarray, min_gap: float = 0.1) -> pd.Series:
    """Per-author pairwise accuracy (see `evaluate`), indexed by author."""
    y, pred, authors = np.asarray(y), np.asarray(pred), np.asarray(authors)
    out = {}
    for a in np.unique(authors):
        i = np.where(authors == a)[0]
        dy = y[i][:, None] - y[i][None, :]
        dp = pred[i][:, None] - pred[i][None, :]
        mask = np.triu(np.abs(dy) >= min_gap, k=1)
        if mask.any():
            out[a] = float(((np.sign(dy) == np.sign(dp)) & (dp != 0))[mask].mean() + 0.5 * (dp == 0)[mask].mean())
    return pd.Series(out)


def bootstrap_ci(per_author: pd.Series, baseline: pd.Series | None = None, n: int = 2000, seed: int = 0) -> dict:
    """95% CI of the mean over authors, resampling authors. With `baseline`, CI of the paired difference."""
    s = per_author if baseline is None else (per_author - baseline.reindex(per_author.index))
    v = s.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return {"mean": round(float(v.mean()), 4), "ci95": [round(float(q), 4) for q in np.quantile(means, [0.025, 0.975])]}


def evaluate(y: np.ndarray, pred: np.ndarray, authors: np.ndarray, min_gap: float = 0.1) -> dict:
    """Metrics that compare posts only within the same author.

    pairwise_accuracy: for same-author pairs whose lift differs by >= min_gap,
      the share where the model ranks the better post higher (0.5 = chance).
      Averaged per author first, so every author counts equally.
    author_spearman: mean per-author rank correlation of prediction and lift.
    """
    y, pred, authors = np.asarray(y), np.asarray(pred), np.asarray(authors)
    accs, rhos = [], []
    for a in np.unique(authors):
        i = np.where(authors == a)[0]
        if len(i) < 2:
            continue
        dy = y[i][:, None] - y[i][None, :]
        dp = pred[i][:, None] - pred[i][None, :]
        mask = np.triu(np.abs(dy) >= min_gap, k=1)
        if mask.any():
            accs.append(float(((np.sign(dy) == np.sign(dp)) & (dp != 0))[mask].mean()
                              + 0.5 * (dp == 0)[mask].mean()))
        rhos.append(_spearman(y[i], pred[i]))
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {
        "pairwise_accuracy": round(float(np.mean(accs)), 4),
        "author_spearman": round(float(np.nanmean(rhos)), 4) if np.isfinite(rhos).any() else None,
        "spearman": round(_spearman(y, pred), 4),
        "r2": round(1 - ss_res / ss_tot, 4),
        "authors": int(len(np.unique(authors))),
        "posts": int(len(y)),
    }


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha).fit(X, y)
