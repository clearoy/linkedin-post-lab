"""Scorecard vs standard ML baselines, and blends of them, on the same held-out authors.

Usage:
    uv run python research/blend.py

Every model is fitted on the 667 train authors and scored once on the 165 test authors (the same
3,032 posts). Blends are chosen on train only:
  equal    each model's prediction divided by the spread of its train out-of-fold prediction, then averaged
  stacked  weights from a within-author regression of lift on the models' train out-of-fold predictions

Writes research/results/blend.json (statistics only).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from joint import frame  # noqa: E402
from scorecard import DIRS, bounded_fit, length_col, oof_predictions  # noqa: E402
from stats import PROCESSED, RESULTS, demean, within_ols  # noqa: E402

from linkedin_analysis.model import bootstrap_ci, cv_splits, gbm_cv, make_gbm, pairwise_by_author  # noqa: E402

HERE = Path(__file__).resolve().parent
ALPHAS = np.logspace(-1, 4, 16)


def oof(make, X, y, folds) -> np.ndarray:
    pred = np.zeros(len(y))
    for a, b in cv_splits(folds):
        pred[b] = make().fit(X[a], y[a]).predict(X[b])
    return pred


def main() -> None:
    bundle = json.loads((PROCESSED / "research" / "scorecard.json").read_text())
    spec = json.loads((HERE / "drivers.json").read_text())
    sign_of = {m[0]: (1.0 if v["direction"] == "+" else -1.0) for v in spec["drivers"].values() for m in v["members"]}
    card = bundle["sizes"][bundle["default"]]
    names = [i["id"] for i in card["items"]]
    cap, lam = bundle["common"]["length_cap"], bundle["common"]["ridge"]

    df = frame(DIRS)
    old147 = [c for c in pd.read_parquet(PROCESSED / "judgments_full_train" / "features.parquet").columns
              if c != "urn" and c in df.columns]
    df = df[df[names + old147].notna().all(axis=1)].reset_index(drop=True)
    jev_all = [c for c in df.columns if (c in old147 or c.startswith("r1_") or c.startswith("r2_")) and df[c].notna().all()]
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    y, au, folds = tr["lift"].to_numpy(), tr["author"].to_numpy(), tr["fold"].to_numpy()
    yt, at = te["lift"].to_numpy(), te["author"].to_numpy()
    print(f"train {tr['author'].nunique()} authors / {len(tr)} posts; test {te['author'].nunique()} / {len(te)}; "
          f"{len(old147)} old questions, {len(jev_all)} Jev questions in all", flush=True)

    cv, test = {}, {}

    # scorecard (full size), refitted exactly as research/scorecard.py does
    X = np.column_stack([tr[names].to_numpy(float), length_col(tr["words"], cap)])
    Xt = np.column_stack([te[names].to_numpy(float), length_col(te["words"], cap)])
    s = np.r_[[sign_of[n] for n in names], 1.0]
    o = oof_predictions(X, y, au, folds, s, lam)
    shrink = float(within_ols(y, o, au)["beta"][0])
    beta = bounded_fit(demean(X, au), demean(y, au), s, lam) * shrink
    cv["Scorecard (64 questions)"], test["Scorecard (64 questions)"] = o * shrink, (Xt - X.mean(axis=0)) @ beta

    def ridge_on(cols):
        A, At = tr[cols].to_numpy(float), te[cols].to_numpy(float)
        alpha = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=cv_splits(folds))).fit(A, y)[-1].alpha_
        make = lambda: make_pipeline(StandardScaler(), Ridge(alpha=alpha))  # noqa: E731
        return oof(make, A, y, folds), make().fit(A, y).predict(At)

    def gbm_on(cols):
        A, At = tr[cols].to_numpy(float), te[cols].to_numpy(float)
        _, info = gbm_cv(A, y, folds, au)
        make = lambda: make_gbm(info["n_iter"])  # noqa: E731
        return oof(make, A, y, folds), make().fit(A, y).predict(At)

    def tfidf():
        vec = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)  # noqa: E731
        alpha = make_pipeline(vec(), RidgeCV(alphas=ALPHAS, cv=cv_splits(folds))).fit(tr["body"], y)[-1].alpha_
        make = lambda: make_pipeline(vec(), Ridge(alpha=alpha))  # noqa: E731
        return oof(make, tr["body"].to_numpy(), y, folds), make().fit(tr["body"], y).predict(te["body"])

    for name, run in [("Ridge, 147 old questions", lambda: ridge_on(old147)),
                      (f"Ridge, all {len(jev_all)} Jev questions", lambda: ridge_on(jev_all)),
                      ("Gradient boosting, 147 old questions", lambda: gbm_on(old147)),
                      (f"Gradient boosting, all {len(jev_all)} Jev questions", lambda: gbm_on(jev_all)),
                      ("TF-IDF words + ridge", tfidf)]:
        print(f"fitting {name} ...", flush=True)
        cv[name], test[name] = run()

    sc, gb, gb_all, tf, rd = ("Scorecard (64 questions)", "Gradient boosting, 147 old questions",
                              f"Gradient boosting, all {len(jev_all)} Jev questions", "TF-IDF words + ridge",
                              f"Ridge, all {len(jev_all)} Jev questions")
    blends = {"Scorecard + gradient boosting": [sc, gb_all], "Scorecard + TF-IDF": [sc, tf],
              "Scorecard + gradient boosting + TF-IDF": [sc, gb_all, tf], "Scorecard + ridge": [sc, rd],
              "All six models": list(cv)}
    weights = {}
    for bname, parts in blends.items():
        sd = {p: demean(cv[p], au).std() for p in parts}
        cv[f"{bname} (equal)"] = sum(cv[p] / sd[p] for p in parts) / len(parts)
        test[f"{bname} (equal)"] = sum(test[p] / sd[p] for p in parts) / len(parts)
        w = within_ols(y, np.column_stack([cv[p] for p in parts]), au)["beta"]
        weights[f"{bname} (stacked)"] = {p: round(float(v), 3) for p, v in zip(parts, w)}
        cv[f"{bname} (stacked)"] = np.column_stack([cv[p] for p in parts]) @ w
        test[f"{bname} (stacked)"] = np.column_stack([test[p] for p in parts]) @ w

    per = {k: pairwise_by_author(yt, v, at) for k, v in test.items()}
    cvper = {k: pairwise_by_author(y, v, au) for k, v in cv.items()}
    out = {}
    print(f"\n{'model':58s} {'CV':>6s} {'test':>6s}  {'test 95% CI':16s} vs scorecard (paired 95% CI)")
    for k in sorted(per, key=lambda k: -per[k].mean()):
        d = bootstrap_ci(per[k], per[sc]) if k != sc else None
        out[k] = {"cv_pairwise": round(float(cvper[k].mean()), 4), "test_pairwise": round(float(per[k].mean()), 4),
                  "test_ci95": bootstrap_ci(per[k])["ci95"], "vs_scorecard": d, "weights": weights.get(k)}
        dd = "—" if d is None else f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
        print(f"{k:58s} {cvper[k].mean():6.4f} {per[k].mean():6.4f}  {str(out[k]['test_ci95']):16s} {dd}")
    for k, w in weights.items():
        print(f"  {k}: {w}")
    (RESULTS / "blend.json").write_text(json.dumps(out, indent=1))
    print(f"\nWrote {RESULTS / 'blend.json'}")


if __name__ == "__main__":
    main()
