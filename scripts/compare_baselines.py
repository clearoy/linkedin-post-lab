"""Compare the theme model with standard ML baselines on the same held-out authors.

Usage:
    uv run python scripts/compare_baselines.py \
        --train data/processed/judgments_full_train --test data/processed/judgments_full_test

Every model predicts the same target (`lift`) and is scored on the same test
authors: those Jev scored in the --test judgments folder. Tuning uses
the author-grouped CV folds from splits.parquet. Writes
data/processed/models/baselines.json.

Baselines:
  tfidf_ridge_subset   TF-IDF words + word pairs, ridge; same train authors as the theme model
  tfidf_ridge_all      the same, trained on every train author (needs no Jev, so it can use them all)
  jev_ridge            ridge on all single Jev questions
  jev_gbm              gradient-boosted trees on all single Jev questions (number of trees chosen by author CV)

Each baseline is also compared with the current theme model on the same test
authors: the paired difference in per-author pairwise accuracy, with a 95% CI
from resampling authors (diff_vs_current).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.model import (  # noqa: E402
    bootstrap_ci, controls_frame, cv_splits, evaluate, fit_ridge, gbm_cv, make_gbm, pairwise_by_author,
)
from linkedin_analysis.paths import PROCESSED  # noqa: E402

ALPHAS = np.logspace(-1, 4, 16)


def oof(make, X, y, folds) -> np.ndarray:
    pred = np.zeros(len(y))
    for tr, va in cv_splits(folds):
        pred[va] = make().fit(X[tr], y[tr]).predict(X[va])
    return pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=PROCESSED / "judgments_text_train")
    p.add_argument("--test", type=Path, default=PROCESSED / "judgments_text_test")
    args = p.parse_args()
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "author", "lift"]]
    splits = pd.read_parquet(PROCESSED / "splits.parquet")[["author", "split", "fold"]]
    posts = load_all()[["urn", "body"]]
    df = labels.merge(splits, on="author").merge(posts, on="urn")

    jtr = pd.read_parquet(args.train / "features.parquet")
    jte = pd.read_parquet(args.test / "features.parquet")
    q_cols = [c for c in jtr.columns if c != "urn"]
    sub_tr = df.merge(jtr, on="urn").reset_index(drop=True)
    test = df.merge(jte, on="urn").reset_index(drop=True)
    all_tr = df[df["split"] == "train"].reset_index(drop=True)
    y_te, a_te = test["lift"].to_numpy(), test["author"].to_numpy()

    def tfidf_ridge(train: pd.DataFrame):
        folds = train["fold"].to_numpy()
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True),
            RidgeCV(alphas=ALPHAS, cv=cv_splits(folds)),
        )
        model.fit(train["body"], train["lift"].to_numpy())
        alpha = model[-1].alpha_
        make = lambda: make_pipeline(  # noqa: E731
            TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True), Ridge(alpha=alpha))
        cv = oof(make, train["body"].to_numpy(), train["lift"].to_numpy(), folds)
        return cv, model.predict(test["body"]), {"alpha": float(alpha), "train_posts": len(train),
                                                   "train_authors": int(train["author"].nunique())}

    def on_questions(make_model):
        X, y, folds = sub_tr[q_cols].to_numpy(), sub_tr["lift"].to_numpy(), sub_tr["fold"].to_numpy()
        cv = oof(make_model, X, y, folds)
        return cv, make_model().fit(X, y).predict(test[q_cols].to_numpy()), {"train_posts": len(sub_tr)}

    def gbm():
        X, y, folds = sub_tr[q_cols].to_numpy(), sub_tr["lift"].to_numpy(), sub_tr["fold"].to_numpy()
        _, info = gbm_cv(X, y, folds, sub_tr["author"].to_numpy())
        cv = oof(lambda: make_gbm(info["n_iter"]), X, y, folds)
        return cv, make_gbm(info["n_iter"]).fit(X, y).predict(test[q_cols].to_numpy()), {
            "train_posts": len(sub_tr), "n_iter": info["n_iter"]}

    sub_folds = sub_tr["fold"].to_numpy()
    # Ridge strength for the Jev-question ridge, chosen once with the author folds.
    jev_alpha = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=cv_splits(sub_folds))).fit(
        sub_tr[q_cols].to_numpy(), sub_tr["lift"].to_numpy())[-1].alpha_
    results, preds = {}, {}
    if sub_tr["author"].nunique() == all_tr["author"].nunique():  # Jev scored every train author
        runs_skip = {"tfidf_ridge_all"}
    else:
        runs_skip = set()
    runs = {
        "tfidf_ridge_subset": lambda: tfidf_ridge(sub_tr),
        "tfidf_ridge_all": lambda: tfidf_ridge(all_tr),
        "jev_ridge": lambda: on_questions(lambda: make_pipeline(StandardScaler(), Ridge(alpha=jev_alpha))),
        "jev_gbm": lambda: gbm(),
    }
    for name, run in runs.items():
        if name in runs_skip:
            continue
        print(f"running {name} ...", flush=True)
        cv_pred, test_pred, info = run()
        train_frame = all_tr if name == "tfidf_ridge_all" else sub_tr
        results[name] = {"cv_train": evaluate(train_frame["lift"].to_numpy(), cv_pred, train_frame["author"].to_numpy()),
                         "test": evaluate(y_te, test_pred, a_te), **info}
        preds[name] = test_pred

    # Current theme model (from the saved bundle) and the format reference, on exactly the same test posts.
    m = json.loads((PROCESSED / "models" / "metrics.json").read_text())
    bundle = json.loads((PROCESSED / "models" / "model_bundle.json").read_text())
    current = bundle["linear"]["intercept"] + sum(
        c * test[bundle["themes"][t]["members"]].mean(axis=1).to_numpy() for t, c in bundle["linear"]["coef"].items())
    meta = load_all()[["urn", "media", "words", "hour_pt", "weekday"]]
    ctl_tr, ctl_te = (controls_frame(f[["urn"]].merge(meta, on="urn")) for f in (sub_tr, test))
    preds["themes_linear (current)"] = current
    preds["format_reference"] = fit_ridge(ctl_tr.to_numpy(), sub_tr["lift"].to_numpy()).predict(ctl_te.to_numpy())
    results["themes_linear (current)"] = {"cv_train": m["linear"]["cv_train"], "test": evaluate(y_te, current, a_te)}
    results["format_reference"] = {"cv_train": m["controls"]["cv_train"],
                                   "test": evaluate(y_te, preds["format_reference"], a_te)}

    per_author = {name: pairwise_by_author(y_te, p, a_te) for name, p in preds.items()}
    for name in preds:
        results[name]["test"]["pairwise_accuracy_ci95"] = bootstrap_ci(per_author[name])["ci95"]
        if name != "themes_linear (current)":  # paired: this model minus the current model, per author
            results[name]["test"]["diff_vs_current"] = bootstrap_ci(per_author[name], per_author["themes_linear (current)"])

    out = PROCESSED / "models" / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{'model':28s} {'CV':>6s} {'test':>6s}  test 95% CI")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["test"]["pairwise_accuracy"]):
        ci = r["test"].get("pairwise_accuracy_ci95", ["", ""])
        print(f"{name:28s} {r['cv_train']['pairwise_accuracy']:6.3f} {r['test']['pairwise_accuracy']:6.3f}  {ci}")
    print(f"\nTest set: {len(test)} posts from {test['author'].nunique()} authors. Wrote {out}")


if __name__ == "__main__":
    main()
