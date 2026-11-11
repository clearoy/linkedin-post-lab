"""Learning curve: how test accuracy grows with the number of training authors.

Usage:
    uv run python scripts/learning_curve.py \
        --judgments data/processed/judgments_full_train data/processed/judgments_full_test

For each training size, whole train authors are sampled at random (several
repeats; the full size runs once) and every model is fit from scratch on that
sample: themes are re-clustered, k and ridge strengths re-chosen with the
author folds. Every run is scored on the same test posts (all scored test
authors), so the sizes are directly comparable.

Models:
  themes_linear   15 co-occurrence themes (>= 3 questions) -> relaxed lasso, <= 10 themes (the dashboard model)
  jev_ridge       ridge on all single Jev questions
  jev_gbm         gradient-boosted trees on all single Jev questions (number of trees chosen by author CV)
  tfidf_ridge     TF-IDF words + word pairs, ridge (no Jev needed)

Writes data/processed/models/learning_curve.json (every run plus a summary).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.model import (  # noqa: E402
    ATTACHMENT_FEATURES, cv_splits, evaluate, gbm_cv, make_themes, relaxed_lasso_cv, theme_scores,
)
from linkedin_analysis.paths import PROCESSED  # noqa: E402

ALPHAS = np.logspace(-1, 4, 11)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judgments", type=Path, nargs="+", required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[40, 80, 160, 320, 0], help="train authors; 0 = all")
    p.add_argument("--repeats", type=int, default=5, help="random author samples per size (the full size runs once)")
    p.add_argument("--themes", type=int, default=15)
    p.add_argument("--min-theme-size", type=int, default=3)
    p.add_argument("--min-prevalence", type=float, default=0.02)
    p.add_argument("--out", type=Path, default=PROCESSED / "models" / "learning_curve.json")
    return p.parse_args()


def themes_linear(tr: pd.DataFrame, te: pd.DataFrame, q_cols: list[str], n_themes: int, min_size: int):
    themes = make_themes(tr[q_cols], n_themes)
    themes = {t: v for t, v in themes.items() if len(v["members"]) >= min_size}
    S_tr, S_te = theme_scores(tr[q_cols], themes).to_numpy(), theme_scores(te[q_cols], themes).to_numpy()
    none = np.zeros((len(tr), 0))
    model, info = relaxed_lasso_cv(S_tr, none, tr["lift"].to_numpy(), tr["fold"].to_numpy(), tr["author"].to_numpy())
    return model.predict(S_te, np.zeros((len(te), 0))), {"themes": len(themes), "k": info["k"]}


def jev_ridge(tr, te, q_cols, *_):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=cv_splits(tr["fold"].to_numpy())))
    m.fit(tr[q_cols].to_numpy(), tr["lift"].to_numpy())
    return m.predict(te[q_cols].to_numpy()), {"alpha": float(m[-1].alpha_)}


def jev_gbm(tr, te, q_cols, *_):
    m, info = gbm_cv(tr[q_cols].to_numpy(), tr["lift"].to_numpy(), tr["fold"].to_numpy(), tr["author"].to_numpy())
    return m.predict(te[q_cols].to_numpy()), {"n_iter": info["n_iter"]}


def tfidf_ridge(tr, te, *_):
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True),
                      RidgeCV(alphas=ALPHAS, cv=cv_splits(tr["fold"].to_numpy())))
    m.fit(tr["body"], tr["lift"].to_numpy())
    return m.predict(te["body"]), {"alpha": float(m[-1].alpha_)}


MODELS = {"themes_linear": themes_linear, "jev_ridge": jev_ridge, "jev_gbm": jev_gbm, "tfidf_ridge": tfidf_ridge}


def main() -> None:
    args = parse_args()
    wide = pd.concat([pd.read_parquet(d / "features.parquet") for d in args.judgments]).drop_duplicates("urn")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "author", "lift"]]
    splits = pd.read_parquet(PROCESSED / "splits.parquet")[["author", "split", "fold"]]
    posts = load_all()[["urn", "body"]]
    df = wide.merge(labels, on="urn").merge(splits, on="author").merge(posts, on="urn").reset_index(drop=True)

    train = df["split"] == "train"
    q_all = [c for c in wide.columns if c != "urn" and c not in ATTACHMENT_FEATURES]
    prev = df.loc[train, q_all].mean()
    q_cols = [c for c in q_all if args.min_prevalence <= prev[c] <= 1 - args.min_prevalence]
    df[q_cols] = df[q_cols].fillna(df.loc[train, q_cols].mean())
    tr_all, te = df[train].reset_index(drop=True), df[~train].reset_index(drop=True)
    y_te, a_te = te["lift"].to_numpy(), te["author"].to_numpy()
    authors = sorted(tr_all["author"].unique())
    print(f"train pool {len(authors)} authors / {len(tr_all)} posts; test {te['author'].nunique()} authors / "
          f"{len(te)} posts; {len(q_cols)} questions", flush=True)

    runs = []
    for size in args.sizes:
        n = size or len(authors)
        reps = 1 if n >= len(authors) else args.repeats
        for rep in range(reps):
            pick = set(random.Random(1000 * n + rep).sample(authors, n)) if n < len(authors) else set(authors)
            tr = tr_all[tr_all["author"].isin(pick)].reset_index(drop=True)
            for name, fit in MODELS.items():
                t0 = time.time()
                pred, info = fit(tr, te, q_cols, args.themes, args.min_theme_size)
                r = {"train_authors": n, "train_posts": len(tr), "repeat": rep, "model": name,
                     "test_pairwise": evaluate(y_te, pred, a_te)["pairwise_accuracy"],
                     "seconds": round(time.time() - t0, 1), **info}
                runs.append(r)
                print(f"  n={n:4d} rep={rep} {name:14s} test={r['test_pairwise']:.4f} ({r['seconds']}s)", flush=True)

    res = pd.DataFrame(runs)
    summary = (res.groupby(["model", "train_authors"])["test_pairwise"]
               .agg(mean="mean", sd="std", runs="count").reset_index())
    summary["sd"] = summary["sd"].fillna(0.0)
    posts_by_n = res.groupby("train_authors")["train_posts"].mean().round().astype(int).to_dict()
    out = {"test_authors": int(te["author"].nunique()), "test_posts": len(te), "questions": len(q_cols),
           "train_posts_by_authors": {str(k): v for k, v in posts_by_n.items()},
           "summary": summary.round(4).to_dict("records"), "runs": runs}
    args.out.write_text(json.dumps(out, indent=2))
    print("\n" + summary.pivot(index="train_authors", columns="model", values="mean").round(3).to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
