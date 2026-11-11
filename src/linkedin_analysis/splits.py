"""Author-level train/test split with cross-validation folds.

Every author's posts land on one side only. Authors are stratified by band,
post count, and typical reactions so the test set looks like the training set.
"""

import numpy as np
import pandas as pd


def make_splits(labels: pd.DataFrame, test_share: float = 0.2, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """One row per author: band, n_posts, size_q, stratum, split, fold.

    `split` is "train" or "test"; `fold` is 0..n_folds-1 for train authors and
    -1 for test authors. Needs labels columns: author, band, reactions.
    """
    authors = (
        labels.groupby("author")
        .agg(band=("band", "first"), n_posts=("reactions", "size"), median_reactions=("reactions", "median"))
        .reset_index()
        .sort_values("author")
    )
    authors["posts_bucket"] = np.where(authors["n_posts"] >= 15, "15+", "5-14")
    # Reaction quartile within band, so each band keeps its own spread of account sizes.
    authors["size_q"] = authors.groupby("band")["median_reactions"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=False)
    ).astype(int)
    authors["stratum"] = authors["band"] + "|" + authors["posts_bucket"] + "|q" + authors["size_q"].astype(str)

    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=authors.index)
    fold = pd.Series(-1, index=authors.index)
    for _, group in authors.groupby("stratum"):
        idx = rng.permutation(group.index.to_numpy())
        n_test = int(round(len(idx) * test_share))
        split[idx[:n_test]] = "test"
        train_idx = idx[n_test:]
        fold[train_idx] = (np.arange(len(train_idx)) + rng.integers(n_folds)) % n_folds

    authors["split"] = split
    authors["fold"] = fold.astype(int)
    return authors.drop(columns=["posts_bucket"]).reset_index(drop=True)


def summarize_splits(splits: pd.DataFrame, labels: pd.DataFrame) -> dict:
    posts = labels.merge(splits[["author", "split", "fold"]], on="author")
    by = lambda col: posts.groupby(col).agg(authors=("author", "nunique"), posts=("urn", "size"),  # noqa: E731
                                            mean_lift=("lift", "mean"), win_rate=("win", "mean")).round(3)
    return {
        "by_split": by("split").to_dict("index"),
        "by_fold": {int(k): v for k, v in by("fold").to_dict("index").items()},
        "by_split_band": posts.groupby(["split", "band"])["author"].nunique().unstack().to_dict("index"),
    }
