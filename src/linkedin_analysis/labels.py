"""Author-relative performance labels.

Each post is compared with the same author's other posts, so the labels
measure how well the content did *for that author* rather than how big the
author's audience is. See docs/labels.md for the reasoning and checks.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    min_posts: int = 5
    """Drop authors with fewer posts; their baseline is too noisy."""

    window: int = 12
    """Baseline = median reactions of this many of the author's posts closest
    in time, excluding the post itself. Tracks slow drift in an author's reach."""

    smoothing: float = 2.0
    """Added to reactions and baseline before the log ratio, so 1 vs 3
    reactions on a tiny account doesn't count as a 3x win."""

    win_ratio: float = 2.0
    """`win` = post got at least this multiple of the author's baseline."""

    tier_share: float = 1 / 3
    """Share of each author's posts in the `top` and `bottom` tiers."""

    def to_dict(self) -> dict:
        return asdict(self)


def local_baseline(reactions: np.ndarray, window: int) -> np.ndarray:
    """Median of the `window` nearest *other* posts, for one author's posts in time order.

    Distance is position in the timeline; on equal distance the earlier post wins.
    When the author has `window` or fewer other posts, this is the plain
    leave-one-out median.
    """
    reactions = np.asarray(reactions, dtype=float)
    n = len(reactions)
    positions = np.arange(n)
    out = np.empty(n)
    for i in range(n):
        order = np.argsort(np.abs(positions - i), kind="stable")
        neighbors = order[order != i][:window]
        out[i] = np.median(reactions[neighbors]) if len(neighbors) else np.nan
    return out


def compute_labels(posts: pd.DataFrame, config: LabelConfig = LabelConfig()) -> pd.DataFrame:
    """Return one row per eligible post with author-relative labels.

    Required input columns: urn, author, posted, reactions. `band` is carried
    through when present.

    Output columns:
      n_posts    the author's post count
      baseline   local leave-one-out median reactions (see LabelConfig.window)
      lift       log((reactions + s) / (baseline + s)); 0 = typical for the author
      lift_pct   within-author percentile of lift, (rank - 0.5) / n, in (0, 1)
      win        lift >= log(win_ratio)
      tier       "top" / "middle" / "bottom" by lift_pct, for contrastive pairs
    """
    keep = ["urn", "author", "band", "posted", "reactions"]
    df = posts[[c for c in keep if c in posts.columns]].copy()
    if df["urn"].duplicated().any():
        raise ValueError("duplicate urn values in input")

    df["n_posts"] = df.groupby("author")["urn"].transform("size")
    df = df[df["n_posts"] >= config.min_posts]
    df = df.sort_values(["author", "posted", "urn"]).reset_index(drop=True)

    df["baseline"] = df.groupby("author", sort=False)["reactions"].transform(
        lambda s: local_baseline(s.to_numpy(), config.window)
    )

    s = config.smoothing
    df["lift"] = np.log((df["reactions"] + s) / (df["baseline"] + s))

    rank = df.groupby("author")["lift"].rank(method="average")
    df["lift_pct"] = (rank - 0.5) / df["n_posts"]

    df["win"] = df["lift"] >= np.log(config.win_ratio)

    q = config.tier_share
    df["tier"] = np.select(
        [df["lift_pct"] >= 1 - q, df["lift_pct"] < q], ["top", "bottom"], "middle"
    )
    return df


def _spearman(a: pd.Series, b: pd.Series) -> float:
    pair = pd.concat([a, b], axis=1).dropna()
    return round(float(pair.rank().corr().iloc[0, 1]), 3)


def summarize(labels: pd.DataFrame, posts: pd.DataFrame) -> dict:
    """Counts plus checks that the labels are free of author size and time drift.

    `posts` is the unfiltered input, used for the dropped counts and the raw
    band percentile (`pct`) that the labels are meant to replace.
    """
    author_median = labels.groupby("author")["reactions"].transform("median")
    timeline = labels.groupby("author")["posted"].rank(pct=True)
    raw_pct = labels["urn"].map(posts.set_index("urn")["pct"])
    size_bucket = pd.cut(
        author_median, [-np.inf, 3, 10, 30, np.inf], labels=["<=3", "4-10", "11-30", ">30"]
    )

    summary = {
        "posts_in": len(posts),
        "posts_labeled": len(labels),
        "authors_in": posts["author"].nunique(),
        "authors_labeled": labels["author"].nunique(),
        "win_rate": round(float(labels["win"].mean()), 3),
        "tiers": labels["tier"].value_counts().to_dict(),
        "lift_quantiles": labels["lift"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).to_dict(),
        "checks": {
            "spearman_raw_pct_vs_author_median": _spearman(raw_pct, author_median),
            "spearman_lift_vs_author_median": _spearman(labels["lift"], author_median),
            "spearman_lift_pct_vs_author_median": _spearman(labels["lift_pct"], author_median),
            "spearman_lift_vs_timeline_position": _spearman(labels["lift"], timeline),
            "win_rate_by_author_median": labels.groupby(size_bucket, observed=False)["win"]
            .mean()
            .round(3)
            .to_dict(),
        },
    }
    if "band" in labels.columns:
        summary["by_band"] = {
            band: {
                "posts": len(g),
                "authors": g["author"].nunique(),
                "win_rate": round(float(g["win"].mean()), 3),
            }
            for band, g in labels.groupby("band")
        }
    return summary
