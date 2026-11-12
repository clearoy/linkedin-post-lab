"""Shared data loading and statistics for the auto-research scripts.

Every effect here compares posts **by the same author**: `lift` is already
author-relative, and on top of that x and y are demeaned within author, so an
author who simply posts more of some kind of content and is popular anyway
cannot create an effect.

Main numbers per property x (a yes/no content property):
  multiplier   exp(beta), where beta is the within-author OLS slope of lift on x.
               lift = log((reactions+2)/(author baseline+2)), so 1.15 means
               about 15% more reactions than the same author's usual post.
  p            two-sided, from author-clustered standard errors (CR1).
  win_rate     for a random post with x and a random post without x from the
               same author, the chance the one with x got the higher lift
               (ties count half). Averaged over authors; 0.5 = no difference.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # see README "Imports on this Mac"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.model import ATTACHMENT_FEATURES  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402

RESEARCH_DATA = PROCESSED / "research"  # gitignored, like all data
RESULTS = Path(__file__).resolve().parent / "results"
EXISTING = [PROCESSED / "judgments_full_train", PROCESSED / "judgments_full_test"]
YES = 0.5  # P(yes) at or above this counts as "the post has it"


def load_posts() -> pd.DataFrame:
    """Every labeled post with lift, split, fold, band, media type and text."""
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "author", "band", "lift", "reactions", "baseline"]]
    splits = pd.read_parquet(PROCESSED / "splits.parquet")[["author", "split", "fold"]]
    posts = load_all()[["urn", "body", "media", "words"]]
    return labels.merge(splits, on="author").merge(posts, on="urn").reset_index(drop=True)


def load_judgments(dirs) -> pd.DataFrame:
    """Wide P(yes) table (urn + one column per question) from judgment folders.

    Folders may hold different questions for the same posts, or the same questions for
    different posts; answers are combined per urn and question (first folder wins on overlap)."""
    wide = None
    for d in dirs:
        w = pd.read_parquet(Path(d) / "features.parquet").drop_duplicates("urn").set_index("urn")
        wide = w if wide is None else wide.combine_first(w)
    wide = wide.reset_index()
    return wide[[c for c in wide.columns if c not in ATTACHMENT_FEATURES]]


# ---------------------------------------------------------------- text checks (free, no LLM)

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
URL = re.compile(r"https?://|www\.|lnkd\.in", re.I)
HASHTAG = re.compile(r"(?:^|\s)#\w+")
BULLET = re.compile(r"^\s*(?:[-•▪►✅✔→➡️👉]|\d+[.)])\s+", re.M)


def text_checks(body: pd.Series) -> pd.DataFrame:
    """Simple yes/no checks a text search can answer. Named `text.*`."""
    b = body.fillna("")
    first = b.str.strip().str.split("\n").str[0].fillna("")
    words = b.str.split().str.len().fillna(0)
    paras = b.str.split(r"\n\s*\n").apply(lambda ps: [p for p in ps if p.strip()])
    words_per_para = words / paras.str.len().clip(lower=1)
    return pd.DataFrame({
        "text.under_80_words": words < 80,
        "text.over_250_words": words > 250,
        "text.has_link": b.str.contains(URL),
        "text.has_hashtag": b.str.contains(HASHTAG),
        "text.three_plus_hashtags": b.str.count(HASHTAG) >= 3,
        "text.has_emoji": b.str.contains(EMOJI),
        "text.three_plus_emoji": b.str.count(EMOJI) >= 3,
        "text.has_bullet_or_numbered_list": b.str.count(BULLET) >= 2,
        "text.first_line_under_60_chars": first.str.len() < 60,
        "text.first_line_is_question": first.str.strip().str.endswith("?"),
        "text.first_line_has_number": first.str.contains(r"\d"),
        "text.ends_with_question": b.str.strip().str.endswith("?"),
        "text.has_exclamation": b.str.contains("!"),
        "text.short_paragraphs": (words_per_para <= 20) & (paras.str.len() >= 3),
        "text.single_block": paras.str.len() <= 1,
    }).astype(float)


# ---------------------------------------------------------------- statistics

def demean(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean (works on 1-D or 2-D arrays)."""
    a = np.asarray(a, dtype=float)
    _, inv = np.unique(groups, return_inverse=True)
    if a.ndim == 1:
        return a - (np.bincount(inv, a) / np.bincount(inv))[inv]
    sums = np.zeros((inv.max() + 1, a.shape[1]))
    np.add.at(sums, inv, a)
    return a - (sums / np.bincount(inv)[:, None])[inv]


def within_ols(y, X, authors, groups=None) -> dict:
    """Within-group OLS of y on the columns of X with author-clustered CR1 standard errors.

    groups: fixed-effect groups (default: author). Use author x media to compare
    posts with the same attachment type only. Returns beta, se, p per column.
    """
    groups = authors if groups is None else groups
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    yd, Xd = demean(y, groups), demean(X, groups)
    keep = Xd.std(axis=0) > 0
    Xd = Xd[:, keep]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ yd
    e = yd - Xd @ beta
    _, ca = np.unique(authors, return_inverse=True)
    score = np.zeros((ca.max() + 1, Xd.shape[1]))
    np.add.at(score, ca, Xd * e[:, None])
    G, N, K = ca.max() + 1, len(y), Xd.shape[1] + len(np.unique(groups))
    V = XtX_inv @ (score.T @ score) @ XtX_inv * G / (G - 1) * (N - 1) / max(N - K, 1)
    se = np.sqrt(np.diag(V))
    out_b, out_se = np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    out_b[keep], out_se[keep] = beta, se
    p = 2 * st.t.sf(np.abs(out_b / out_se), df=G - 1)
    return {"beta": out_b, "se": out_se, "p": p}


def win_rate(y, x, authors) -> tuple[float, int]:
    """Mean over authors of P(lift of a post with x > lift of a post without x); ties count half.
    Returns (mean, number of authors with both kinds of post)."""
    df = pd.DataFrame({"y": y, "x": np.asarray(x) >= YES, "a": authors})
    df["r"] = df.groupby("a")["y"].rank()
    g = df.groupby(["a", "x"])["r"].agg(["sum", "count"]).unstack("x")
    g = g.dropna()
    if g.empty:
        return np.nan, 0
    n1, n0 = g[("count", True)], g[("count", False)]
    auc = (g[("sum", True)] - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(auc.mean()), int(len(g))


def effect(y, x, authors, groups=None, covars=None) -> dict:
    """One property's within-author effect (x is P(yes); counted as yes at >= 0.5)."""
    xb = (np.asarray(x) >= YES).astype(float)
    X = xb[:, None] if covars is None else np.column_stack([xb, covars])
    r = within_ols(y, X, authors, groups)
    b, se = r["beta"][0], r["se"][0]
    wr, n_auth = win_rate(y, xb, authors)
    return {"share": float(xb.mean()), "beta": float(b), "se": float(se), "p": float(r["p"][0]),
            "multiplier": float(np.exp(b)), "lo": float(np.exp(b - 1.96 * se)), "hi": float(np.exp(b + 1.96 * se)),
            "win_rate": wr, "authors_with_both": n_auth}


def bh(p) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def fmt_x(m: float) -> str:
    return f"{m:.2f}×"
