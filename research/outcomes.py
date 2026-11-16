"""Beyond the average number of reactions: comments, reposts, big hits and flops.

Usage:
    uv run python research/outcomes.py

For every content property, the same within-author test as effects.py
(discover on train authors with BH q < 0.05, confirm on held-out test authors)
is run on five more outcomes:

  comments   log((comments + 1) / (author's local baseline + 1)), same baseline rule as reactions
  reposts    the same for reposts
  talk       log((comments + 1) / (reactions + 2)): comments per reaction, "does it start a conversation"
  hit        the post got at least 2x the author's usual reactions (percentage points)
  flop       the post got at most half the author's usual reactions (percentage points)

Writes research/results/outcomes.csv (one row per property and outcome) and
research/results/outcomes_summary.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compare_dashboard import DIRS  # noqa: E402
from joint import frame  # noqa: E402
from stats import RESULTS, bh, effect  # noqa: E402

from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.labels import local_baseline  # noqa: E402

LOG_OUTCOMES = ["lift", "comments", "reposts", "talk"]  # effects are multipliers
BINARY_OUTCOMES = ["hit", "flop"]                       # effects are percentage points
WINDOW = 12  # same as the reactions label


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Add the extra outcome columns to a frame of labeled posts."""
    raw = load_all()[["urn", "comments", "reposts", "posted"]].rename(columns={"comments": "n_comments", "reposts": "n_reposts"})
    df = df.merge(raw, on="urn").sort_values(["author", "posted", "urn"]).reset_index(drop=True)
    for name, col in [("comments", "n_comments"), ("reposts", "n_reposts")]:
        base = df.groupby("author", sort=False)[col].transform(lambda s: local_baseline(s.to_numpy(), WINDOW))
        df[name] = np.log((df[col] + 1) / (base + 1))
    df["talk"] = np.log((df["n_comments"] + 1) / (df["reactions"] + 2))
    df["hit"] = (df["lift"] >= np.log(2)).astype(float)
    df["flop"] = (df["lift"] <= -np.log(2)).astype(float)
    return df


def question_text() -> dict[str, str]:
    q = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id")
    return q.set_index("id")["question"].to_dict()


def main() -> None:
    df = add_outcomes(frame(DIRS))
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id")
    cols = [c for c in eff["id"] if c in df.columns]
    qtext = question_text()
    print(f"{len(df)} posts, {len(cols)} properties; hit rate {df['hit'].mean():.3f}, flop rate {df['flop'].mean():.3f}", flush=True)

    rows = []
    for c in cols:
        d = df[df[c].notna()]
        tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
        for o in LOG_OUTCOMES + BINARY_OUTCOMES:
            a, b, al = (effect(g[o].to_numpy(), g[c].to_numpy(), g["author"].to_numpy()) for g in (tr, te, d))
            one_sided = b["p"] / 2 if np.sign(b["beta"]) == np.sign(a["beta"]) else 1 - b["p"] / 2
            rows.append({"id": c, "question": qtext.get(c, c), "outcome": o, "share": al["share"],
                         "beta": al["beta"], "se": al["se"], "p": al["p"], "win_rate": al["win_rate"],
                         "train_beta": a["beta"], "train_p": a["p"], "test_beta": b["beta"], "test_p_one_sided": one_sided})
    res = pd.DataFrame(rows)
    res["train_q"] = res.groupby("outcome")["train_p"].transform(bh)
    res["passed"] = res["train_q"] < 0.05
    res["test_q"] = np.nan
    for o, g in res[res["passed"]].groupby("outcome"):
        res.loc[g.index, "test_q"] = bh(g["test_p_one_sided"])
    res["confirmed"] = res["passed"] & (res["test_q"] < 0.05)
    log = res["outcome"].isin(LOG_OUTCOMES)
    res["mult"] = np.where(log, np.exp(res["beta"]), np.nan)
    res["lo"] = np.where(log, np.exp(res["beta"] - 1.96 * res["se"]), np.nan)
    res["hi"] = np.where(log, np.exp(res["beta"] + 1.96 * res["se"]), np.nan)
    res["test_mult"] = np.where(log, np.exp(res["test_beta"]), np.nan)
    res["points"] = np.where(~log, 100 * res["beta"], np.nan)  # percentage points for hit / flop
    res["points_lo"] = np.where(~log, 100 * (res["beta"] - 1.96 * res["se"]), np.nan)
    res["points_hi"] = np.where(~log, 100 * (res["beta"] + 1.96 * res["se"]), np.nan)
    res.round(4).to_csv(RESULTS / "outcomes.csv", index=False)

    wide = res.pivot(index="id", columns="outcome", values="beta")
    conf = res.pivot(index="id", columns="outcome", values="confirmed")
    summary = {"posts": len(df), "authors": int(df["author"].nunique()),
               "hit_rate": float(df["hit"].mean()), "flop_rate": float(df["flop"].mean()),
               "zero_comment_share": float((df["n_comments"] == 0).mean()),
               "zero_repost_share": float((df["n_reposts"] == 0).mean()),
               "confirmed_by_outcome": conf.sum().astype(int).to_dict(),
               "corr_of_effects_with_reactions": wide.corr()["lift"].round(3).to_dict()}
    (RESULTS / "outcomes_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    def show(title, ids):
        print(f"\n{title}")
        for i in ids:
            r = res[res["id"] == i].set_index("outcome")
            star = lambda o: "*" if r.loc[o, "confirmed"] else " "  # noqa: E731
            print(f"  react {r.loc['lift', 'mult']:.2f}{star('lift')} comm {r.loc['comments', 'mult']:.2f}{star('comments')} "
                  f"repost {r.loc['reposts', 'mult']:.2f}{star('reposts')} talk {r.loc['talk', 'mult']:.2f}{star('talk')} "
                  f"hit {r.loc['hit', 'points']:+.1f}{star('hit')} flop {r.loc['flop', 'points']:+.1f}{star('flop')}  {i}")

    both = conf.index[conf["talk"]]
    show("more comments per reaction (confirmed), top 15", wide.loc[both].sort_values("talk", ascending=False).head(15).index)
    show("fewer comments per reaction (confirmed), top 10", wide.loc[both].sort_values("talk").head(10).index)
    show("reposts (confirmed), top 12 up", wide.loc[conf.index[conf["reposts"]]].sort_values("reposts", ascending=False).head(12).index)
    show("reposts (confirmed), top 8 down", wide.loc[conf.index[conf["reposts"]]].sort_values("reposts").head(8).index)
    opp = wide.index[(conf["lift"] & conf["comments"]) & (np.sign(wide["lift"]) != np.sign(wide["comments"]))]
    show("reactions and comments move in opposite directions (both confirmed)", opp)


if __name__ == "__main__":
    main()
