"""Turn confirmed effects into guidance: a checklist and per-post-type advice.

Usage:
    uv run python research/playbook.py --effects existing r1 --judgments DIR [DIR ...]

1. Checklist. Positive properties are chosen with train authors only: from a
   joint model over every property that passed discovery on train
   (joint.py --select train), those with a train-significant unique effect above 1.
   On test authors, posts are grouped by how many checklist items they have,
   and each group's mean lift (vs the same author's other posts) is reported
   as a reaction multiplier with an author-bootstrap CI.
2. By post type. For common kinds of post (advice, promotion, job listing,
   news/link, event), the within-author effect of each confirmed property
   among posts of that kind only, BH-adjusted within the kind. This answers
   "I am writing an advice post: what goes with more reactions?"

Writes research/results/checklist.csv and research/results/by_post_type.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from joint import frame  # noqa: E402
from stats import RESULTS, YES, bh, demean, effect  # noqa: E402

POST_TYPES = {
    "advice / how-to": "call_to_action.provides_actionable_advice",
    "promotes own product or service": "promotion_intent.promotes_own_product_or_service",
    "job opening": "promotion_intent.announces_job_opening",
    "shares news or an external source": "media_links.summarizes_external_source",
    "promotes an event": "promotion_intent.promotes_event",
    "industry trend / analysis": "argument_claims.describes_industry_trend_or_change",
}


def author_boot(values: pd.Series, authors: pd.Series, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of a mean, resampling authors."""
    g = pd.DataFrame({"v": values.to_numpy(), "a": authors.to_numpy()}).groupby("a")["v"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    means = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--effects", nargs="+", default=["existing", "r1"])
    p.add_argument("--joint", default="r2_train", help="joint fit from train-only selection (joint.py --select train)")
    p.add_argument("--judgments", type=Path, nargs="+", required=True)
    args = p.parse_args()

    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in args.effects]).drop_duplicates("id")
    conf = eff[eff["confirmed"]]
    joint = pd.read_csv(RESULTS / f"joint_{args.joint}.csv")
    df = frame(args.judgments)
    df = df[df[list(set(conf["id"]) | set(joint["id"]) - {"log_words"})].notna().all(axis=1)].reset_index(drop=True)

    # 1. checklist: unique positive effects, chosen on train (train_mult > 1 and train_q < 0.05)
    items = joint[(joint["train_mult"] > 1) & (joint["train_q"] < 0.05) & (joint["id"] != "log_words")]
    cols = list(items["id"])
    df["n_items"] = (df[cols] >= YES).sum(axis=1)
    df["lift_w"] = demean(df["lift"].to_numpy(), df["author"].to_numpy())  # vs the same author's mean
    te = df[df["split"] == "test"]
    rows = []
    for k, g in te.groupby(te["n_items"].clip(upper=4)):
        lo, hi = author_boot(g["lift_w"], g["author"])
        rows.append({"checklist_items": f"{k}+" if k == 4 else str(k), "posts": len(g), "share": len(g) / len(te),
                     "mult": float(np.exp(g["lift_w"].mean())), "lo": float(np.exp(lo)), "hi": float(np.exp(hi)),
                     "median_reaction_ratio": float(((g["reactions"] + 2) / (g["baseline"] + 2)).median())})
    ck = pd.DataFrame(rows)
    ck.attrs["items"] = cols
    ck.to_csv(RESULTS / "checklist.csv", index=False)
    pd.Series(cols, name="id").to_csv(RESULTS / "checklist_items.csv", index=False)
    rho = te[["n_items", "lift_w"]].corr("spearman").iloc[0, 1]
    print(f"checklist ({len(cols)} items, chosen on train): {cols}")
    print(ck.round(3).to_string(index=False))
    print(f"test spearman(n_items, within-author lift) = {rho:.3f}")

    # 2. levers within each post type (all authors; BH within type)
    rows = []
    for label, col in POST_TYPES.items():
        sub = df[df[col] >= YES]
        sub = sub[sub.groupby("author")["urn"].transform("count") >= 2]
        for c in conf["id"]:
            if c == col or c in POST_TYPES.values():
                continue
            share = (sub[c] >= YES).mean()
            if not 0.05 <= share <= 0.95:
                continue
            e = effect(sub["lift"].to_numpy(), sub[c].to_numpy(), sub["author"].to_numpy())
            t = sub[sub["split"] == "test"]
            et = effect(t["lift"].to_numpy(), t[c].to_numpy(), t["author"].to_numpy()) if (t[c] >= YES).std() > 0 else {"multiplier": np.nan}
            rows.append({"post_type": label, "type_posts": len(sub), "type_authors": sub["author"].nunique(),
                         "id": c, "question": conf.set_index("id").loc[c, "question"], "share": share,
                         "mult": e["multiplier"], "lo": e["lo"], "hi": e["hi"], "p": e["p"],
                         "test_mult": et["multiplier"], "overall_mult": conf.set_index("id").loc[c, "mult"]})
    bt = pd.DataFrame(rows)
    bt["q"] = bt.groupby("post_type")["p"].transform(bh)
    bt = bt.sort_values(["post_type", "mult"], ascending=[True, False])
    bt.to_csv(RESULTS / "by_post_type.csv", index=False)
    for label, g in bt.groupby("post_type"):
        s = g[g["q"] < 0.05]
        print(f"\n{label}: {g['type_posts'].iloc[0]} posts / {g['type_authors'].iloc[0]} authors; {len(s)} significant")
        for _, r in pd.concat([s.head(5), s.tail(3)]).drop_duplicates("id").iterrows():
            print(f"   {r['mult']:.2f}x [{r['lo']:.2f},{r['hi']:.2f}] test {r['test_mult']:.2f} share {r['share']:.2f}  {r['id']}")


if __name__ == "__main__":
    main()
