"""Post types: eight kinds of post found by clustering Jev's answers, and what works inside each.

Usage:
    uv run python research/archetypes.py

1. K-means (k = 8) on the 147 original questions' P(yes), fit on train authors' posts only.
   Each cluster is named by hand from its most distinctive questions (NAMES below).
2. A new post gets the type whose cluster centre is nearest to its 147 answers
   (about $0.001 of Jev per draft, so no shortcut is needed).
3. League table: each type's reactions, comments, reposts, hit and flop rate against
   the same author's other posts, on train and on held-out test authors.
4. Levers inside each type: the within-author effect of every confirmed property
   among posts of that type only (BH within type, with the held-out authors' estimate).

Writes research/results/archetypes.csv, archetype_levers.csv, archetypes.json (centres for the classifier).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from compare_dashboard import DIRS  # noqa: E402
from joint import frame  # noqa: E402
from outcomes import add_outcomes, question_text  # noqa: E402
from playbook import author_boot  # noqa: E402
from stats import RESULTS, YES, bh, demean, effect  # noqa: E402

K = 8
# cluster -> name, by the question whose centre value is highest in that cluster (checked to be one-to-one)
NAMES = {
    "tone_emotion.expresses_gratitude_sympathy_well_wishes": ("Personal moment or thank-you", "A real moment from the author's life or work: an event attended, a milestone, thanks to people."),
    "promotion_intent.announces_job_opening": ("Job opening", "Announces a role and says who should apply."),
    "call_to_action.ends_with_reader_question": ("Discussion question", "Makes a point and ends by asking readers what they think or do."),
    "promotion_intent.promotes_own_product_or_service": ("Product or service promotion", "Explains what the author's product, service or project does."),
    "promotion_intent.promotes_event": ("Event or link promotion", "Points readers to an event, sign-up or external page."),
    "specificity_evidence.cites_external_authority": ("Industry news and analysis", "Reports or analyses a development, citing outside sources and numbers."),
    "argument_claims.promotes_value_or_mindset": ("Opinion and advice", "Argues for a way of thinking or working; advice, contrasts, problem then solution."),
}
SHORT = ("Short update", "A few lines with little detail (typically under 40 words).")


def name_clusters(centres: pd.DataFrame) -> dict[int, tuple[str, str]]:
    names = {int(centres[q].idxmax()): v for q, v in NAMES.items()}
    assert len(names) == len(NAMES), "two marker questions point at the same cluster"
    rest = [c for c in centres.index if c not in names]
    assert len(rest) == 1
    names[rest[0]] = SHORT
    return names


def assign(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1).argmin(1)


def main() -> None:
    df = add_outcomes(frame(DIRS))
    qtext = question_text()
    old_q = [c for c in pd.read_parquet(DIRS[0] / "features.parquet").columns
             if c != "urn" and c in df.columns and df[c].notna().all()]
    tr = df["split"] == "train"
    km = KMeans(K, n_init=10, random_state=0).fit(df.loc[tr, old_q].to_numpy(float))
    centres = pd.DataFrame(km.cluster_centers_, columns=old_q)
    names = name_clusters(centres)

    df["type"] = assign(df[old_q].to_numpy(float), centres.to_numpy())
    print("share of posts by type, train vs test:",
          pd.crosstab(df["type"].map(lambda c: names[c][0]), df["split"], normalize="columns").round(3).to_dict())

    rows = []
    for o in ["lift", "comments", "reposts", "hit", "flop"]:
        df[o + "_w"] = demean(df[o].to_numpy(), df["author"].to_numpy())
    for c in range(K):
        g = df[df["type"] == c]
        row = {"type": names[c][0], "description": names[c][1], "share": len(g) / len(df), "posts": len(g),
               "authors": g["author"].nunique(), "median_words": float(g["words"].median())}
        for o in ["lift", "comments", "reposts"]:
            lo, hi = author_boot(g[o + "_w"], g["author"])
            row[o], row[o + "_lo"], row[o + "_hi"] = np.exp(g[o + "_w"].mean()), np.exp(lo), np.exp(hi)
            row[o + "_test"] = np.exp(g.loc[g["split"] == "test", o + "_w"].mean())
        for o in ["hit", "flop"]:
            row[o + "_rate"] = g[o].mean()
            row[o + "_points"] = 100 * g[o + "_w"].mean()
        diff = (centres.loc[c] - centres.mean()).sort_values(ascending=False)
        row["most_distinctive"] = " | ".join(qtext.get(i, i) for i in diff.index[:5])
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("lift", ascending=False)
    table.round(4).to_csv(RESULTS / "archetypes.csv", index=False)
    print(table[["type", "share", "median_words", "lift", "lift_test", "comments", "reposts", "hit_rate", "flop_rate"]].round(2).to_string(index=False))

    # levers inside each type
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id")
    conf = eff[eff["confirmed"]].set_index("id")
    rows = []
    for c in range(K):
        sub = df[df["type"] == c]
        for i in conf.index:
            s = sub[sub[i].notna()]
            s = s[s.groupby("author")["urn"].transform("count") >= 2]
            share = (s[i] >= YES).mean()
            if not 0.04 <= share <= 0.96:
                continue
            e = effect(s["lift"].to_numpy(), s[i].to_numpy(), s["author"].to_numpy())
            t = s[s["split"] == "test"]
            et = effect(t["lift"].to_numpy(), t[i].to_numpy(), t["author"].to_numpy())["multiplier"] if (t[i] >= YES).std() > 0 else np.nan
            rows.append({"type": names[c][0], "id": i, "question": conf.loc[i, "question"], "share_in_type": share,
                         "mult": e["multiplier"], "lo": e["lo"], "hi": e["hi"], "p": e["p"], "test_mult": et,
                         "overall_mult": conf.loc[i, "mult"], "authors": e["authors_with_both"]})
    lev = pd.DataFrame(rows)
    lev["q"] = lev.groupby("type")["p"].transform(bh)
    lev["significant"] = (lev["q"] < 0.05) & (np.sign(np.log(lev["test_mult"])) == np.sign(np.log(lev["mult"])))
    lev.sort_values(["type", "mult"], ascending=[True, False]).round(4).to_csv(RESULTS / "archetype_levers.csv", index=False)
    for t, g in lev[lev["significant"]].groupby("type"):
        g = g.sort_values("mult", ascending=False)
        print(f"\n{t}: {len(g)} significant levers")
        for _, r in pd.concat([g.head(5), g.tail(3)]).drop_duplicates("id").iterrows():
            print(f"   {r['mult']:.2f}x [{r['lo']:.2f},{r['hi']:.2f}] test {r['test_mult']:.2f} share {r['share_in_type']:.2f}  {r['id']}")

    out = {"k": K, "questions": old_q,
           "centres": {names[c][0]: [round(float(v), 4) for v in centres.loc[c]] for c in range(K)},
           "descriptions": {names[c][0]: names[c][1] for c in range(K)}}
    (RESULTS / "archetypes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
