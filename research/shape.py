"""Shape of the effects: how much length, does it wear off, do properties stack, does variety help.

Usage:
    uv run python research/shape.py

1. Length curve: reactions and comments by word-count band, within author, with and
   without holding the content properties (train-only joint model columns) fixed.
2. Wear-off: for each confirmed property, does its effect shrink for authors who
   use it in a larger share of their posts? Interaction x * (author share), BH over properties.
3. Combinations: pairs of the main confirmed properties. Interaction term on train
   (BH q < 0.05), confirmed on test authors with the same sign.
4. Variety: is a post that is unlike the author's other posts (distance from the
   author's average answer profile) better or worse, before and after holding its content fixed?

Writes research/results/shape_length.csv, shape_wearoff.csv, shape_combos.csv, shape_variety.json.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from compare_dashboard import DIRS  # noqa: E402
from joint import frame  # noqa: E402
from outcomes import add_outcomes, question_text  # noqa: E402
from stats import RESULTS, YES, bh, demean, within_ols  # noqa: E402

BANDS = [0, 40, 80, 120, 160, 200, 250, 300, 400, 1000]
REF = "120-160"


def length_curve(df: pd.DataFrame, joint_cols: list[str]) -> pd.DataFrame:
    labels = [f"{a}-{b}" if b < 1000 else f"{a}+" for a, b in zip(BANDS[:-1], BANDS[1:])]
    band = pd.cut(df["words"], BANDS, labels=labels, right=False)
    D = pd.get_dummies(band).astype(float)
    cols = [c for c in labels if c != REF]
    full = df[df[joint_cols].notna().all(axis=1)].index
    rows = []
    for outcome in ["lift", "comments"]:
        raw = within_ols(df[outcome].to_numpy(), D[cols].to_numpy(), df["author"].to_numpy())
        Z = np.column_stack([D.loc[full, cols].to_numpy(), (df.loc[full, joint_cols] >= YES).to_numpy(float)])
        adj = within_ols(df.loc[full, outcome].to_numpy(), Z, df.loc[full, "author"].to_numpy())
        for i, c in enumerate(cols):
            rows.append({"outcome": outcome, "words": c, "share": float(D[c].mean()),
                         "mult": np.exp(raw["beta"][i]), "lo": np.exp(raw["beta"][i] - 1.96 * raw["se"][i]),
                         "hi": np.exp(raw["beta"][i] + 1.96 * raw["se"][i]), "p": raw["p"][i],
                         "content_fixed_mult": np.exp(adj["beta"][i]),
                         "content_fixed_lo": np.exp(adj["beta"][i] - 1.96 * adj["se"][i]),
                         "content_fixed_hi": np.exp(adj["beta"][i] + 1.96 * adj["se"][i])})
        rows.append({"outcome": outcome, "words": REF, "share": float(D[REF].mean()), "mult": 1.0, "lo": 1.0, "hi": 1.0,
                     "p": np.nan, "content_fixed_mult": 1.0, "content_fixed_lo": 1.0, "content_fixed_hi": 1.0})
    out = pd.DataFrame(rows)
    out["order"] = out["words"].map({c: i for i, c in enumerate(labels)})
    return out.sort_values(["outcome", "order"]).drop(columns="order")


def wearoff(df: pd.DataFrame, props: pd.DataFrame, qtext: dict) -> pd.DataFrame:
    rows = []
    for c in props["id"]:
        d = df[df[c].notna()]
        x = (d[c] >= YES).to_numpy(float)
        au = d["author"].to_numpy()
        share = pd.Series(x).groupby(au).transform("mean").to_numpy()
        if not 0.03 <= x.mean() <= 0.7:
            continue
        r = within_ols(d["lift"].to_numpy(), np.column_stack([x, x * (share - share.mean())]), au)
        row = {"id": c, "question": qtext.get(c, c), "share": x.mean(), "mult": np.exp(r["beta"][0]),
               "interaction_per_10pts": np.exp(r["beta"][1] * 0.1), "p": r["p"][1]}
        mixed = (share > 0) & (share < 1)  # only authors with both kinds of post carry information
        cuts = np.quantile(pd.Series(share[mixed]).groupby(au[mixed]).first(), [1 / 3, 2 / 3])
        for name, m in [("rare", mixed & (share <= cuts[0])), ("some", mixed & (share > cuts[0]) & (share <= cuts[1])),
                        ("often", mixed & (share > cuts[1]))]:
            e = within_ols(d["lift"].to_numpy()[m], x[m], au[m])
            row[f"mult_{name}"], row[f"share_{name}"] = np.exp(e["beta"][0]), share[m].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q"] = bh(out["p"])
    return out.sort_values("p")


def combos(df: pd.DataFrame, ids: list[str], qtext: dict) -> pd.DataFrame:
    d = df[df[ids].notna().all(axis=1)]
    X = (d[ids] >= YES).astype(float)
    rows = []
    for a, b in combinations(ids, 2):
        both = X[a] * X[b]
        if both.mean() < 0.01 or abs(np.corrcoef(X[a], X[b])[0, 1]) > 0.6:
            continue
        res = {}
        for name, m in [("train", d["split"] == "train"), ("test", d["split"] == "test"), ("all", d["split"].notna())]:
            Z = np.column_stack([X.loc[m, a], X.loc[m, b], both[m]])
            res[name] = within_ols(d.loc[m, "lift"].to_numpy(), Z, d.loc[m, "author"].to_numpy())
        tr, te, al = res["train"], res["test"], res["all"]
        same = np.sign(te["beta"][2]) == np.sign(tr["beta"][2])
        rows.append({"a": a, "b": b, "a_question": qtext.get(a, a), "b_question": qtext.get(b, b),
                     "share_both": both.mean(), "a_alone": np.exp(al["beta"][0]), "b_alone": np.exp(al["beta"][1]),
                     "extra_when_both": np.exp(al["beta"][2]), "both_total": np.exp(al["beta"].sum()),
                     "train_extra": np.exp(tr["beta"][2]), "train_p": tr["p"][2], "test_extra": np.exp(te["beta"][2]),
                     "test_p_one_sided": te["p"][2] / 2 if same else 1 - te["p"][2] / 2})
    out = pd.DataFrame(rows)
    out["train_q"] = bh(out["train_p"])
    passed = out["train_q"] < 0.05
    out["test_q"] = np.nan
    out.loc[passed, "test_q"] = bh(out.loc[passed, "test_p_one_sided"])
    out["confirmed"] = passed & (out["test_q"] < 0.05)
    return out.sort_values("train_p")


def variety(df: pd.DataFrame, old_q: list[str], joint_cols: list[str]) -> dict:
    """Distance of each post from the author's other posts in answer space (1 - cosine to the leave-one-out mean)."""
    X = df[old_q].to_numpy(float)
    au = df["author"].to_numpy()
    n = pd.Series(1.0, index=df.index).groupby(au).transform("sum").to_numpy()
    sums = np.zeros_like(X)
    for c in range(X.shape[1]):
        sums[:, c] = pd.Series(X[:, c]).groupby(au).transform("sum").to_numpy()
    others = (sums - X) / (n - 1)[:, None]
    Xc, Oc = X - X.mean(0), others - X.mean(0)  # centre on the average post so "typical" answers do not dominate
    cos = (Xc * Oc).sum(1) / (np.linalg.norm(Xc, axis=1) * np.linalg.norm(Oc, axis=1) + 1e-12)
    dist = 1 - cos
    z = demean(dist, au)
    z = z / z.std()
    out = {"posts": len(df)}
    r = within_ols(df["lift"].to_numpy(), z, au)
    out["per_sd_unusual"] = {"mult": float(np.exp(r["beta"][0])), "p": float(r["p"][0]),
                             "lo": float(np.exp(r["beta"][0] - 1.96 * r["se"][0])), "hi": float(np.exp(r["beta"][0] + 1.96 * r["se"][0]))}
    full = df[joint_cols].notna().all(axis=1).to_numpy()
    Z = np.column_stack([z[full], (df.loc[full, joint_cols] >= YES).to_numpy(float), np.log1p(df.loc[full, "words"])])
    r = within_ols(df.loc[full, "lift"].to_numpy(), Z, au[full])
    out["per_sd_unusual_content_fixed"] = {"mult": float(np.exp(r["beta"][0])), "p": float(r["p"][0]),
                                           "lo": float(np.exp(r["beta"][0] - 1.96 * r["se"][0])),
                                           "hi": float(np.exp(r["beta"][0] + 1.96 * r["se"][0]))}
    q = pd.qcut(pd.Series(z).groupby(au).rank(pct=True), 5, labels=False)
    lw = demean(df["lift"].to_numpy(), au)
    out["by_fifth_most_usual_to_most_unusual"] = [float(np.exp(lw[q == k].mean())) for k in range(5)]
    return out


def main() -> None:
    df = add_outcomes(frame(DIRS))
    qtext = question_text()
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id")
    conf = eff[eff["confirmed"]]
    joint_cols = [c for c in pd.read_csv(RESULTS / "joint_r2_train.csv")["id"] if c != "log_words"]
    old_q = [c for c in pd.read_parquet(DIRS[0] / "features.parquet").columns
             if c != "urn" and c in df.columns and df[c].notna().all()]

    lc = length_curve(df, [c for c in joint_cols if c not in ("text.under_80_words", "text.over_250_words")])
    lc.round(4).to_csv(RESULTS / "shape_length.csv", index=False)
    print(lc[lc["outcome"] == "lift"].round(3).to_string(index=False))

    w = wearoff(df, conf, qtext)
    w.round(4).to_csv(RESULTS / "shape_wearoff.csv", index=False)
    pos = w[w["mult"] > 1]
    print(f"\nwear-off: {len(w)} properties; interaction q<0.05: {(w['q'] < 0.05).sum()}; among {len(pos)} positive ones, "
          f"{(pos['interaction_per_10pts'] < 1).sum()} shrink with author usage; median rare/some/often "
          f"{pos['mult_rare'].median():.2f}/{pos['mult_some'].median():.2f}/{pos['mult_often'].median():.2f}")
    print(w.head(12)[["id", "mult", "mult_rare", "mult_some", "mult_often", "interaction_per_10pts", "q"]].round(3).to_string(index=False))

    main_ids = list(pd.read_csv(RESULTS / "joint_r2.csv").query("id != 'log_words'")
                    .assign(s=lambda d: np.abs(np.log(d["mult"]))).sort_values("s", ascending=False)["id"].head(28))
    cb = combos(df, main_ids, qtext)
    cb.round(4).to_csv(RESULTS / "shape_combos.csv", index=False)
    print(f"\ncombos: {len(cb)} pairs tested, {int((cb['train_q'] < 0.05).sum())} pass on train, {int(cb['confirmed'].sum())} confirmed")
    print(cb.head(10)[["a", "b", "share_both", "a_alone", "b_alone", "extra_when_both", "train_q", "test_extra", "confirmed"]].round(3).to_string(index=False))

    v = variety(df, old_q, joint_cols)
    (RESULTS / "shape_variety.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
