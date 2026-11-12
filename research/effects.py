"""Discovery -> confirmation test of single content properties.

Usage:
    uv run python research/effects.py --name existing                       # the 147 Jev questions + free text checks
    uv run python research/effects.py --name new --judgments DIR [DIR ...] --bank BANK.json --no-text

For every property:
  1. Discovery on train authors: within-author effect, Benjamini-Hochberg q
     over all properties tested in this run.
  2. Confirmation on test authors (never used to pick anything): a property is
     "confirmed" when q_train < 0.05 and, on test authors, it points the same
     way with BH-adjusted one-sided p < 0.05 (adjusted over the properties
     that passed discovery).
  3. Pooled estimate on all authors (the size reported), plus robustness:
     same media type only (author x media groups), with post length held
     fixed (log words), and separately in each follower band.

Writes research/results/effects_<name>.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stats import EXISTING, PROCESSED, RESULTS, bh, effect, load_judgments, load_posts, text_checks, within_ols  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="existing")
    p.add_argument("--judgments", type=Path, nargs="+", default=EXISTING)
    p.add_argument("--bank", type=Path, default=PROCESSED / "features" / "feature_bank_general.json")
    p.add_argument("--no-text", action="store_true", help="skip the free text checks")
    p.add_argument("--min-share", type=float, default=0.03)
    args = p.parse_args()

    posts = load_posts()
    wide = load_judgments(args.judgments)
    df = posts.merge(wide, on="urn").reset_index(drop=True)
    cols = [c for c in wide.columns if c != "urn"]
    if not args.no_text:
        tx = text_checks(df["body"])
        df = pd.concat([df, tx], axis=1)
        cols += list(tx.columns)
    share = (df[cols] >= 0.5).mean()
    cols = [c for c in cols if args.min_share <= share[c] <= 1 - args.min_share and df[c].notna().all()]
    bank = {f["id"]: f for f in json.loads(args.bank.read_text())["features"]}
    print(f"{len(df)} posts, {df['author'].nunique()} authors, {len(cols)} properties", flush=True)

    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    rows = []
    for c in cols:
        a, b, al = effect(tr["lift"].values, tr[c].values, tr["author"].values), \
                   effect(te["lift"].values, te[c].values, te["author"].values), \
                   effect(df["lift"].values, df[c].values, df["author"].values)
        same_media = effect(df["lift"].values, df[c].values, df["author"].values,
                            groups=(df["author"] + "|" + df["media"].fillna("none")).values)
        length = effect(df["lift"].values, df[c].values, df["author"].values,
                        covars=np.log1p(df["words"].values))
        band = {}
        for bd, g in df.groupby("band"):
            band[bd] = effect(g["lift"].values, g[c].values, g["author"].values)["multiplier"]
        # band interaction: does the slope differ between follower bands?
        xb = (df[c].values >= 0.5).astype(float)
        hi = (df["band"] == "10k-20k").values.astype(float)
        inter = within_ols(df["lift"].values, np.column_stack([xb, xb * hi]), df["author"].values)
        one_sided = b["p"] / 2 if np.sign(b["beta"]) == np.sign(a["beta"]) else 1 - b["p"] / 2
        q = bank.get(c, {})
        rows.append({
            "id": c, "question": q.get("question", c), "source": "text check" if c.startswith("text.") else "Jev",
            "share": al["share"],
            "train_mult": a["multiplier"], "train_p": a["p"],
            "test_mult": b["multiplier"], "test_lo": b["lo"], "test_hi": b["hi"], "test_p_one_sided": one_sided,
            "test_win_rate": b["win_rate"],
            "mult": al["multiplier"], "lo": al["lo"], "hi": al["hi"], "p": al["p"], "win_rate": al["win_rate"],
            "same_media_mult": same_media["multiplier"], "same_media_p": same_media["p"],
            "length_adj_mult": length["multiplier"], "length_adj_p": length["p"],
            "band_1k_10k_mult": band.get("1k-10k"), "band_10k_20k_mult": band.get("10k-20k"),
            "band_diff_p": float(inter["p"][1]),
        })
    res = pd.DataFrame(rows)
    res["train_q"] = bh(res["train_p"])
    passed = res["train_q"] < 0.05
    res["test_q"] = np.nan
    res.loc[passed, "test_q"] = bh(res.loc[passed, "test_p_one_sided"])
    res["confirmed"] = passed & (res["test_q"] < 0.05)
    res["status"] = np.where(res["confirmed"], "confirmed",
                             np.where(passed, "not replicated", "no clear effect"))
    res = res.sort_values(["confirmed", "mult"], ascending=[False, False])
    out = RESULTS / f"effects_{args.name}.csv"
    RESULTS.mkdir(exist_ok=True)
    res.round(4).to_csv(out, index=False)

    print(f"\npassed discovery: {passed.sum()}  confirmed on test authors: {res['confirmed'].sum()}")
    show = res[res["confirmed"]].sort_values("mult", ascending=False)
    for _, r in show.iterrows():
        print(f"  {r['mult']:.2f}x [{r['lo']:.2f},{r['hi']:.2f}] test {r['test_mult']:.2f}x win {r['win_rate']:.3f} "
              f"share {r['share']:.2f} media-adj {r['same_media_mult']:.2f} len-adj {r['length_adj_mult']:.2f}  {r['id']}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
