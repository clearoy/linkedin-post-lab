"""Which confirmed properties still matter once the others are held fixed?

Usage:
    uv run python research/joint.py --effects existing [new ...] --judgments DIR [DIR ...]

Single-property effects overlap: posts that thank people also tend to
describe an event and express pride. This script
  1. groups the confirmed properties that mostly co-occur (|corr| >= --corr on
     train posts) and keeps the one with the largest train effect per group;
  2. fits one within-author OLS of lift on all kept properties plus log words
     (post length is part of the text), on train authors, author-clustered SEs;
  3. refits on test authors and reports whether each unique effect keeps its
     sign there (p one-sided), then a pooled fit on all authors.

Writes research/results/joint_<name>.csv and research/results/joint_<name>.json
(the pooled coefficients, used for residuals by hypotheses.py).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from stats import EXISTING, RESULTS, YES, bh, load_judgments, load_posts, text_checks, within_ols  # noqa: E402


def frame(judgment_dirs) -> pd.DataFrame:
    df = load_posts().merge(load_judgments(judgment_dirs), on="urn").reset_index(drop=True)
    return pd.concat([df, text_checks(df["body"])], axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--effects", nargs="+", default=["existing"])
    p.add_argument("--judgments", type=Path, nargs="+", default=EXISTING)
    p.add_argument("--corr", type=float, default=0.4)
    p.add_argument("--name", default="existing")
    p.add_argument("--select", choices=["confirmed", "train"], default="confirmed",
                   help="confirmed: properties confirmed on test authors; train: every property that passed "
                        "discovery on train (no test information, used for the checklist)")
    args = p.parse_args()

    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in args.effects])
    eff = eff[eff["confirmed"] if args.select == "confirmed" else eff["train_q"] < 0.05].drop_duplicates("id")
    df = frame(args.judgments)
    df = df[df[eff["id"]].notna().all(axis=1)].reset_index(drop=True)  # posts with every answer
    print(f"{len(df)} posts with all answers, {df['author'].nunique()} authors", flush=True)
    X = (df[eff["id"]] >= YES).astype(float)
    tr = (df["split"] == "train").values

    # 1. groups of co-occurring properties (complete linkage: every pair in a group has |corr| >= threshold)
    C = X[tr].corr().abs().fillna(0).to_numpy().copy()
    np.fill_diagonal(C, 1)
    groups = fcluster(linkage(squareform(1 - C, checks=False), "complete"), 1 - args.corr, "distance")
    eff = eff.assign(group=groups, strength=np.abs(np.log(eff["train_mult"])))
    keep = eff.sort_values("strength", ascending=False).groupby("group").head(1)
    members = eff.groupby("group")["id"].apply(list).to_dict()
    cols = list(keep["id"])
    print(f"{len(eff)} properties ({args.select}) -> {len(cols)} groups", flush=True)

    Z = np.column_stack([X[cols].to_numpy(), np.log1p(df["words"].to_numpy())])
    names = cols + ["log_words"]

    def fit(mask):
        return within_ols(df.loc[mask, "lift"].to_numpy(), Z[mask], df.loc[mask, "author"].to_numpy())

    a, b, al = fit(tr), fit(~tr), fit(np.ones(len(df), bool))
    rows = []
    for j, n in enumerate(names):
        same = np.sign(b["beta"][j]) == np.sign(a["beta"][j])
        one = b["p"][j] / 2 if same else 1 - b["p"][j] / 2
        info = keep.set_index("id").loc[n] if n in cols else None
        rows.append({
            "id": n, "question": info["question"] if info is not None else "log(1 + words)",
            "also_covers": [m for m in members[info["group"]] if m != n] if info is not None else [],
            "single_mult": info["mult"] if info is not None else np.nan,
            "train_mult": np.exp(a["beta"][j]), "train_p": a["p"][j],
            "test_mult": np.exp(b["beta"][j]), "test_p_one_sided": one,
            "mult": np.exp(al["beta"][j]), "lo": np.exp(al["beta"][j] - 1.96 * al["se"][j]),
            "hi": np.exp(al["beta"][j] + 1.96 * al["se"][j]), "p": al["p"][j],
        })
    res = pd.DataFrame(rows)
    res["train_q"] = bh(res["train_p"])
    res["test_q"] = bh(res["test_p_one_sided"])
    res["unique_effect"] = np.where((res["train_q"] < 0.05) & (res["test_q"] < 0.05), "confirmed",
                                    np.where(res["train_q"] < 0.05, "train only", "none"))
    res = res.sort_values("mult", ascending=False)
    res.to_csv(RESULTS / f"joint_{args.name}.csv", index=False)
    (RESULTS / f"joint_{args.name}.json").write_text(json.dumps(
        {"columns": names, "beta": dict(zip(names, map(float, al["beta"])))}, indent=1))
    for _, r in res.iterrows():
        print(f"  {r['unique_effect']:10s} joint {r['mult']:.2f}x [{r['lo']:.2f},{r['hi']:.2f}] "
              f"(train {r['train_mult']:.2f} test {r['test_mult']:.2f}; alone {r['single_mult']:.2f}) "
              f"{r['id']}  +{len(r['also_covers'])}")
    print(f"Wrote {RESULTS / f'joint_{args.name}.csv'}")


if __name__ == "__main__":
    main()
