"""Cheap first test of new hypotheses on the screening authors.

Usage:
    uv run python research/screen.py --round 1 --judgments data/processed/research/judgments_r1_screen

For each candidate, on screening authors only:
  marginal     within-author effect on its own
  incremental  effect with the joint model's properties and length held fixed
               (does it add anything the 147 questions do not already say?)
  overlap      highest |correlation| with any existing question
A candidate (true of 1% to 99% of posts) advances to the paid confirmation run when marginal or incremental
p < 0.05 and it is not a near-copy of an existing question (overlap < 0.8).
The screen is deliberately liberal: the real test is on test authors later.

Writes research/results/screen_r<N>.csv (with an `advance` column).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from joint import frame  # noqa: E402
from stats import EXISTING, RESEARCH_DATA, RESULTS, YES, effect  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--judgments", type=Path, required=True)
    p.add_argument("--joint", default="existing")
    p.add_argument("--base", type=Path, nargs="+", default=EXISTING,
                   help="judgments already measured (overlap check and joint-model columns)")
    p.add_argument("--max-advance", type=int, default=30)
    args = p.parse_args()

    bank = {f["id"]: f for f in json.loads((RESEARCH_DATA / f"bank_round{args.round}.json").read_text())["features"]}
    new = pd.read_parquet(args.judgments / "features.parquet")
    base = frame(args.base)
    old_cols = [c for c in base.columns if "." in c]
    df = base.merge(new, on="urn").reset_index(drop=True)
    jcols = json.loads((RESULTS / f"joint_{args.joint}.json").read_text())["columns"]
    Z = np.column_stack([(df[[c for c in jcols if c != "log_words"]] >= YES).to_numpy(float),
                         np.log1p(df["words"].to_numpy())])
    OLD = (df[old_cols] >= YES).to_numpy(float)
    y, a = df["lift"].to_numpy(), df["author"].to_numpy()
    print(f"screen: {len(df)} posts, {df['author'].nunique()} authors, {len(bank)} candidates", flush=True)

    rows = []
    for fid, f in bank.items():
        x = df[fid].to_numpy()
        xb = (x >= YES).astype(float)
        if xb.std() == 0:
            continue
        m = effect(y, x, a)
        inc = effect(y, x, a, covars=Z)
        corr = np.abs(np.nan_to_num([np.corrcoef(xb, OLD[:, j])[0, 1] for j in range(OLD.shape[1])]))
        j = int(corr.argmax())
        rows.append({"id": fid, "question": f["question"], "source": f["dimension"], "share": m["share"],
                     "mult": m["multiplier"], "p": m["p"], "win_rate": m["win_rate"],
                     "incr_mult": inc["multiplier"], "incr_p": inc["p"],
                     "overlap": corr[j], "closest_existing": old_cols[j], "hypothesis": f.get("hypothesis", "")})
    res = pd.DataFrame(rows)
    ok = (res["share"].between(0.01, 0.99)) & (res["overlap"] < 0.8) & ((res["p"] < 0.05) | (res["incr_p"] < 0.05))
    res["screen_score"] = np.minimum(res["p"], res["incr_p"])
    res["advance"] = False
    res.loc[res[ok].nsmallest(args.max_advance, "screen_score").index, "advance"] = True
    res = res.sort_values(["advance", "screen_score"], ascending=[False, True])
    out = RESULTS / f"screen_r{args.round}.csv"
    res.round(4).to_csv(out, index=False)
    for _, r in res.iterrows():
        print(f"  {'ADV' if r['advance'] else '   '} {r['mult']:.2f}x p={r['p']:.3f} incr {r['incr_mult']:.2f}x "
              f"p={r['incr_p']:.3f} share {r['share']:.2f} overlap {r['overlap']:.2f}  {r['id']}")
    print(f"{res['advance'].sum()} advance. Wrote {out}")


if __name__ == "__main__":
    main()
