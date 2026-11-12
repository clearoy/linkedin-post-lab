"""Markdown tables for FINDINGS.md from the result CSVs.

Usage:
    uv run python research/tables.py --effects existing r1 r2 --joint r2 > research/results/tables.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from stats import RESULTS  # noqa: E402


def pct(m: float) -> str:
    return f"{(m - 1) * 100:+.0f}%"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--effects", nargs="+", default=["existing", "r1"])
    p.add_argument("--joint", default="r1")
    args = p.parse_args()
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv").assign(run=n) for n in args.effects]).drop_duplicates("id")
    joint = pd.read_csv(RESULTS / f"joint_{args.joint}.csv").set_index("id")

    print("## Confirmed single effects\n")
    print("| Property | Share of posts | Reactions vs author's usual (95% CI) | Held-out authors | Win rate | Same media only | Same length | Unique effect (joint) |")
    print("|---|---|---|---|---|---|---|---|")
    c = eff[eff["confirmed"]].sort_values("mult", ascending=False)
    for _, r in c.iterrows():
        j = joint.loc[r["id"]] if r["id"] in joint.index else None
        uj = f"{pct(j['mult'])} [{pct(j['lo'])}, {pct(j['hi'])}]" if j is not None else "grouped with a similar one"
        print(f"| {r['question']} | {r['share']:.0%} | {pct(r['mult'])} [{pct(r['lo'])}, {pct(r['hi'])}] | "
              f"{pct(r['test_mult'])} | {r['win_rate']:.2f} | {pct(r['same_media_mult'])} | {pct(r['length_adj_mult'])} | {uj} |")

    print("\n## Tested, no confirmed effect\n")
    print("| Property | Share | Pooled estimate (95% CI) | Status |")
    print("|---|---|---|---|")
    for _, r in eff[~eff["confirmed"]].sort_values("p").iterrows():
        print(f"| {r['question']} | {r['share']:.0%} | {pct(r['mult'])} [{pct(r['lo'])}, {pct(r['hi'])}] | {r['status']} |")

    ck = pd.read_csv(RESULTS / "checklist.csv")
    print("\n## Checklist (held-out authors)\n")
    print("| Checklist items in the post | Share of posts | Reactions vs author's usual (95% CI) |")
    print("|---|---|---|")
    for _, r in ck.iterrows():
        print(f"| {r['checklist_items']} | {r['share']:.0%} | {pct(r['mult'])} [{pct(r['lo'])}, {pct(r['hi'])}] |")

    led = json.loads((RESULTS / "jev_ledger.json").read_text())
    print("\n## Jev spend\n\n| Run | Posts | Questions | Cost |\n|---|---|---|---|")
    for r in led["runs"]:
        print(f"| {r['name']} | {r['posts']:,} | {r['questions']} | ${r['cost_usd']:.2f} |")
    print(f"| **Total** | | | **${led['spent_usd']:.2f}** of ${led['budget_usd']:.2f} |")


if __name__ == "__main__":
    main()
