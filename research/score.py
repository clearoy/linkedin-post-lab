"""Score research hypotheses with Jev on chosen author sets, inside the $5 research budget.

Usage:
    uv run python research/score.py --round 1 --sets screen --name r1_screen
    uv run python research/score.py --round 1 --sets rest test --ids-from results/screen_r1.csv --name r1_confirm

Every paid run is appended to research/results/jev_ledger.json. A run is
refused if its estimate would take the ledger past --budget (default $5.00),
and it stops sending requests once reported spend hits what is left.

Writes data/processed/research/judgments_<name>/ (features.parquet, judgments.parquet, run_meta.json).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from stats import PROCESSED, RESEARCH_DATA, RESULTS  # noqa: E402

from linkedin_analysis import judge  # noqa: E402
from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.llm import DecisionsClient, LLMSettings  # noqa: E402

LEDGER = RESULTS / "jev_ledger.json"


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(r["cost_usd"] for r in json.loads(LEDGER.read_text())["runs"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--sets", nargs="+", required=True, help="generate / screen / rest / test")
    p.add_argument("--ids-from", type=Path, default=None, help="CSV with an `id` column (and `advance`) to restrict questions")
    p.add_argument("--name", required=True)
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    features = json.loads((RESEARCH_DATA / f"bank_round{args.round}.json").read_text())["features"]
    if args.ids_from:
        sel = pd.read_csv(args.ids_from)
        if "advance" in sel:
            sel = sel[sel["advance"]]
        keep = set(sel["id"])
        features = [f for f in features if f["id"] in keep]
    sets = json.loads((RESEARCH_DATA / "authors.json").read_text())
    splits = pd.read_parquet(PROCESSED / "splits.parquet")
    sets["test"] = sorted(splits.loc[splits["split"] == "test", "author"])
    authors = {a for s in args.sets for a in sets[s]}
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "author"]]
    urns = set(labels.loc[labels["author"].isin(authors), "urn"])
    posts = load_all().loc[lambda d: d["urn"].isin(urns), ["urn", "media", "body"]].sort_values("urn").to_dict("records")

    estimate = judge.estimate_cost(len(posts), len(features))
    left = args.budget - spent()
    print(f"{len(authors)} authors, {len(posts)} posts x {len(features)} questions; estimate ${estimate:.2f}; "
          f"budget left ${left:.2f}", flush=True)
    if args.dry_run:
        return
    if estimate > left:
        sys.exit("Estimate is over the remaining research budget; refusing to run.")

    client = DecisionsClient(LLMSettings.from_env("openrouter"), max_connections=args.workers, max_cost_usd=left)
    start = time.time()
    rows, skipped = judge.judge(client.decide, posts, features, 50, args.workers)
    long = pd.DataFrame(rows)
    wide = long.pivot(index="urn", columns="feature_id", values="p_yes")[[f["id"] for f in features]].reset_index()
    out = RESEARCH_DATA / f"judgments_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(out / "judgments.parquet", index=False)
    wide.to_parquet(out / "features.parquet", index=False)
    usage = client.usage.to_dict()
    meta = {"name": args.name, "round": args.round, "sets": args.sets, "authors": len(authors), "posts": len(posts),
            "questions": len(features), "estimate_usd": round(estimate, 4), "skipped_requests": skipped,
            "status": long["status"].value_counts().to_dict(), "seconds": round(time.time() - start, 1), "usage": usage}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))

    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"budget_usd": args.budget, "runs": []}
    ledger["runs"].append({"name": args.name, "posts": len(posts), "questions": len(features),
                           "estimate_usd": round(estimate, 4), "cost_usd": round(usage.get("cost_usd", 0.0), 4)})
    ledger["spent_usd"] = round(sum(r["cost_usd"] for r in ledger["runs"]), 4)
    LEDGER.write_text(json.dumps(ledger, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"Spent ${usage.get('cost_usd', 0):.4f} this run; ledger total ${ledger['spent_usd']:.2f} of ${args.budget:.2f}")


if __name__ == "__main__":
    main()
