"""Score posts on every feature in a feature bank with Jev (OpenRouter decisions API).

Usage:
    uv run python scripts/judge_features.py --dry-run                          # request + cost estimate, no calls
    uv run python scripts/judge_features.py --split train --n-authors 150      # small first run, whole authors
    uv run python scripts/judge_features.py --max-cost 25                      # every labeled post

Spending is capped by --max-cost (default $5): the run refuses to start if the
estimate is over it, and stops sending requests once reported spend reaches it.
Results finished before the stop are saved.

Needs OPENROUTER_API_KEY in .env, data/processed/labels.parquet, and a
feature bank from scripts/discover_features.py.

Writes to --out (default data/processed/judgments/):
    judgments.parquet  long: urn, feature_id, p_yes, status
    features.parquet   wide: one row per urn, one P(yes) column per feature
    run_meta.json      config, counts, timing, token usage and cost
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import pandas as pd  # noqa: E402

from linkedin_analysis import judge  # noqa: E402
from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.llm import DecisionsClient, LLMSettings  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, default=PROCESSED / "features" / "feature_bank.json")
    p.add_argument("--split", default="all", choices=["all", "train", "test"], help="which authors to score")
    p.add_argument("--n-authors", type=int, default=None, help="random sample of whole authors (default: all)")
    p.add_argument("--n-posts", type=int, default=None, help="random sample of posts, for smoke tests")
    p.add_argument("--max-cost", type=float, default=5.0, help="hard spending cap in USD")
    p.add_argument("--max-features", type=int, default=None, help="first N features of the bank (default: all)")
    p.add_argument("--features-per-call", type=int, default=50, help="questions per decisions request")
    p.add_argument("--workers", type=int, default=32, help="parallel requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=None, help="override JEV_MODEL")
    p.add_argument("--out", type=Path, default=PROCESSED / "judgments")
    p.add_argument("--dry-run", action="store_true", help="print the first request body and the call count, then exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features = json.loads(args.bank.read_text())["features"][: args.max_features]
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "author"]]
    if args.split != "all":
        splits = pd.read_parquet(PROCESSED / "splits.parquet")
        labels = labels[labels["author"].isin(splits.loc[splits["split"] == args.split, "author"])]
    authors = sorted(labels["author"].unique())
    if args.n_authors is not None:
        authors = sorted(random.Random(args.seed).sample(authors, args.n_authors))
        labels = labels[labels["author"].isin(authors)]
    posts = (load_all().loc[lambda d: d["urn"].isin(labels["urn"]), ["urn", "media", "body"]]
             .sort_values("urn").to_dict("records"))
    if args.n_posts is not None:
        posts = random.Random(args.seed).sample(posts, args.n_posts)

    n_calls = len(posts) * -(-len(features) // args.features_per_call)
    estimate = judge.estimate_cost(len(posts), len(features))
    print(f"{len(authors):,} authors, {len(posts):,} posts x {len(features)} features = "
          f"{len(posts) * len(features):,} questions in {n_calls:,} requests. "
          f"Estimated cost ${estimate:.2f} (cap ${args.max_cost:.2f}).")
    if args.dry_run:
        chunk = features[: args.features_per_call]
        keys = judge.question_keys(chunk)
        body = {
            "model": LLMSettings.from_env("openrouter").model if args.model is None else args.model,
            "state": judge.post_state(posts[0]),
            "questions": {k: judge.noul_question(f) for k, f in zip(keys, chunk)},
        }
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return
    if estimate > args.max_cost:
        sys.exit(f"Estimated ${estimate:.2f} is over the ${args.max_cost:.2f} cap. "
                 "Use fewer --n-authors or raise --max-cost.")

    client = DecisionsClient(LLMSettings.from_env("openrouter", model=args.model),
                             max_connections=args.workers, max_cost_usd=args.max_cost)
    print(f"Judging with {client.settings.model}: {n_calls:,} requests, {args.workers} in parallel ...")
    start = time.time()
    step = max(1, n_calls // 20)

    def progress(done: int, total: int) -> None:
        if done % step == 0 or done == total:
            rate = done / (time.time() - start)
            print(f"  {done:,}/{total:,} requests  {rate:.1f}/s  eta {(total - done) / rate / 60:.1f} min", flush=True)

    rows, skipped = judge.judge(client.decide, posts, features, args.features_per_call, args.workers, progress)
    elapsed = time.time() - start
    if skipped:
        print(f"Budget cap reached: {skipped:,} requests not sent. Saving partial results.")

    long = pd.DataFrame(rows)
    wide = long.pivot(index="urn", columns="feature_id", values="p_yes")[[f["id"] for f in features]].reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(args.out / "judgments.parquet", index=False)
    wide.to_parquet(args.out / "features.parquet", index=False)
    meta = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "model": client.settings.model,
        "authors": len(authors),
        "posts": len(posts),
        "features": len(features),
        "requests": n_calls,
        "requests_skipped_budget": skipped,
        "estimated_cost_usd": round(estimate, 4),
        "judgments": len(long),
        "status": long["status"].value_counts().to_dict(),
        "seconds": round(elapsed, 1),
        "p_yes_mean_by_feature": long.groupby("feature_id")["p_yes"].mean().round(3).to_dict(),
        "usage": client.usage.to_dict(),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
