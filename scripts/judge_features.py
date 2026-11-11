"""Score posts on every feature in a feature bank with Jev (via OpenRouter).

Usage:
    uv run python scripts/judge_features.py --dry-run                          # print a prompt, no calls
    uv run python scripts/judge_features.py --n-posts 3 --max-features 5 \
        --bank data/processed/features_smoke/feature_bank.json --out data/processed/judgments_smoke
    uv run python scripts/judge_features.py                                     # every labeled post

Needs OPENROUTER_API_KEY in .env, data/processed/labels.parquet, and a
feature bank from scripts/discover_features.py.

Writes to --out (default data/processed/judgments/):
    judgments.parquet  long: urn, feature_id, p_yes, method, answer
    features.parquet   wide: one row per urn, one P(yes) column per feature
    run_meta.json      config, counts, method breakdown, token usage and cost
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import pandas as pd  # noqa: E402

from linkedin_analysis import judge  # noqa: E402
from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.llm import LLMClient, LLMSettings  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, default=PROCESSED / "features" / "feature_bank.json")
    p.add_argument("--mode", choices=["logprob", "batch"], default="logprob")
    p.add_argument("--n-posts", type=int, default=None, help="random sample of labeled posts (default: all)")
    p.add_argument("--max-features", type=int, default=None, help="first N features of the bank (default: all)")
    p.add_argument("--features-per-call", type=int, default=25, help="batch mode only")
    p.add_argument("--reasoning-effort", default=None,
                   help="OpenRouter reasoning effort to send, e.g. minimal or low (default: not sent)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--model", default=None, help="override JEV_MODEL")
    p.add_argument("--out", type=Path, default=PROCESSED / "judgments")
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and the call count, then exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features = json.loads(args.bank.read_text())["features"][: args.max_features]
    urns = pd.read_parquet(PROCESSED / "labels.parquet")["urn"]
    posts_df = load_all().loc[lambda d: d["urn"].isin(urns), ["urn", "media", "body"]].sort_values("urn")
    posts = posts_df.to_dict("records")
    if args.n_posts is not None:
        posts = random.Random(args.seed).sample(posts, args.n_posts)

    n_calls = len(posts) * (len(features) if args.mode == "logprob" else -(-len(features) // args.features_per_call))
    if args.dry_run:
        if args.mode == "logprob":
            system, user = judge.judge_system_prompt(), judge.judge_user_message(posts[0], features[0])
        else:
            system = judge.judge_batch_system_prompt()
            user = judge.judge_batch_user_message(posts[0], features[: args.features_per_call])
        print("=== SYSTEM ===\n" + system + "\n=== USER ===\n" + user)
        print(f"\n{len(posts):,} posts x {len(features)} features = {n_calls:,} calls in {args.mode} mode")
        return

    client = LLMClient(LLMSettings.from_env("openrouter", model=args.model))
    reasoning = {"effort": args.reasoning_effort} if args.reasoning_effort else None
    print(f"Judging with {client.settings.model} ({args.mode}): {n_calls:,} calls ...")
    if args.mode == "logprob":
        rows = judge.judge_logprob(client.complete, posts, features, args.workers, reasoning)
    else:
        rows = judge.judge_batch(client.complete, posts, features, args.features_per_call, args.workers, reasoning)

    long = pd.DataFrame(rows)
    wide = long.pivot(index="urn", columns="feature_id", values="p_yes")[[f["id"] for f in features]].reset_index()
    args.out.mkdir(parents=True, exist_ok=True)
    long.to_parquet(args.out / "judgments.parquet", index=False)
    wide.to_parquet(args.out / "features.parquet", index=False)
    meta = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "model": client.settings.model,
        "posts": len(posts),
        "features": len(features),
        "judgments": len(long),
        "methods": long["method"].value_counts().to_dict(),
        "p_yes_mean_by_feature": long.groupby("feature_id")["p_yes"].mean().round(3).to_dict(),
        "usage": client.usage.to_dict(),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
