"""Discover candidate post properties with DeepSeek and merge them into a feature bank.

Usage:
    uv run python scripts/discover_features.py                 # full run
    uv run python scripts/discover_features.py --dry-run       # print prompts, no API calls
    uv run python scripts/discover_features.py --n-pairs 10 --out data/processed/features_smoke

Needs data/processed/labels.parquet (scripts/build_labels.py) and
DEEPSEEK_API_KEY in .env.

Writes to --out (default data/processed/features/):
    pairs.jsonl          sampled pairs, including the hidden tiers (never sent to the LLM)
    raw_proposals.jsonl  every property proposed, with batch and lint notes
    feature_bank.json    merged features with support counts and provenance
    run_meta.json        config, counts, token usage
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import pandas as pd  # noqa: E402

from linkedin_analysis import discovery  # noqa: E402
from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.llm import LLMClient, LLMSettings  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-pairs", type=int, default=200, help="same-author pairs to show in total")
    p.add_argument("--pairs-per-batch", type=int, default=5, help="pairs per discovery call")
    p.add_argument("--n-properties", type=int, default=15, help="properties requested per call")
    p.add_argument("--temperature", type=float, default=1.0, help="discovery temperature (merge uses 0)")
    p.add_argument("--max-per-merge", type=int, default=120, help="proposals per merge call")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default="train", choices=["train", "all"],
                   help="sample pairs only from train authors (default) so test authors stay unseen")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default=None, help="override DEEPSEEK_MODEL")
    p.add_argument("--out", type=Path, default=PROCESSED / "features")
    p.add_argument("--dry-run", action="store_true", help="print the first prompt and exit")
    return p.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows))


def main() -> None:
    args = parse_args()
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "tier"]]
    labeled = load_all().merge(labels, on="urn")
    if args.split == "train":
        splits = pd.read_parquet(PROCESSED / "splits.parquet")
        labeled = labeled[labeled["author"].isin(splits.loc[splits["split"] == "train", "author"])]
    pairs = discovery.sample_pairs(labeled, args.n_pairs, seed=args.seed)
    batches = discovery.batched(pairs, args.pairs_per_batch)

    if args.dry_run:
        system = discovery.discover_system_prompt(args.n_properties)
        user = discovery.discover_user_message(batches[0])
        print("=== SYSTEM ===\n" + system + "\n=== USER (batch 0) ===\n" + user)
        print(f"\n{len(batches)} discovery calls, about {(len(system) + len(user)) // 4:,} input tokens each")
        return

    client = LLMClient(LLMSettings.from_env("deepseek", model=args.model))
    print(f"Discovering with {client.settings.model}: {len(batches)} calls ...")
    proposals = discovery.discover(
        client.chat_json, pairs, args.pairs_per_batch, args.n_properties, args.temperature, args.workers
    )
    print(f"  {len(proposals)} raw proposals. Tagging dimensions ...")
    proposals = discovery.tag(client.chat_json, proposals, workers=args.workers)
    print("  Merging ...")
    bank = discovery.merge(client.chat_json, proposals, args.max_per_merge, args.workers)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "pairs.jsonl", pairs)
    write_jsonl(args.out / "raw_proposals.jsonl", proposals)
    (args.out / "feature_bank.json").write_text(json.dumps(bank, indent=2, ensure_ascii=False))
    meta = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "model": client.settings.model,
        "pairs": len(pairs),
        "authors": len({p["author"] for p in pairs}),
        "split": args.split,
        "raw_proposals": len(proposals),
        "raw_with_lint_issues": sum(bool(p["lint"]) for p in proposals),
        "raw_by_dimension": pd.Series([p["dimension"] for p in proposals]).value_counts().to_dict(),
        "features": len(bank["features"]),
        "features_with_lint_issues": sum(bool(f["lint"]) for f in bank["features"]),
        "dropped": len(bank["dropped"]),
        "unaccounted": len(bank["unaccounted_ids"]),
        "usage": client.usage.to_dict(),
    }
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
