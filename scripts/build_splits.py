"""Split authors into train (80%, with 5 CV folds) and test (20%).

Usage:
    uv run python scripts/build_splits.py [--test-share 0.2] [--n-folds 5] [--seed 0]

Needs data/processed/labels.parquet. Writes:
    data/processed/splits.parquet       one row per author: split, fold, stratum
    data/processed/splits_summary.json  authors, posts, mean lift, win rate per split and fold
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import pandas as pd  # noqa: E402

from linkedin_analysis.paths import PROCESSED  # noqa: E402
from linkedin_analysis.splits import make_splits, summarize_splits  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-share", type=float, default=0.2)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    splits = make_splits(labels, args.test_share, args.n_folds, args.seed)
    summary = {"config": vars(args), **summarize_splits(splits, labels)}

    splits.to_parquet(PROCESSED / "splits.parquet", index=False)
    (PROCESSED / "splits_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
