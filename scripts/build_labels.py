"""Build author-relative labels for every post.

Usage:
    uv run python scripts/build_labels.py [--min-posts 5] [--window 12] ...

Writes:
    data/processed/labels.parquet       one row per labeled post, keyed by `urn`
    data/processed/labels_summary.json  counts, config and sanity checks
"""

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable without relying on the editable install's .pth file,
# which macOS keeps flagging as hidden (Python 3.13+ then skips it).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.labels import LabelConfig, compute_labels, summarize  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402


def parse_args() -> LabelConfig:
    defaults = LabelConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, value in defaults.to_dict().items():
        p.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return LabelConfig(**vars(p.parse_args()))


def main() -> None:
    config = parse_args()
    posts = load_all()
    labels = compute_labels(posts, config)
    summary = {"config": config.to_dict(), **summarize(labels, posts)}

    PROCESSED.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(PROCESSED / "labels.parquet", index=False)
    (PROCESSED / "labels_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {len(labels):,} labels to {PROCESSED / 'labels.parquet'}")


if __name__ == "__main__":
    main()
