"""Project paths and dataset constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

BANDS = ("1k-10k", "10k-20k")

GCS_POSTS = "gs://prototypes-498523-linkedin-post-model/2026-10-19/posts/"
