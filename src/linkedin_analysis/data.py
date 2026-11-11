"""Load the raw posts parquet files."""

import pandas as pd

from .paths import BANDS, GCS_POSTS, RAW


def load_posts(band: str) -> pd.DataFrame:
    """Load one follower band and add a `band` column."""
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}, expected one of {BANDS}")
    path = RAW / f"posts_{band}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Download it with:\n"
            f'  gcloud storage cp "{GCS_POSTS}*.parquet" {RAW}/'
        )
    return pd.read_parquet(path).assign(band=band)


def load_all() -> pd.DataFrame:
    """Load every band into one frame. Authors never appear in two bands."""
    return pd.concat([load_posts(b) for b in BANDS], ignore_index=True)
