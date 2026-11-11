import numpy as np
import pandas as pd

from linkedin_analysis.splits import make_splits


def make_labels(n_authors: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_authors):
        n = int(rng.integers(5, 30))
        scale = rng.lognormal(2, 1)
        band = "1k-10k" if a % 3 else "10k-20k"
        for i in range(n):
            rows.append({"urn": f"a{a}-{i}", "author": f"a{a}", "band": band,
                         "reactions": int(rng.poisson(scale))})
    return pd.DataFrame(rows)


def test_each_author_once_and_test_share():
    labels = make_labels()
    s = make_splits(labels, test_share=0.2, n_folds=5)
    assert s["author"].is_unique and len(s) == labels["author"].nunique()
    assert abs((s["split"] == "test").mean() - 0.2) < 0.05


def test_folds_only_for_train_and_balanced():
    s = make_splits(make_labels(400), n_folds=5)
    assert (s.loc[s.split == "test", "fold"] == -1).all()
    counts = s.loc[s.split == "train", "fold"].value_counts()
    assert sorted(counts.index) == [0, 1, 2, 3, 4]
    assert counts.max() - counts.min() <= 6


def test_both_bands_in_test():
    s = make_splits(make_labels())
    assert set(s.loc[s.split == "test", "band"]) == {"1k-10k", "10k-20k"}


def test_deterministic_by_seed():
    labels = make_labels()
    pd.testing.assert_frame_equal(make_splits(labels, seed=3), make_splits(labels, seed=3))
    assert not make_splits(labels, seed=3)["split"].equals(make_splits(labels, seed=4)["split"])
