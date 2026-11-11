import numpy as np
import pandas as pd
import pytest

from linkedin_analysis.labels import LabelConfig, compute_labels, local_baseline


def make_posts(author: str, reactions: list[int], start: str = "2026-08-01") -> pd.DataFrame:
    n = len(reactions)
    return pd.DataFrame(
        {
            "urn": [f"{author}-{i}" for i in range(n)],
            "author": author,
            "posted": pd.date_range(start, periods=n, freq="D", tz="UTC"),
            "reactions": reactions,
        }
    )


# local_baseline


def test_baseline_excludes_the_post_itself():
    out = local_baseline(np.array([100, 1, 1, 1, 1]), window=12)
    assert out[0] == 1  # the 100 is not in its own baseline
    assert out[1] == 1  # median of [100, 1, 1, 1]


def test_baseline_is_leave_one_out_median_when_window_covers_everything():
    r = np.array([5, 9, 2, 7, 4, 8])
    expected = [np.median(np.delete(r, i)) for i in range(len(r))]
    np.testing.assert_allclose(local_baseline(r, window=12), expected)


def test_baseline_uses_nearest_neighbors_in_time():
    # Early posts get ~0 reactions, late posts ~100; a small window should follow that.
    r = np.array([0, 0, 0, 0, 100, 100, 100, 100])
    out = local_baseline(r, window=2)
    assert out[0] == 0
    assert out[-1] == 100


# compute_labels


def test_drops_authors_below_min_posts():
    posts = pd.concat([make_posts("a", [1, 2, 3, 4, 5]), make_posts("b", [1, 2, 3])])
    labels = compute_labels(posts, LabelConfig(min_posts=5))
    assert set(labels.author) == {"a"}


def test_typical_post_has_zero_lift():
    labels = compute_labels(make_posts("a", [10, 10, 10, 10, 10]))
    np.testing.assert_allclose(labels.lift, 0)
    assert not labels.win.any()


def test_win_threshold():
    labels = compute_labels(make_posts("a", [8, 8, 8, 8, 18]), LabelConfig(smoothing=2, win_ratio=2))
    # (18 + 2) / (8 + 2) = 2.0 exactly
    assert labels.set_index("urn").loc["a-4", "win"]
    assert labels.win.sum() == 1


def test_labels_ignore_author_scale():
    pattern = [1, 3, 2, 8, 4, 2]
    small = make_posts("small", [x * 10 for x in pattern])
    big = make_posts("big", [x * 1000 for x in pattern])
    labels = compute_labels(pd.concat([small, big]))
    s = labels[labels.author == "small"].lift_pct.to_numpy()
    b = labels[labels.author == "big"].lift_pct.to_numpy()
    np.testing.assert_allclose(s, b)


def test_tiers_are_symmetric():
    labels = compute_labels(make_posts("a", [1, 2, 3, 4, 5, 6]))
    counts = labels.tier.value_counts()
    assert counts["top"] == counts["bottom"] == 2
    assert labels.loc[labels.reactions == 6, "tier"].item() == "top"
    assert labels.loc[labels.reactions == 1, "tier"].item() == "bottom"


def test_input_order_does_not_matter():
    posts = make_posts("a", [5, 1, 9, 3, 7, 2, 8])
    a = compute_labels(posts).set_index("urn").sort_index()
    b = compute_labels(posts.sample(frac=1, random_state=0)).set_index("urn").sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_rejects_duplicate_urns():
    posts = make_posts("a", [1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="duplicate"):
        compute_labels(pd.concat([posts, posts.iloc[[0]]]))
