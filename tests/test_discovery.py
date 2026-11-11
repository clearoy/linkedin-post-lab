import json
import re

import pandas as pd

from linkedin_analysis.discovery import (
    DIMENSIONS,
    discover,
    discover_system_prompt,
    discover_user_message,
    lint_property,
    merge,
    merge_system_prompt,
    sample_pairs,
)


def make_labeled(n_authors: int = 4, per_author: int = 6) -> pd.DataFrame:
    rows = []
    for a in range(n_authors):
        for i in range(per_author):
            tier = "top" if i < 2 else "bottom" if i >= per_author - 2 else "middle"
            rows.append(
                {"urn": f"u{a}-{i}", "author": f"author{a}", "band": "1k-10k", "tier": tier,
                 "media": "text", "body": f"body {a}-{i}"}
            )
    return pd.DataFrame(rows)


def prop(name: str, dimension: str = "opening", **kw) -> dict:
    return {"name": name, "dimension": dimension, "question": f"Does the post {name}?",
            "definition": f"Yes when the post {name}.", "yes_example": "x", "no_example": "y", **kw}


# Prompts


def test_system_prompts_render_fully():
    s = discover_system_prompt(15)
    assert "$" not in s and "exactly 15 properties" in s
    assert all(f"`{d}`" in s for d in DIMENSIONS)
    assert "$" not in merge_system_prompt("opening")


# Pair sampling


def test_pairs_are_same_author_top_and_bottom():
    labeled = make_labeled()
    pairs = sample_pairs(labeled, n_pairs=4, seed=1)
    tier = labeled.set_index("urn")["tier"]
    author = labeled.set_index("urn")["author"]
    for p in pairs:
        urns = [x["urn"] for x in p["posts"]]
        assert {author[u] for u in urns} == {p["author"]}
        assert sorted(tier[u] for u in urns) == ["bottom", "top"]
    assert len({p["author"] for p in pairs}) == 4  # every author once before repeating


def test_pairs_are_deterministic_and_order_is_shuffled():
    labeled = make_labeled(n_authors=30)
    a = sample_pairs(labeled, 30, seed=7)
    assert a == sample_pairs(labeled, 30, seed=7)
    first_is_top = [p["hidden_tiers"][p["posts"][0]["urn"]] == "top" for p in a]
    assert 0 < sum(first_is_top) < len(first_is_top)


def test_more_pairs_than_authors_reuses_authors():
    assert len(sample_pairs(make_labeled(n_authors=3), n_pairs=7)) == 7


def test_user_message_reveals_no_performance_info():
    pairs = sample_pairs(make_labeled(), 2)
    msg = discover_user_message(pairs).lower()
    for leak in ("top", "bottom", "tier", "lift", "reaction", "u0-", "author0"):
        assert leak not in msg
    assert "post 1a" in msg and "post 2b" in msg


# Lint


def test_lint_flags_evaluative_and_structural_problems():
    assert lint_property(prop("opens_with_question")) == []
    issues = lint_property({**prop("x"), "question": "Is the hook strong?", "dimension": "vibes"})
    assert any("evaluative" in i for i in issues)
    assert any("unknown dimension" in i for i in issues)
    assert lint_property({**prop("x"), "question": "Hook present"}) == ["question does not end with '?'"]


# Discover


def test_discover_collects_and_cleans_proposals():
    pairs = sample_pairs(make_labeled(n_authors=4), 4)

    def fake(system, user, temperature):
        return {"properties": [prop("a"), prop("b", dimension="made_up"), {"name": "incomplete"}, "junk"]}

    out = discover(fake, pairs, pairs_per_batch=2, workers=1)
    assert len(out) == 4  # 2 batches x 2 valid proposals
    assert {p["id"] for p in out} == {"b000-p00", "b000-p01", "b001-p00", "b001-p01"}
    assert {p["dimension"] for p in out} == {"opening", "other"}
    assert out[0]["pair_ids"] == ["pair0000", "pair0001"]


# Merge


def proposals(n: int, dimension: str = "opening") -> list[dict]:
    return [{**prop(f"p{i}", dimension), "id": f"b{i:03d}-p00", "batch": i} for i in range(n)]


def merge_ids_from(user: str) -> list[str]:
    return [x["id"] for x in json.loads(user.split(":\n", 1)[1])]


def test_merge_keeps_provenance_and_ignores_invented_ids():
    def fake(system, user, temperature):
        ids = merge_ids_from(user)
        return {
            "properties": [
                {**prop("merged"), "source_ids": ids[:3]},
                {**prop("invented"), "source_ids": ["nope"]},
            ],
            "dropped": [{"id": ids[3], "reason": "evaluative"}],
        }

    bank = merge(fake, proposals(5), workers=1)
    [f] = bank["features"]
    assert f["id"] == "opening.merged"
    assert f["support"] == 3 and f["batch_support"] == 3
    assert bank["dropped"][0]["id"] == "b003-p00"
    assert bank["unaccounted_ids"] == ["b004-p00"]


def test_merge_chunks_large_dimensions_and_expands_ids():
    calls = []

    def fake(system, user, temperature):
        ids = merge_ids_from(user)
        calls.append(len(ids))
        # Merge everything in each call into a single property.
        return {"properties": [{**prop(f"m{len(calls)}"), "source_ids": ids}], "dropped": []}

    bank = merge(fake, proposals(25), max_per_call=10, workers=1)
    assert calls == [10, 10, 5, 3]  # three chunks, then one merge of the chunk results
    [f] = bank["features"]
    assert f["support"] == 25
    assert sorted(f["source_ids"]) == sorted(p["id"] for p in proposals(25))
    assert bank["unaccounted_ids"] == []


def test_feature_ids_are_unique():
    def fake(system, user, temperature):
        ids = merge_ids_from(user)
        return {"properties": [{**prop("same"), "source_ids": [i]} for i in ids]}

    ids = [f["id"] for f in merge(fake, proposals(3), workers=1)["features"]]
    assert len(set(ids)) == 3 and all(re.fullmatch(r"opening\.same(_\d)?", i) for i in ids)
