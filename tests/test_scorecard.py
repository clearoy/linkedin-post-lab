import math

from linkedin_analysis import scorecard as sc


def item(id_, weight, mean, kind="write", status="confirmed", driver="d"):
    return {"id": id_, "label": id_, "weight": weight, "mean": mean, "kind": kind, "driver": driver,
            "single": {"status": status}}


def bundle():
    return {
        "items": [item("good", 0.3, 0.2), item("bad", -0.2, 0.4), item("event", 0.4, 0.1, kind="occasion"),
                  item("tag", 0.1, 0.5, status="not replicated"), item("topic", 0.2, 0.3, kind="descr")],
        "drivers": [{"id": "d", "name": "D", "direction": "+", "summary": "", "advice": ""}],
        "length": {"weight": 0.1, "cap": 120, "mean": math.log(100)},
        "calibration": {"a": 0.05, "b": 0.5},
        "bands": [{"name": "Low", "lo": None, "hi": 0.0}, {"name": "High", "lo": 0.0, "hi": None}],
        "pair_gap": [{"lo": 0.0, "hi": 0.5, "accuracy": 0.55}, {"lo": 0.5, "hi": None, "accuracy": 0.8}],
        "types": {"questions": ["good", "bad"], "centres": {"A": [1.0, 0.0], "B": [0.0, 1.0]},
                  "table": {"A": {"description": "a"}, "B": {"description": "b"}}},
    }


TEXT = " ".join(["word"] * 100) + "\n\nsecond paragraph"


def test_average_post_scores_zero():
    b = bundle()
    values = sc.post_values(b, {}, " ".join(["w"] * 100))  # every answer missing -> average
    r = sc.score(b, values)
    assert abs(r["score"]) < 1e-9
    assert math.isclose(r["multiplier"], math.exp(0.05))
    assert r["band"]["name"] == "High"


def test_score_adds_weight_times_difference_from_average():
    b = bundle()
    r = sc.score(b, sc.post_values(b, {"good": 1.0, "bad": 0.0, "event": 0.1, "tag": 0.5, "topic": 0.3}, " ".join(["w"] * 100)))
    assert math.isclose(r["score"], 0.3 * 0.8 + (-0.2) * (-0.4))


def test_length_is_capped():
    b = bundle()
    long = sc.score(b, sc.post_values(b, {}, " ".join(["w"] * 500)))
    capped = sc.score(b, sc.post_values(b, {}, " ".join(["w"] * 120)))
    assert math.isclose(long["score"], capped["score"])


def test_suggestions_only_confirmed_writing_choices():
    b = bundle()
    r = sc.score(b, sc.post_values(b, {"good": 0.0, "bad": 1.0, "event": 0.0, "tag": 0.0, "topic": 0.0}, " ".join(["w"] * 40)))
    s = sc.suggestions(b, r)
    edits = [e["id"] for e in s["edits"]]
    assert edits[0] == "good" and "bad" in edits and "length" in edits
    assert "tag" not in edits and "topic" not in edits  # not replicated / not a writing choice
    assert [e["id"] for e in s["if_true"]] == ["event"]
    bad = next(e for e in s["edits"] if e["id"] == "bad")
    assert bad["action"] == "remove" and math.isclose(bad["gain"], 0.2)


def test_applying_a_suggestion_moves_the_score_by_its_gain():
    b = bundle()
    before = sc.score(b, sc.post_values(b, {"good": 0.0}, TEXT))
    gain = next(e for e in sc.suggestions(b, before)["edits"] if e["id"] == "good")["gain"]
    after = sc.score(b, sc.post_values(b, {"good": 1.0}, TEXT))
    assert math.isclose(after["score"] - before["score"], gain)


def test_post_type_and_pair_confidence():
    b = bundle()
    assert sc.post_type(b, {"good": 0.9, "bad": 0.1})["name"] == "A"
    assert sc.pair_confidence(b, -0.7)["accuracy"] == 0.8


def test_same_label_suggestions_merge_and_gains_add():
    b = bundle()
    b["items"] += [item("link_jev", -0.1, 0.3), item("link_text", -0.05, 0.3)]
    for i in b["items"][-2:]:
        i["label"] = "Includes a link"
    r = sc.score(b, sc.post_values(b, {"link_jev": 1.0, "link_text": 1.0}, TEXT))
    links = [e for e in sc.suggestions(b, r)["edits"] if e["label"] == "Includes a link"]
    assert len(links) == 1 and math.isclose(links[0]["gain"], 0.15) and links[0]["id"] == "link_jev"
