import math

from linkedin_analysis import scorecard as sc


def item(id_, weight, mean):
    return {"id": id_, "label": id_, "question": id_ + "?", "weight": weight, "mean": mean}


def card():
    return {
        "items": [item("good", 0.3, 0.2), item("bad", -0.2, 0.4), item("text.has_link", -0.1, 0.3)],
        "length": {"weight": 0.1, "cap": 120, "mean": math.log(100)},
        "calibration": {"a": 0.05, "b": 0.5},
        "bands": [{"name": "Low", "lo": None, "hi": 0.0}, {"name": "High", "lo": 0.0, "hi": None}],
        "pair_gap": [{"lo": 0.0, "hi": 0.5, "accuracy": 0.55}, {"lo": 0.5, "hi": None, "accuracy": 0.8}],
    }


WORDS_100 = " ".join(["w"] * 100)


def test_average_post_scores_zero():
    c = card()
    values = sc.post_values(c, {}, WORDS_100)  # every Jev answer missing -> average
    values["text.has_link"] = 0.3  # text check set to the average too
    r = sc.score(c, values)
    assert abs(r["score"]) < 1e-9
    assert math.isclose(r["multiplier"], math.exp(0.05))
    assert r["band"]["name"] == "High"


def test_score_is_sum_of_contributions():
    c = card()
    r = sc.score(c, sc.post_values(c, {"good": 1.0, "bad": 0.0}, WORDS_100 + " https://x.com"))
    by = {row["id"]: row for row in r["rows"]}
    assert math.isclose(by["good"]["contribution"], 0.3 * 0.8)
    assert math.isclose(by["bad"]["contribution"], -0.2 * -0.4)
    assert math.isclose(by["text.has_link"]["contribution"], -0.1 * 0.7)  # text check computed from the text
    assert math.isclose(r["score"], sum(row["contribution"] for row in r["rows"]))


def test_answers_outside_the_card_are_ignored():
    c = card()
    values = sc.post_values(c, {"good": 1.0, "other": 1.0}, WORDS_100)
    assert "other" not in values


def test_length_is_capped_and_shown_per_doubling():
    c = card()
    long = sc.score(c, sc.post_values(c, {}, " ".join(["w"] * 500)))
    capped = sc.score(c, sc.post_values(c, {}, " ".join(["w"] * 120)))
    assert math.isclose(long["score"], capped["score"])
    ln = next(row for row in long["rows"] if row["id"] == "length")
    assert math.isclose(ln["weight"], 0.1 * math.log(2))


def test_jev_features_only_for_the_chosen_size():
    b = {"sizes": {"2": {"items": [item("good", 0.3, 0.2), item("text.has_link", -0.1, 0.3)]},
                   "3": card()},
         "features": [{"id": "good"}, {"id": "bad"}]}
    assert sc.sizes(b) == ["2", "3"]
    assert [f["id"] for f in sc.jev_features(b, "2")] == ["good"]
    assert [f["id"] for f in sc.jev_features(b, "3")] == ["good", "bad"]


def test_pair_confidence():
    assert sc.pair_confidence(card(), -0.7)["accuracy"] == 0.8
    assert sc.pair_confidence(card(), 0.2)["accuracy"] == 0.55
