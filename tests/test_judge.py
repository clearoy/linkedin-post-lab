import json
import math

import pytest

from linkedin_analysis.judge import answer_p_yes, judge, noul_question, post_state, question_keys

POST = {"urn": "u1", "media": "image", "body": "We shipped v2 today. What should we build next?"}


def feature(i: int) -> dict:
    return {"id": f"call_to_action.f{i}", "question": f"Does the post do thing {i}?",
            "definition": f"Yes when the post does thing {i}.", "yes_example": "What do you think?",
            "no_example": "We shipped it."}


def test_question_is_noul_with_definition_and_judge_note():
    q = noul_question(feature(1))
    assert q["type"] == "noul"
    assert "Does the post do thing 1?" in q["instructions"]
    assert "Do not consider whether the post is good" in q["instructions"]
    assert set(q["criteria"]) == {"true", "false"}
    assert "Yes when the post does thing 1." in q["criteria"]["true"]


def test_state_has_post_only_and_no_performance_info():
    state = post_state(POST)
    assert state == {"post_text": POST["body"]}  # text only, no attached media
    assert "u1" not in json.dumps(state)


def test_answer_parsing():
    assert answer_p_yes({"type": "noul", "noul": 0.96}) == (0.96, "ok")
    assert answer_p_yes({"type": "noul", "noul": 1.3}) == (1.0, "ok")
    p, status = answer_p_yes({"type": "refusal"})
    assert math.isnan(p) and status == "refusal"
    assert answer_p_yes(None)[1] == "missing"


def test_judge_batches_features_per_post_and_maps_keys_back():
    calls = []

    def fake(state, questions, validate):
        calls.append(list(questions))
        response = {"answers": {k: {"type": "noul", "noul": 0.1 * (int(k[1:]) + 1)} for k in questions}}
        validate(response)
        return response

    features = [feature(i) for i in range(5)]
    rows, skipped = judge(fake, [POST, {**POST, "urn": "u2"}], features, features_per_call=2, workers=1)
    assert skipped == 0
    assert len(calls) == 6  # 2 posts x 3 chunks
    assert sorted(calls)[0] == ["q000"] and ["q000", "q001"] in calls
    assert len(rows) == 10
    p = {(r["urn"], r["feature_id"]): r["p_yes"] for r in rows}
    assert p[("u1", "call_to_action.f0")] == pytest.approx(0.1)
    assert p[("u2", "call_to_action.f3")] == pytest.approx(0.2)  # second in its chunk -> q001
    assert p[("u2", "call_to_action.f4")] == pytest.approx(0.1)  # alone in its chunk -> q000


def test_judge_validation_rejects_missing_answers():
    def fake(state, questions, validate):
        validate({"answers": {}})

    with pytest.raises(KeyError, match="missing"):
        judge(fake, [POST], [feature(0)], workers=1)


def test_question_keys_are_short_and_unique():
    keys = question_keys([feature(i) for i in range(12)])
    assert list(keys)[:2] == ["q000", "q001"] and len(set(keys)) == 12


def test_budget_stop_keeps_finished_rows():
    from linkedin_analysis.llm import BudgetExceeded

    calls = []

    def fake(state, questions, validate):
        calls.append(1)
        if len(calls) > 2:
            raise BudgetExceeded("cap")
        return {"answers": {k: {"type": "noul", "noul": 0.5} for k in questions}}

    posts = [{**POST, "urn": f"u{i}"} for i in range(5)]
    rows, skipped = judge(fake, posts, [feature(0)], workers=1)
    assert len(rows) == 2 and skipped == 3


def test_client_refuses_requests_after_cap(tmp_path):
    import pytest as _pytest

    from linkedin_analysis.llm import BudgetExceeded, DecisionsClient, LLMSettings

    client = DecisionsClient(LLMSettings("openrouter", "k", "m", "https://example.invalid/api/v1"),
                             cache_dir=tmp_path, max_cost_usd=1.0)
    client.usage.cost_usd = 1.0
    with _pytest.raises(BudgetExceeded):
        client.decide({"post_text": "x"}, {"q000": {"type": "noul"}})
