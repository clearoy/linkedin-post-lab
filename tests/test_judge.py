import json
import math

import pytest

from linkedin_analysis.judge import (
    judge_batch,
    judge_logprob,
    judge_user_message,
    p_yes_from_response,
)

POST = {"urn": "u1", "media": "image", "body": "We shipped v2 today. What should we build next?"}
FEATURE = {"id": "call_to_action.asks_question", "question": "Does the post ask readers a question?",
           "definition": "Yes when the post directly asks readers something.",
           "yes_example": "What do you think?", "no_example": "We shipped it."}


def response(content: str, top: list[tuple[str, float]] | None = None) -> dict:
    logprobs = None
    if top is not None:
        logprobs = {"content": [{"token": top[0][0], "logprob": math.log(top[0][1]),
                                 "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top]}]}
    return {"choices": [{"message": {"content": content}, "logprobs": logprobs}]}


# p_yes_from_response


def test_logprobs_are_normalized_over_yes_and_no():
    p, method, _ = p_yes_from_response(response("Yes", [("Yes", 0.6), (" yes", 0.1), ("No", 0.2), ("Maybe", 0.1)]))
    assert method == "logprob"
    assert p == pytest.approx(0.7 / 0.9)


def test_falls_back_to_text_without_logprobs():
    assert p_yes_from_response(response("No.")) == (0.0, "text", "No.")
    assert p_yes_from_response(response("Yes")) == (1.0, "text", "Yes")
    p, method, _ = p_yes_from_response(response("I cannot tell"))
    assert method == "unparsed" and math.isnan(p)


# Prompts


def test_user_message_has_post_then_question_and_no_performance_info():
    msg = judge_user_message(POST, FEATURE)
    assert msg.index("<post>") < msg.index("Question:")
    assert FEATURE["definition"] in msg
    for leak in ("reaction", "lift", "tier", "u1"):
        assert leak not in msg.lower()


# Modes


def test_logprob_mode_sends_one_question_per_call():
    requests = []

    def fake(request):
        requests.append(request)
        return response("Yes", [("Yes", 0.9), ("No", 0.1)])

    rows = judge_logprob(fake, [POST, {**POST, "urn": "u2"}], [FEATURE, {**FEATURE, "id": "f2"}], workers=1,
                         reasoning={"effort": "minimal"})
    assert len(rows) == len(requests) == 4
    assert requests[0]["logprobs"] is True and requests[0]["temperature"] == 0
    assert requests[0]["extra_body"] == {"reasoning": {"effort": "minimal"}}
    assert {r["feature_id"] for r in rows} == {FEATURE["id"], "f2"}
    assert all(r["p_yes"] == pytest.approx(0.9) for r in rows)


def test_batch_mode_chunks_features_validates_and_clamps():
    features = [{**FEATURE, "id": f"f{i}"} for i in range(5)]
    calls = []

    def fake(request, validate):
        ids = [line.split("id: ")[1] for line in request["messages"][1]["content"].splitlines() if "- id: " in line]
        calls.append(ids)
        r = {"choices": [{"message": {"content": json.dumps({"answers": {i: 1.7 if i == "f0" else 0.25 for i in ids}})}}]}
        validate(r)
        return r

    rows = judge_batch(fake, [POST], features, features_per_call=2, workers=1)
    assert calls == [["f0", "f1"], ["f2", "f3"], ["f4"]]
    p = {r["feature_id"]: r["p_yes"] for r in rows}
    assert p["f0"] == 1.0 and p["f3"] == 0.25


def test_batch_validation_rejects_missing_answers():
    def fake(request, validate):
        r = {"choices": [{"message": {"content": json.dumps({"answers": {}})}}]}
        validate(r)
        return r

    with pytest.raises(ValueError, match="missing"):
        judge_batch(fake, [POST], [FEATURE], workers=1)
