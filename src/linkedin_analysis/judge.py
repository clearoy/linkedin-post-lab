"""Jev judge: P(yes) that a post has each feature from the feature bank.

The judge only decides whether a property is present. It never sees
performance and is told not to consider it.

Two modes:
  logprob  one Yes/No question per call; P(yes) comes from the probabilities
           of the first answer token (normalized over Yes and No). Most precise,
           one call per post x feature.
  batch    one call per post and chunk of features; the model returns a
           probability for each feature as JSON. Far fewer calls, less
           calibrated.
"""

import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from .discovery import batched, format_post, load_prompt
from .llm import parse_json

YES = {"yes", "y"}
NO = {"no", "n"}


def judge_system_prompt() -> str:
    return load_prompt("judge_system").template


def judge_batch_system_prompt() -> str:
    return load_prompt("judge_batch_system").template


def post_block(media: str, body: str) -> str:
    return f"<post>\n{format_post(media, body, max_chars=6000)}\n</post>"


def judge_user_message(post: dict, feature: dict) -> str:
    # The post comes first so providers can cache the shared prefix across features.
    return (
        f"{post_block(post['media'], post['body'])}\n\n"
        f"Question: {feature['question']}\n"
        f"Definition of yes: {feature['definition']}\n"
        f"Example of yes: {feature['yes_example']}\n"
        f"Example of no: {feature['no_example']}\n\n"
        "Answer Yes or No."
    )


def judge_batch_user_message(post: dict, features: Sequence[dict]) -> str:
    lines = [post_block(post["media"], post["body"]), "", "Properties:"]
    for f in features:
        lines.append(
            f"- id: {f['id']}\n  question: {f['question']}\n  definition of yes: {f['definition']}"
            f"\n  example of yes: {f['yes_example']}\n  example of no: {f['no_example']}"
        )
    return "\n".join(lines)


def p_yes_from_response(response: dict) -> tuple[float, str, str]:
    """Return (p_yes, method, answer_text) from a chat completion dict.

    Uses the first token's top logprobs when present. Falls back to the
    answer text (1.0 or 0.0) when the provider returns no logprobs.
    """
    choice = response["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    content = ((choice.get("logprobs") or {}).get("content")) or []
    if content:
        p_yes = p_no = 0.0
        for alt in content[0].get("top_logprobs") or []:
            token = alt["token"].strip().lower().strip(".,!\"'")
            if token in YES:
                p_yes += math.exp(alt["logprob"])
            elif token in NO:
                p_no += math.exp(alt["logprob"])
        if p_yes + p_no > 0:
            return p_yes / (p_yes + p_no), "logprob", text
    word = text.lower().strip(".,!\"' ").split(" ")[0] if text else ""
    if word in YES:
        return 1.0, "text", text
    if word in NO:
        return 0.0, "text", text
    return math.nan, "unparsed", text


def judge_logprob(complete: Callable[[dict], dict], posts: Sequence[dict], features: Sequence[dict],
                  workers: int = 16, reasoning: dict | None = None) -> list[dict]:
    """One call per post x feature. Returns long rows: urn, feature_id, p_yes, method, answer."""
    system = judge_system_prompt()

    def run(job: tuple[dict, dict]) -> dict:
        post, feature = job
        request = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": judge_user_message(post, feature)}],
            "temperature": 0.0,
            "max_tokens": 5,
            "logprobs": True,
            "top_logprobs": 10,
        }
        if reasoning is not None:
            request["extra_body"] = {"reasoning": reasoning}
        p, method, text = p_yes_from_response(complete(request))
        return {"urn": post["urn"], "feature_id": feature["id"], "p_yes": p, "method": method, "answer": text}

    jobs = [(p, f) for p in posts for f in features]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, jobs))


def judge_batch(complete: Callable[..., dict], posts: Sequence[dict], features: Sequence[dict],
                features_per_call: int = 25, workers: int = 16, reasoning: dict | None = None) -> list[dict]:
    """One call per post x chunk of features, verbalized probabilities."""
    system = judge_batch_system_prompt()

    def validate(response: dict, ids: list[str]) -> None:
        answers = parse_json(response["choices"][0]["message"]["content"] or "").get("answers", {})
        missing = [i for i in ids if i not in answers]
        if missing:
            raise ValueError(f"answers missing for {missing[:3]}")

    def run(job: tuple[dict, list[dict]]) -> list[dict]:
        post, chunk = job
        ids = [f["id"] for f in chunk]
        request = {
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": judge_batch_user_message(post, chunk)}],
            "temperature": 0.0,
            "max_tokens": 40 * len(chunk) + 200,
            "response_format": {"type": "json_object"},
        }
        if reasoning is not None:
            request["extra_body"] = {"reasoning": reasoning}
        response = complete(request, validate=lambda r: validate(r, ids))
        answers = parse_json(response["choices"][0]["message"]["content"])["answers"]
        rows = []
        for fid in ids:
            try:
                p = min(max(float(answers[fid]), 0.0), 1.0)
            except (TypeError, ValueError):
                p = math.nan
            rows.append({"urn": post["urn"], "feature_id": fid, "p_yes": p, "method": "batch", "answer": str(answers[fid])})
        return rows

    jobs = [(p, chunk) for p in posts for chunk in batched(features, features_per_call)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [row for rows in pool.map(run, jobs) for row in rows]
