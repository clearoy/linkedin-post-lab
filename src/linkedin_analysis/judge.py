"""Jev judge: P(yes) that a post has each feature from the feature bank.

Uses OpenRouter's decisions endpoint, the only way to call Jev. Each request
sends one post as `state` and a chunk of features as `noul` (yes/no)
questions, and Jev returns P(true) for each. One request covers many
features, and requests run in parallel, which is what makes this fast.

The judge only decides whether a property is present as defined. It never
sees performance and each question tells it not to consider quality.
"""

import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from .discovery import batched

JUDGE_NOTE = (
    "Judge only whether the post meets this definition, reading it literally and using only "
    "the post's text and media type. Do not consider whether the post is good or how it might perform."
)

DecideFn = Callable[..., dict]


def post_state(post: dict, max_chars: int = 6000) -> dict:
    body = post["body"].strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + " [...]"
    return {"media_attached": post["media"], "post_text": body}


def noul_question(feature: dict) -> dict:
    """A feature as a decisions `noul` (yes/no) question."""
    return {
        "type": "noul",
        "instructions": f"{feature['question']} {JUDGE_NOTE}",
        "criteria": {
            "true": f"{feature['definition']} Example: {feature['yes_example']}",
            "false": f"The post does not meet that definition. Example: {feature['no_example']}",
        },
    }


def question_keys(features: Sequence[dict]) -> dict[str, str]:
    """Short, safe question keys (q000, q001, ...) mapped to feature ids."""
    return {f"q{i:03d}": f["id"] for i, f in enumerate(features)}


def answer_p_yes(answer: dict | None) -> tuple[float, str]:
    """(p_yes, status) from one decisions answer. Refusals and gaps give NaN."""
    if not answer or answer.get("type") != "noul":
        return math.nan, (answer or {}).get("type", "missing")
    try:
        return min(max(float(answer["noul"]), 0.0), 1.0), "ok"
    except (KeyError, TypeError, ValueError):
        return math.nan, "invalid"


def judge(decide: DecideFn, posts: Sequence[dict], features: Sequence[dict],
          features_per_call: int = 50, workers: int = 32, progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Score every post on every feature.

    Calls: len(posts) x ceil(len(features) / features_per_call), run `workers`
    at a time. Returns long rows: urn, feature_id, p_yes, status.
    """
    chunks = batched(list(features), features_per_call)

    def validate(response: dict, keys: Sequence[str]) -> None:
        missing = [k for k in keys if k not in response.get("answers", {})]
        if missing:
            raise KeyError(f"answers missing for {missing[:3]}")

    def run(job: tuple[dict, list[dict]]) -> list[dict]:
        post, chunk = job
        keys = question_keys(chunk)
        questions = {k: noul_question(f) for k, f in zip(keys, chunk)}
        response = decide(post_state(post), questions, validate=lambda r: validate(r, list(keys)))
        rows = []
        for k, fid in keys.items():
            p, status = answer_p_yes(response["answers"].get(k))
            rows.append({"urn": post["urn"], "feature_id": fid, "p_yes": p, "status": status})
        return rows

    jobs = [(p, c) for p in posts for c in chunks]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, j) for j in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            rows += future.result()
            if progress is not None:
                progress(done, len(jobs))
    return rows
