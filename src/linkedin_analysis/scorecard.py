"""Score a post with a research scorecard: one weight per yes/no question, plus length.

The scorecards are fitted by research/scorecard.py at several sizes (10, 20, 40 questions and
the full set). Each question has a weight on Jev's P(yes), with the direction fixed in advance.

    contribution of a question = weight * (answer - average post's answer)
    score = sum of contributions + length weight * (log(min(words, cap)) - average)

Reading the score
  * Two drafts by the same author: the one with the higher score is expected to get about
    exp(score difference) times the reactions (calibrated on held-out authors, slope about 1).
    A difference of 0.10 is about 10% more. `pair_confidence` says how often that held.
  * One post by an unknown author: expected reactions vs the author's usual post
    = exp(a + b * score), the size's pooled calibration (b is about 0.5, because part of
    any score is the author's habitual style, which the author-relative label removes).
    `band` gives what happened to held-out posts with a similar score.
"""

import json
import math
from pathlib import Path

import pandas as pd

from .paths import PROCESSED
from .textchecks import text_checks

BUNDLE = PROCESSED / "research" / "scorecard.json"


def load(path: Path = BUNDLE) -> dict:
    return json.loads(path.read_text())


def sizes(bundle: dict) -> list[str]:
    """Available scorecard sizes, smallest first."""
    return sorted(bundle["sizes"], key=int)


def jev_features(bundle: dict, size: str) -> list[dict]:
    """The yes/no questions Jev answers for one post with this scorecard (text checks are computed locally)."""
    need = {i["id"] for i in bundle["sizes"][size]["items"]}
    return [f for f in bundle["features"] if f["id"] in need]


def length_value(words: int, cap: int | None) -> float:
    words = max(words, 1)
    return math.log(min(words, cap) if cap else words)


def post_values(card: dict, p_yes: dict[str, float], text: str) -> dict[str, float]:
    """Every input of one scorecard for one post: Jev answers, free text checks and the word count.

    A missing or refused Jev answer falls back to the average post's value, so it adds nothing."""
    values = {i["id"]: i["mean"] for i in card["items"]}
    values.update({k: float(v) for k, v in p_yes.items() if k in values and v is not None and not math.isnan(v)})
    checks = text_checks(pd.Series([text])).iloc[0].to_dict()
    values.update({k: float(v) for k, v in checks.items() if k in values})
    values["words"] = len(text.split())
    return values


def score(card: dict, values: dict[str, float]) -> dict:
    """Score, expected multiplier, band, and one row per question (plus length) with its weight and contribution."""
    rows = []
    for i in card["items"]:
        v = values.get(i["id"], i["mean"])
        label = i["label"] + (" (text check)" if i["id"].startswith("text.") else "")
        rows.append({"id": i["id"], "label": label, "question": i["question"], "answer": v,
                     "weight": i["weight"], "mean": i["mean"], "contribution": i["weight"] * (v - i["mean"])})
    ln = card["length"]
    words = int(values["words"])
    rows.append({"id": "length", "label": f"Length (per doubling of words, counted up to {ln['cap']})",
                 "question": "Number of words", "answer": float(words), "weight": ln["weight"] * math.log(2),
                 "mean": math.exp(ln["mean"]), "contribution": ln["weight"] * (length_value(words, ln["cap"]) - ln["mean"])})
    total = sum(r["contribution"] for r in rows)
    cal = card["calibration"]
    return {"score": total, "multiplier": math.exp(cal["a"] + cal["b"] * total), "rows": rows, "words": words,
            "band": band(card, total)}


def band(card: dict, s: float) -> dict:
    """The score band, with what happened to held-out authors' posts in it."""
    for b in card["bands"]:
        if b["hi"] is None or s < b["hi"]:
            return b
    return card["bands"][-1]


def pair_confidence(card: dict, gap: float) -> dict:
    """How often, for two posts by the same held-out author with a similar score gap, the higher score did better."""
    for row in card["pair_gap"]:
        if row["hi"] is None or abs(gap) < row["hi"]:
            return row
    return card["pair_gap"][-1]
