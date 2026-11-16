"""Score a post with the research scorecard and say what would change the score.

The scorecard is an additive model fitted by research/scorecard.py. Every content
property that passed discovery on train authors has its own weight on Jev's
P(yes), with the direction fixed in advance (helpful properties can only add,
harmful ones can only subtract), plus a length term. Properties are grouped into
named drivers for display.

    score = sum over items of weight * (value - average post's value)
            + length weight * (log(min(words, cap)) - average)

Reading the score
  * Two drafts by the same author: the one with the higher score is expected to
    get about exp(score difference) times the reactions (calibrated on held-out
    authors, slope about 1). `pair_confidence` says how often that held.
  * One post by an unknown author: expected reactions vs the author's usual post
    = exp(a + b * score), the pooled calibration in the bundle (b is about 0.5,
    because part of any score is the author's habitual style, which the
    author-relative label removes). `band` gives what happened to held-out posts
    with a similar score.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import PROCESSED
from .textchecks import text_checks

BUNDLE = PROCESSED / "research" / "scorecard.json"
YES = 0.5


def load(path: Path = BUNDLE) -> dict:
    return json.loads(path.read_text())


def jev_features(bundle: dict) -> list[dict]:
    """The yes/no questions Jev answers for one post (scorecard items plus post-type questions)."""
    return bundle["features"]


def length_value(words: int, cap: int | None) -> float:
    words = max(words, 1)
    return math.log(min(words, cap) if cap else words)


def post_values(bundle: dict, p_yes: dict[str, float], text: str) -> dict[str, float]:
    """Every input of the scorecard for one post: Jev answers, free text checks and the word count.

    A missing or refused Jev answer falls back to the average post's value, so it adds nothing."""
    values = {i["id"]: i["mean"] for i in bundle["items"]}
    values.update({k: float(v) for k, v in p_yes.items() if v is not None and not math.isnan(v)})
    values.update(text_checks(pd.Series([text])).iloc[0].to_dict())
    values["words"] = len(text.split())
    return values


def score(bundle: dict, values: dict[str, float]) -> dict:
    """Score, expected multiplier, band, and the contribution of every item and driver."""
    items = []
    for i in bundle["items"]:
        v = values.get(i["id"], i["mean"])
        items.append({**i, "value": v, "has": v >= YES, "contribution": i["weight"] * (v - i["mean"])})
    ln = bundle["length"]
    words = int(values["words"])
    drivers = []
    for d in bundle["drivers"]:
        mine = [i for i in items if i["driver"] == d["id"]]
        drivers.append({**d, "contribution": sum(i["contribution"] for i in mine), "items": mine,
                        "has": [i for i in mine if i["has"]]})
    drivers.append({"id": "length", "name": "Length", "direction": "+", "items": [], "has": [],
                    "contribution": ln["weight"] * (length_value(words, ln["cap"]) - ln["mean"]),
                    "summary": f"{words} words. More words help up to about {ln['cap']}; past that, length alone adds nothing.",
                    "advice": f"Give it at least {ln['cap']} words."})
    total = sum(d["contribution"] for d in drivers)
    cal = bundle["calibration"]
    return {"score": total, "multiplier": math.exp(cal["a"] + cal["b"] * total), "drivers": drivers, "items": items,
            "words": words, "band": band(bundle, total)}


def band(bundle: dict, s: float) -> dict:
    """The score band, with what happened to held-out authors' posts in it."""
    for b in bundle["bands"]:
        if b["hi"] is None or s < b["hi"]:
            return b
    return bundle["bands"][-1]


def suggestions(bundle: dict, result: dict, min_gain: float = 0.01) -> dict[str, list[dict]]:
    """What would raise the score, largest gain first.

    `edits`: writing choices. Add a confirmed helpful property the post lacks, or remove a
      confirmed harmful one it has. The gain is the scorecard's own weight, so applying the
      edit and re-scoring moves the score by about that much (exp(gain) for the same author).
    `if_true`: real occasions (a milestone, an event, today's news) that help only if they
      actually happened; shown as "if this is true, say it".
    """
    edits, if_true = [], []
    for i in result["items"]:
        if i["single"]["status"] != "confirmed" or i["kind"] == "descr":
            continue
        if i["weight"] > 0 and not i["has"]:
            gain, action = i["weight"] * (1 - i["value"]), "add"
        elif i["weight"] < 0 and i["has"]:
            gain, action = -i["weight"] * i["value"], "remove"
        else:
            continue
        (if_true if i["kind"] == "occasion" else edits).append({**i, "action": action, "gain": gain})
    edits, if_true = _merge(edits), _merge(if_true)
    edits = [e for e in edits if e["gain"] >= min_gain]
    if_true = [e for e in if_true if e["gain"] >= min_gain]
    ln, words = bundle["length"], result["words"]
    if ln["cap"] and words < ln["cap"]:
        gain = ln["weight"] * (math.log(ln["cap"]) - math.log(max(words, 1)))
        if gain >= min_gain:
            edits.append({"id": "length", "label": f"Write at least {ln['cap']} words (now {words})", "kind": "write",
                          "driver": "length", "action": "add", "gain": gain, "gain_multiplier": math.exp(gain),
                          "weight": ln["weight"], "single": None})
    return {"edits": sorted(edits, key=lambda s: -s["gain"]), "if_true": sorted(if_true, key=lambda s: -s["gain"])}


def _merge(suggested: list[dict]) -> list[dict]:
    """One suggestion per label: when Jev and a text check measure the same thing (a link), one edit changes both,
    so their gains add. The evidence shown is the item with the larger weight."""
    out: dict[tuple[str, str], dict] = {}
    for s in suggested:
        key = (s["label"], s["action"])
        if key not in out:
            out[key] = dict(s)
            continue
        m = out[key]
        if abs(s["weight"]) > abs(m["weight"]):
            m.update({k: v for k, v in s.items() if k != "gain"})
        m["gain"] += s["gain"]
    for m in out.values():
        m["gain_multiplier"] = math.exp(m["gain"])
    return list(out.values())


def post_type(bundle: dict, values: dict[str, float]) -> dict:
    """The post type whose cluster centre is nearest to this post's answers."""
    t = bundle["types"]
    x = np.array([values.get(q, 0.0) for q in t["questions"]])
    names = list(t["centres"])
    d = [float(((x - np.array(t["centres"][n])) ** 2).sum()) for n in names]
    name = names[int(np.argmin(d))]
    return {"name": name, **t["table"][name]}


def pair_confidence(bundle: dict, gap: float) -> dict:
    """How often, for two posts by the same held-out author with a similar score gap, the higher score did better."""
    for row in bundle["pair_gap"]:
        if row["hi"] is None or abs(gap) < row["hi"]:
            return row
    return bundle["pair_gap"][-1]
