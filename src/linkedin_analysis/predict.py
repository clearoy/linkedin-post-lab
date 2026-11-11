"""Score a new post from Jev P(yes) values with the saved model bundle (JSON).

Used by the dashboard. Controls that are unknown for a draft post (posting
time) are set to their training mean, so they add nothing to the prediction.
"""

import json
import math
from pathlib import Path

import numpy as np

from .model import MEDIA, word_count
from .paths import PROCESSED

BUNDLE = PROCESSED / "models" / "model_bundle.json"


def load_bundle(path: Path = BUNDLE) -> dict:
    return json.loads(path.read_text())


def build_row(bundle: dict, p_yes: dict[str, float], media: str, text: str) -> dict[str, float]:
    """Full input row: Jev features, known controls, training means for the rest."""
    row = dict(bundle["column_means"])
    clean = {k: v for k, v in p_yes.items() if v is not None and not math.isnan(v)}
    row.update({k: v for k, v in clean.items() if k in row})
    for t, theme in bundle.get("themes", {}).items():  # theme score = mean P(yes) of its members
        vals = [clean[m] for m in theme["members"] if m in clean]
        if vals and t in row:
            row[t] = sum(vals) / len(vals)
    if "ctl.log_words" in row:  # only when the models were trained with controls
        for m in MEDIA:
            row[f"ctl.media_{m}"] = float(media == m)
        row["ctl.log_words"] = math.log1p(word_count(text))
    return row


def predict_linear(bundle: dict, row: dict[str, float]) -> tuple[float, list[dict]]:
    """Prediction and each term's contribution relative to the average training post."""
    lin = bundle["linear"]
    pred = lin["intercept"] + sum(c * row[k] for k, c in lin["coef"].items())
    contrib = [
        {"column": k, "coef": c, "value": row[k], "contribution": c * (row[k] - bundle["column_means"][k])}
        for k, c in lin["coef"].items()
    ]
    contrib.sort(key=lambda d: -abs(d["contribution"]))
    return pred, contrib


def _rule_holds(conditions: list, row: dict[str, float]) -> bool:
    return all((row[c] <= t) if op == "<=" else (row[c] > t) for c, op, t in conditions)


def predict_rulefit(bundle: dict, row: dict[str, float]) -> tuple[float, list[dict]]:
    """Prediction and the non-zero rules that fire for this post."""
    rf = bundle["rulefit"]
    pred = rf["intercept"]
    fired = []
    for rule in rf["rules"]:
        if _rule_holds(rule["conditions"], row):
            pred += rule["coef"]
            fired.append(rule)
    for col, (coef, mean, scale) in rf["linear"].items():
        pred += coef * (row[col] - mean) / scale
    fired.sort(key=lambda r: -abs(r["coef"]))
    return pred, fired


def multiplier(lift: float) -> float:
    """lift is log((reactions + s) / (baseline + s)); exp gives the approximate ratio."""
    return float(np.exp(lift))
