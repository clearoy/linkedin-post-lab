"""Yes/no checks on a post's text that need no LLM (used by the research scripts and the dashboard)."""

import re

import pandas as pd

EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
URL = re.compile(r"https?://|www\.|lnkd\.in", re.I)
HASHTAG = re.compile(r"(?:^|\s)#\w+")
BULLET = re.compile(r"^\s*(?:[-•▪►✅✔→➡️👉]|\d+[.)])\s+", re.M)


def text_checks(body: pd.Series) -> pd.DataFrame:
    """Simple yes/no checks a text search can answer. Named `text.*`."""
    b = body.fillna("")
    first = b.str.strip().str.split("\n").str[0].fillna("")
    words = b.str.split().str.len().fillna(0)
    paras = b.str.split(r"\n\s*\n").apply(lambda ps: [p for p in ps if p.strip()])
    words_per_para = words / paras.str.len().clip(lower=1)
    return pd.DataFrame({
        "text.under_80_words": words < 80,
        "text.over_250_words": words > 250,
        "text.has_link": b.str.contains(URL),
        "text.has_hashtag": b.str.contains(HASHTAG),
        "text.three_plus_hashtags": b.str.count(HASHTAG) >= 3,
        "text.has_emoji": b.str.contains(EMOJI),
        "text.three_plus_emoji": b.str.count(EMOJI) >= 3,
        "text.has_bullet_or_numbered_list": b.str.count(BULLET) >= 2,
        "text.first_line_under_60_chars": first.str.len() < 60,
        "text.first_line_is_question": first.str.strip().str.endswith("?"),
        "text.first_line_has_number": first.str.contains(r"\d"),
        "text.ends_with_question": b.str.strip().str.endswith("?"),
        "text.has_exclamation": b.str.contains("!"),
        "text.short_paragraphs": (words_per_para <= 20) & (paras.str.len() >= 3),
        "text.single_block": paras.str.len() <= 1,
    }).astype(float)
