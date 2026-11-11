"""LLM feature discovery: propose neutral, observable post properties.

Two stages:
  1. discover: show the LLM batches of same-author post pairs (one from the
     author's top third by lift, one from the bottom third, in random order,
     with no performance information) and ask for properties that vary.
  2. merge: group the raw proposals by dimension and have the LLM merge
     duplicates into one canonical feature bank, keeping track of how many
     independent proposals support each feature.

The LLM never sees performance and is told not to judge direction; that is
the job of the model fitted later.
"""

import json
import random
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from string import Template

import pandas as pd

DIMENSIONS: dict[str, str] = {
    "opening": "the first line or two, what the reader sees before 'see more'",
    "structure_format": "layout, line breaks, lists, sections, emojis, hashtags, formatting",
    "length_density": "length, sentence length, how much information is packed in",
    "voice_perspective": "first/second/third person, who is speaking, point of view",
    "tone_emotion": "emotional register, humor, vulnerability, urgency, formality",
    "narrative": "stories, anecdotes, scenes, characters, before/after arcs",
    "argument_claims": "opinions, contrarian takes, predictions, frameworks, reasoning",
    "specificity_evidence": "numbers, names of things, examples, data, concrete detail",
    "topic": "what the post is about: subject matter and domain",
    "audience_relationship": "who is addressed, direct address, community, shared identity",
    "call_to_action": "questions to readers, requests to comment, share, follow, sign up",
    "promotion_intent": "self-promotion, announcements, hiring, selling, launches",
    "media_links": "attached media, external links, mentions, references to other content",
    "language_style": "word choice, jargon, buzzwords, readability, rhetorical devices",
}

FIELDS = ("name", "dimension", "question", "definition", "yes_example", "no_example")

EVALUATIVE = re.compile(
    r"\b(good|bad|strong|weak|compelling|engaging|effective|clear|clearly|interesting|"
    r"high[- ]quality|professional|authentic|valuable|polished|viral|powerful|great|"
    r"excellent|poor|catchy|impactful|persuasive|relatable|insightful|well[- ]written|"
    r"attention[- ]grabbing)\b",
    re.IGNORECASE,
)


# Prompts


def load_prompt(name: str) -> Template:
    return Template((files("linkedin_analysis") / "prompts" / f"{name}.md").read_text())


def dimensions_block() -> str:
    return "\n".join(f"- `{k}`: {v}" for k, v in DIMENSIONS.items())


def discover_system_prompt(n_properties: int) -> str:
    return load_prompt("discover_system").substitute(
        dimensions=dimensions_block(), n_properties=n_properties
    )


def merge_system_prompt(dimension: str) -> str:
    return load_prompt("merge_system").substitute(dimension=dimension)


# Pair sampling


def sample_pairs(labeled: pd.DataFrame, n_pairs: int, seed: int = 0) -> list[dict]:
    """Pick same-author pairs: one `top` tier post and one `bottom` tier post.

    `labeled` needs urn, author, band, tier, media, body. Authors are drawn
    without replacement until every eligible author has been used once.
    Post order within each pair is random, so position reveals nothing.
    """
    rng = random.Random(seed)
    eligible = sorted(
        a for a, g in labeled.groupby("author") if {"top", "bottom"} <= set(g["tier"])
    )
    if not eligible:
        raise ValueError("no author has both top and bottom tier posts")

    authors: list[str] = []
    while len(authors) < n_pairs:
        authors += rng.sample(eligible, min(len(eligible), n_pairs - len(authors)))

    by_author = {a: g for a, g in labeled.groupby("author")}
    used: set[str] = set()
    pairs = []
    for i, author in enumerate(authors):
        g = by_author[author]
        chosen = []
        for tier in ("top", "bottom"):
            options = sorted(set(g.loc[g["tier"] == tier, "urn"]) - used) or sorted(
                g.loc[g["tier"] == tier, "urn"]
            )
            chosen.append(rng.choice(options))
        used.update(chosen)
        rng.shuffle(chosen)
        rows = g.set_index("urn").loc[chosen]
        pairs.append(
            {
                "pair_id": f"pair{i:04d}",
                "author": author,
                "band": rows["band"].iloc[0],
                "posts": [
                    {"urn": urn, "media": r["media"], "body": r["body"]} for urn, r in rows.iterrows()
                ],
                # Kept for later analysis only. Never sent to the LLM.
                "hidden_tiers": rows["tier"].to_dict(),
            }
        )
    return pairs


def format_post(media: str, body: str, max_chars: int = 3000) -> str:
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + " [...]"
    return f"Media attached: {media}\n\n{body}"


def discover_user_message(pairs: Sequence[dict]) -> str:
    parts = [f"Here are {len(pairs)} pairs of posts. Within each pair, both posts are by the same author."]
    for i, pair in enumerate(pairs, 1):
        a, b = pair["posts"]
        parts.append(
            f"=== Pair {i} ===\n\n--- Post {i}A ---\n{format_post(a['media'], a['body'])}"
            f"\n\n--- Post {i}B ---\n{format_post(b['media'], b['body'])}"
        )
    return "\n\n".join(parts)


# Validation


def lint_property(p: dict) -> list[str]:
    """Return problems with a proposed property. Empty list = clean."""
    issues = [f"missing {f}" for f in FIELDS if not str(p.get(f, "")).strip()]
    if p.get("dimension") not in DIMENSIONS:
        issues.append(f"unknown dimension {p.get('dimension')!r}")
    if not str(p.get("question", "")).strip().endswith("?"):
        issues.append("question does not end with '?'")
    words = sorted({m.lower() for m in EVALUATIVE.findall(f"{p.get('question', '')} {p.get('definition', '')}")})
    if words:
        issues.append(f"evaluative words: {', '.join(words)}")
    return issues


# Stage 1: discover

ChatFn = Callable[..., dict]


def batched(items: Sequence, size: int) -> list[list]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def discover(
    chat_json: ChatFn,
    pairs: Sequence[dict],
    pairs_per_batch: int = 5,
    n_properties: int = 15,
    temperature: float = 1.0,
    workers: int = 8,
) -> list[dict]:
    """Run discovery over batches of pairs and return flat raw proposals.

    Proposals missing required fields are dropped. Others keep a `lint` list,
    so the merge step can rewrite or drop evaluative wording.
    """
    system = discover_system_prompt(n_properties)
    batches = batched(pairs, pairs_per_batch)

    def run(args: tuple[int, list[dict]]) -> list[dict]:
        b, batch = args
        reply = chat_json(system, discover_user_message(batch), temperature=temperature)
        out = []
        for j, p in enumerate(reply.get("properties", [])):
            if not isinstance(p, dict):
                continue
            p = {f: str(p.get(f, "")).strip() for f in FIELDS}
            lint = lint_property(p)
            if any(i.startswith("missing") for i in lint):
                continue
            if p["dimension"] not in DIMENSIONS:
                p["dimension"] = "other"
            out.append(
                {"id": f"b{b:03d}-p{j:02d}", "batch": b, "pair_ids": [x["pair_id"] for x in batch], **p, "lint": lint}
            )
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, enumerate(batches)))
    return [p for batch in results for p in batch]


# Stage 2: merge


def _merge_call(chat_json: ChatFn, dimension: str, items: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """One merge call. Returns (properties with source_ids, dropped)."""
    payload = [{"id": x["id"], "name": x["name"], "question": x["question"], "definition": x["definition"]} for x in items]
    user = f"Dimension: {dimension}\n\nProposals ({len(payload)}):\n{json.dumps(payload, indent=1, ensure_ascii=False)}"
    reply = chat_json(merge_system_prompt(dimension), user, temperature=0.0)

    known = {x["id"] for x in items}
    props = []
    for p in reply.get("properties", []):
        ids = [i for i in p.get("source_ids", []) if i in known]
        if not ids:
            continue  # nothing real behind it: invented or all ids hallucinated
        props.append({**{f: str(p.get(f, "")).strip() for f in FIELDS}, "dimension": dimension, "source_ids": ids})
    dropped = [d for d in reply.get("dropped", []) if isinstance(d, dict) and d.get("id") in known]
    return props, dropped


def merge_dimension(chat_json: ChatFn, dimension: str, items: Sequence[dict], max_per_call: int = 120) -> tuple[list[dict], list[dict]]:
    """Merge one dimension's proposals, splitting into chunks when there are many.

    Chunk results are merged again, and source ids are expanded back to the
    raw proposal ids at every level.
    """
    if len(items) <= max_per_call:
        return _merge_call(chat_json, dimension, items)

    merged, dropped = [], []
    for c, chunk in enumerate(batched(items, max_per_call)):
        props, drops = _merge_call(chat_json, dimension, chunk)
        dropped += drops
        merged += [{**p, "id": f"{dimension}-c{c}-{k}"} for k, p in enumerate(props)]

    if len(merged) >= len(items):  # chunk merges did not shrink anything; stop recursing
        return [{k: v for k, v in m.items() if k != "id"} for m in merged], dropped

    expand = {m["id"]: m["source_ids"] for m in merged}
    props, drops = merge_dimension(chat_json, dimension, merged, max_per_call)
    for p in props:
        p["source_ids"] = [raw for mid in p["source_ids"] for raw in expand.get(mid, [mid])]
    for d in drops:
        dropped += [{"id": raw, "reason": d.get("reason", "")} for raw in expand.get(d["id"], [d["id"]])]
    return props, dropped


def merge(chat_json: ChatFn, proposals: Sequence[dict], max_per_call: int = 120, workers: int = 8) -> dict:
    """Merge raw proposals into a feature bank.

    Each feature gets `support` (raw proposals merged into it) and
    `batch_support` (distinct discovery batches that proposed it), a rough
    measure of how often it was noticed independently.
    """
    by_dim: dict[str, list[dict]] = {}
    for p in proposals:
        by_dim.setdefault(p["dimension"], []).append(p)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(by_dim, pool.map(lambda d: merge_dimension(chat_json, d, by_dim[d], max_per_call), by_dim)))

    batch_of = {p["id"]: p["batch"] for p in proposals}
    features, dropped, seen_names = [], [], set()
    for dimension, (props, drops) in results.items():
        dropped += [{**d, "dimension": dimension} for d in drops]
        for p in props:
            name = re.sub(r"[^a-z0-9_]+", "_", p["name"].lower()).strip("_") or "feature"
            fid, k = f"{dimension}.{name}", 2
            while fid in seen_names:
                fid, k = f"{dimension}.{name}_{k}", k + 1
            seen_names.add(fid)
            features.append(
                {
                    "id": fid,
                    **{f: p[f] for f in FIELDS},
                    "support": len(p["source_ids"]),
                    "batch_support": len({batch_of[i] for i in p["source_ids"]}),
                    "lint": lint_property(p),
                    "source_ids": p["source_ids"],
                }
            )

    features.sort(key=lambda f: (list(DIMENSIONS).index(f["dimension"]) if f["dimension"] in DIMENSIONS else 99, -f["support"]))
    accounted = {i for f in features for i in f["source_ids"]} | {d["id"] for d in dropped}
    return {
        "features": features,
        "dropped": dropped,
        "unaccounted_ids": sorted({p["id"] for p in proposals} - accounted),
    }
