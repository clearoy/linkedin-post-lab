"""Generate new content hypotheses from the posts the current model explains worst.

Usage:
    uv run python research/hypotheses.py --round 1

1. Splits train authors once (seeded) into three disjoint sets, saved to
   data/processed/research/authors.json:
     generate  DeepSeek reads these authors' posts to propose hypotheses
     screen    Jev scores the candidates on these authors; cheap first test
     rest      the remaining train authors, scored only for candidates that pass
   Test authors are never shown to DeepSeek and never used to pick anything.
2. Fits the joint model (joint_existing.json columns) on train authors and takes
   within-author residuals. For each generation author it pairs the post that
   most beat the model with the one that most fell short.
3. DeepSeek proposes properties for batches of pairs (prompts/hypothesize.md),
   then merges them into one list (prompts/select.md). The hand-written
   seeds.json (round 1) / seeds_round<N>.json hypotheses are added on top.
   Round N pairs each author's N-th best and N-th worst post, and every
   earlier round's candidates are listed as already measured.

Writes data/processed/research/bank_round<N>.json (Jev feature-bank format)
and research/results/hypotheses_round<N>.json (questions only, no post text).
"""

import argparse
import json
import random
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from joint import frame  # noqa: E402
from stats import EXISTING, PROCESSED, RESEARCH_DATA, RESULTS, YES, demean, within_ols  # noqa: E402

from linkedin_analysis.discovery import lint_property  # noqa: E402
from linkedin_analysis.llm import LLMClient, LLMSettings  # noqa: E402

HERE = Path(__file__).resolve().parent
SIZES = {"generate": 150, "screen": 250}


def author_sets(train_authors: list[str]) -> dict[str, list[str]]:
    path = RESEARCH_DATA / "authors.json"
    if path.exists():
        return json.loads(path.read_text())
    a = sorted(train_authors)
    random.Random(2026).shuffle(a)
    n1, n2 = SIZES["generate"], SIZES["generate"] + SIZES["screen"]
    sets = {"generate": sorted(a[:n1]), "screen": sorted(a[n1:n2]), "rest": sorted(a[n2:])}
    RESEARCH_DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sets, indent=1))
    return sets


def post_text(body: str, max_chars: int = 2500) -> str:
    body = body.strip()
    return body if len(body) <= max_chars else body[:max_chars].rstrip() + " [...]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--joint", default="existing", help="joint_<name>.json gives the model whose misses are studied")
    p.add_argument("--judgments", type=Path, nargs="+", default=EXISTING)
    p.add_argument("--pairs-per-call", type=int, default=25)
    p.add_argument("--per-call", type=int, default=12)
    p.add_argument("--keep", type=int, default=42, help="DeepSeek hypotheses kept after merging")
    args = p.parse_args()

    df = frame(args.judgments)
    sets = author_sets(df.loc[df["split"] == "train", "author"].unique().tolist())
    cols = json.loads((RESULTS / f"joint_{args.joint}.json").read_text())["columns"]
    Z = np.column_stack([(df[[c for c in cols if c != "log_words"]] >= YES).to_numpy(float),
                         np.log1p(df["words"].to_numpy())])
    tr = (df["split"] == "train").to_numpy()
    beta = within_ols(df.loc[tr, "lift"].to_numpy(), Z[tr], df.loc[tr, "author"].to_numpy())["beta"]
    a = df["author"].to_numpy()
    df["resid"] = demean(df["lift"].to_numpy(), a) - demean(Z, a) @ np.nan_to_num(beta)

    gen = df[df["author"].isin(sets["generate"])]
    rng = random.Random(args.round)
    pairs = []
    for author, g in gen.groupby("author"):
        if len(g) < 4:
            continue
        # round N shows each author's N-th best and N-th worst post vs the model, so rounds see different posts
        order = g.sort_values("resid")
        hi, lo = order.iloc[-args.round], order.iloc[args.round - 1]
        pairs.append((hi, lo))
    rng.shuffle(pairs)
    print(f"{len(pairs)} pairs from {len(sets['generate'])} generation authors", flush=True)

    bank = json.loads((PROCESSED / "features" / "feature_bank_general.json").read_text())["features"]
    for r in range(1, args.round):  # earlier rounds' candidates count as measured too
        bank += json.loads((RESEARCH_DATA / f"bank_round{r}.json").read_text())["features"]
    measured = "\n".join(f"- {f['question']}" for f in bank) + (
        "\n- Simple text checks: length, links, hashtags, emoji, lists, line breaks, question marks.")
    client = LLMClient(LLMSettings.from_env("deepseek"))
    system = Template((HERE / "prompts" / "hypothesize.md").read_text()).substitute(n=args.per_call)

    proposals = []
    for i in range(0, len(pairs), args.pairs_per_call):
        chunk = pairs[i:i + args.pairs_per_call]
        parts = [f"Already measured (do not repeat):\n{measured}\n\nHere are {len(chunk)} pairs."]
        for k, (hi, lo) in enumerate(chunk, 1):
            first_hi = rng.random() < 0.5
            posts = [("better than expected", hi), ("worse than expected", lo)]
            if not first_hi:
                posts.reverse()
            parts.append(f"=== Pair {k} ===\n" + "\n\n".join(
                f"--- Post {k}{'AB'[j]} ({tag}) ---\n{post_text(r['body'])}" for j, (tag, r) in enumerate(posts)))
        reply = client.chat_json(system, "\n\n".join(parts), temperature=0.3)
        got = [x for x in reply.get("properties", []) if not lint_property(x, check_dimension=False)]
        proposals += got
        print(f"  batch {i // args.pairs_per_call + 1}: {len(got)} proposals", flush=True)

    select = Template((HERE / "prompts" / "select.md").read_text()).substitute(n=args.keep)
    reply = client.chat_json(select, f"Already measured:\n{measured}\n\nProposals:\n" + json.dumps(proposals, indent=1))
    auto = [x for x in reply.get("properties", []) if not lint_property(x, check_dimension=False)][: args.keep]
    seed_file = HERE / ("seeds.json" if args.round == 1 else f"seeds_round{args.round}.json")
    seeds = json.loads(seed_file.read_text())["properties"] if seed_file.exists() else []

    features, seen = [], set()
    for src, items in (("seed", seeds), ("auto", auto)):
        for x in items:
            fid = f"r{args.round}_{src}.{x['name']}"
            if fid in seen:
                continue
            seen.add(fid)
            features.append({"id": fid, "name": x["name"], "dimension": src, "question": x["question"],
                             "definition": x["definition"], "yes_example": x["yes_example"],
                             "no_example": x["no_example"], "hypothesis": x.get("hypothesis", "")})
    out = RESEARCH_DATA / f"bank_round{args.round}.json"
    out.write_text(json.dumps({"features": features, "pairs": len(pairs), "proposals": len(proposals)}, indent=1))
    (RESULTS / f"hypotheses_round{args.round}.json").write_text(json.dumps(
        {"round": args.round, "pairs_shown": len(pairs), "raw_proposals": len(proposals), "features": features}, indent=1))
    print(f"{len(proposals)} raw proposals -> {len(auto)} DeepSeek + {len(seeds)} seeds = {len(features)} candidates")
    print(f"DeepSeek usage: {client.usage.to_dict()}")
    for f in features:
        print(f"  {f['id']}: {f['question']}")


if __name__ == "__main__":
    main()
