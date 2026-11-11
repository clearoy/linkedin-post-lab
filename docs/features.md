# Features: discovery (DeepSeek) and judging (Jev)

Two LLM stages turn post text into a table of P(yes) per post and feature.
Neither LLM decides what helps or hurts. The model fitted in step 4 does that,
using the `lift` label from [labels.md](labels.md).

```
labels.parquet ──► discover_features.py (DeepSeek) ──► feature_bank.json
                                                            │
posts + feature_bank ──► judge_features.py (Jev) ──► features.parquet (urn × feature P(yes))
```

## Stage 1: discovery

Code: [`discovery.py`](../src/linkedin_analysis/discovery.py),
prompts [`discover_system.md`](../src/linkedin_analysis/prompts/discover_system.md),
[`tag_system.md`](../src/linkedin_analysis/prompts/tag_system.md) and
[`merge_system.md`](../src/linkedin_analysis/prompts/merge_system.md).

1. **Sample same-author pairs.** Each pair is one post from the author's `top`
   third by lift and one from their `bottom` third. The two are shown in random
   order as "Post A" and "Post B", with no reactions, tiers, or names. Because
   the author is the same, the most visible differences are in the content.
   Showing contrasting pairs steers the LLM toward properties that actually vary,
   without telling it which side did better.
2. **Propose properties.** For each batch (default 5 pairs), DeepSeek proposes
   15 yes/no properties. The prompt requires each property to be:
   - observable from the text;
   - descriptive, never evaluative;
   - atomic, about one characteristic;
   - present in roughly 5–70% of posts;
   - about content, not identity or timing.

   It also states that properties which might help, hurt, or do nothing are
   equally useful, and that the LLM must not guess which. Each property
   carries a definition and invented yes/no examples.

   **No categories are given.** An earlier version listed 14 dimensions, and
   DeepSeek returned almost exactly one property per dimension, filling a
   checklist instead of reporting what differed. The prompt now asks for the
   differences it actually sees, several of the same kind if that is what
   stands out.

   **Content over surface form.** Without guidance, the first smoke run leaned
   toward surface checks (question marks, emoji, hashtags, URLs, pronouns),
   which a text search could answer. The prompt now asks for mostly content
   properties: topic, claims, stance, story, insight, framing, relationship to
   the reader. Surface-form properties are capped at about a quarter. This
   steers which kinds of property are proposed, not which direction they go.
3. **Lint.** Proposals with missing fields are dropped. Evaluative words (strong,
   engaging, compelling and so on) are flagged in `lint` for the merge step to
   fix.
4. **Tag.** A separate DeepSeek call at temperature 0 assigns each proposal to
   one of 14 dimensions, or `other`. This is used only to group likely
   duplicates for merging, never to steer discovery.
5. **Merge.** Within each dimension, DeepSeek merges duplicates at temperature 0. It enforces the rules again,
   rewriting or dropping violations, and must account for every input id.
   Large dimensions are merged in chunks, and the chunk results are merged again.
   Each feature records `support` (raw proposals merged into it) and
   `batch_support` (independent batches that proposed it).

**No train/test split here:** the properties describe language in general, and
the LLM never sees performance. The hidden tiers are saved in `pairs.jsonl`
only for later analysis.

## Stage 2: judging

Code: [`judge.py`](../src/linkedin_analysis/judge.py), client `DecisionsClient` in
[`llm.py`](../src/linkedin_analysis/llm.py).

Jev (`typesafe/jev-1.13`) is a *decisions* model. It is only reachable through
OpenRouter's `POST /api/alpha/decisions`, not through chat completions. Each
request carries:

- `state`: the post, `{"media_attached": ..., "post_text": ...}`. No author,
  reactions, or labels.
- `questions`: a chunk of features (default 50), each a `noul` (yes/no)
  question:
  - `instructions`: the feature's question, plus a note to judge only whether
    the definition is met, read literally, without considering quality or
    performance;
  - `criteria.true`: the definition and the yes example;
  - `criteria.false`: "does not meet that definition" and the no example.

Jev returns `answers[key].noul`, which is P(true), for every question. Refusals
or missing answers are stored as NaN with their `status`.

**Speed.** A single request answers many features for one post, so a full run
needs posts × ⌈features / 50⌉ requests. They run 32 at a time on a pooled HTTP
client, with retries and backoff on 429/5xx. Measured on 60 posts × 26
features: 60 requests in 1.6 s (about 1,000 judgments per second), about
$0.00016 per request. Responses are cached, so re-runs are free.

**Why not chat completions?** The first version sent one Yes/No question per
call to `typesafe/jev-router` and read logprobs. The router sent each call to a
different backend (GPT-6-luna, GPT-6.1-sol, Gemini 3.8 Flash, DeepSeek V4.1
Flash), and most returned no logprobs. The result was mixed judges and hard
0/1 answers, and it was not Jev.
