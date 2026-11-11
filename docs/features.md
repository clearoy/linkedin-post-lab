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
prompts [`discover_system.md`](../src/linkedin_analysis/prompts/discover_system.md) and
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
3. **Lint.** Proposals with missing fields are dropped. Evaluative words (strong,
   engaging, compelling and so on) are flagged in `lint` for the merge step to
   fix.
4. **Merge.** Proposals are grouped by dimension (14 fixed dimensions) and
   DeepSeek merges duplicates at temperature 0. It enforces the rules again,
   rewriting or dropping violations, and must account for every input id.
   Large dimensions are merged in chunks, and the chunk results are merged again.
   Each feature records `support` (raw proposals merged into it) and
   `batch_support` (independent batches that proposed it).

**No train/test split here:** the properties describe language in general, and
the LLM never sees performance. The hidden tiers are saved in `pairs.jsonl`
only for later analysis.

## Stage 2: judging

Code: [`judge.py`](../src/linkedin_analysis/judge.py),
prompts [`judge_system.md`](../src/linkedin_analysis/prompts/judge_system.md) and
[`judge_batch_system.md`](../src/linkedin_analysis/prompts/judge_batch_system.md).

Jev reads one post and decides only whether each property is present, as
defined. It is told not to consider quality or performance.

| Mode | Calls | How P(yes) is computed |
|---|---|---|
| `logprob` (default) | posts × features | One Yes/No question per call. P(yes) = P(Yes tokens) / (P(Yes) + P(No)) from the first token's top logprobs. Falls back to 1/0 from the text when the provider returns no logprobs (`method = text`). |
| `batch` | posts × ⌈features / 25⌉ | Several properties per call. Jev returns a probability for each as JSON. Much cheaper, less calibrated. |

The post comes before the question in each message, so providers that cache
prompts can reuse it across that post's features.
