You are consolidating candidate properties for a study of LinkedIn posts. The properties were proposed independently in many batches, so many of them describe the same characteristic in different words.

Each property is a yes/no question about a single post. They will be used as features in a statistical model, which will decide later whether and how each one matters. You are not judging whether any property is good or bad for a post.

## Your task

1. **Merge duplicates.** Proposals that ask about the same observable characteristic, even when worded differently, become one canonical property. List every merged proposal's id in `source_ids`.
2. **Keep distinct things distinct.** Do not merge two properties just because they are related or belong to the same dimension. "Opens with a question" and "Ends with a question" are different properties. When one proposal is a narrower version of another, keep both only if both are likely to vary across posts; otherwise keep the broader one.
3. **Enforce the rules.** Every canonical property must be:
   - answerable yes/no from the post text and media type alone, without outside knowledge;
   - descriptive, never evaluative: no judging words such as good, strong, compelling, engaging, effective, clear, authentic, or high-quality, and no synonyms;
   - atomic, about one characteristic;
   - about content and form, not a specific person, company, or product, and not about timing or engagement.
   Rewrite a proposal to fix a violation when the underlying idea is sound. If the idea cannot be fixed, drop it and give the reason.
4. **Do not invent new properties.** Every canonical property must come from at least one proposal.
5. **Write clean definitions.** For each canonical property, write the clearest question, a one-sentence definition of "yes", and short invented yes/no examples (20 words or fewer each).

Every input id must appear exactly once: either in one property's `source_ids` or in `dropped`.

## Output

```json
{
  "properties": [
    {
      "name": "short_snake_case_name",
      "dimension": "$dimension",
      "question": "Yes/no question about a single post?",
      "definition": "One sentence defining when the answer is yes.",
      "yes_example": "Short invented snippet.",
      "no_example": "Short invented snippet.",
      "source_ids": ["id1", "id2"]
    }
  ],
  "dropped": [{"id": "id3", "reason": "evaluative: asks whether the hook is strong"}]
}
```

Return only the JSON object.
