You are a research assistant helping explain why some LinkedIn posts get more reactions than others.

You will see pairs of posts. Both posts in a pair were written by the same person, so the author and audience are the same. In each pair, one post did **better than a statistical model expected** and one did **worse than expected**. The model already knows about the properties listed under "Already measured", so the difference between the two posts is something those properties do not capture.

Your job: propose new, testable **hypotheses** about post content that could explain the gap. Each hypothesis is a yes/no property of a single post. Every one will be tested on thousands of other authors' posts, so a wrong guess is cheap. A property that sounds plausible but is not visible in these pairs is not useful.

## Rules for every property

1. **Yes/no question about one post**, answerable by a careful reader from the post text alone, with no outside knowledge.
2. **Descriptive, not evaluative.** Never use words like good, strong, engaging, compelling, authentic, clear, valuable, viral, or their synonyms. Describe what is on the page.
3. **Atomic**: one characteristic per question; do not join two ideas with "and" or "or".
4. **Varies**: true of roughly 3% to 60% of LinkedIn posts.
5. **Actionable for a writer**: something a person could choose to do or not do in their next post (a kind of content, a framing, a structure, a stance), not who the author is. No follower counts, names, companies or timing.
6. **New**: not a restatement of anything under "Already measured". A narrower or sharper version of a measured property is fine if it separates cases the measured one lumps together (for example: whose achievement is celebrated, the author's or someone else's).
7. Give a one-sentence definition of "yes", plus a short invented yes example and no example (20 words or fewer each, never copied from the posts).

## Output

Return a JSON object:

```json
{
  "properties": [
    {
      "name": "short_snake_case_name",
      "question": "Yes/no question about a single post?",
      "definition": "One sentence defining when the answer is yes.",
      "yes_example": "Short invented snippet where the answer is yes.",
      "no_example": "Short invented snippet where the answer is no.",
      "hypothesis": "One sentence: which way you expect it to go and why, citing pair numbers."
    }
  ]
}
```

Return exactly $n properties. Return only the JSON object.
