You are helping build a bank of candidate features for a study of LinkedIn posts.

You will see several pairs of LinkedIn posts. Both posts in a pair were written by the same person, so the author and audience are the same within a pair. Your job is to propose **properties**: concrete, observable characteristics of a post that are true of some posts and false of others.

## You are not judging anything

You are not told how any post performed, and you must not guess. Do not decide which properties are good or bad, and do not hint at it. A statistical model trained later will measure whether each property matters and in which direction. A property that might help a post, one that might hurt it, and one that might make no difference are all equally useful to propose. Describe what is on the page, nothing more.

## Rules for every property

1. **Yes/no question about one post.** A careful reader must be able to answer it from the post's text and media type alone, with no outside knowledge about the author, company, or events.
2. **Descriptive, not evaluative.** Never use judging words such as good, bad, strong, weak, compelling, engaging, effective, clear, interesting, high-quality, professional, authentic, valuable, polished, or viral, and avoid their synonyms. Ask about what is on the page.
   - Bad: "Does the post have a strong hook?"
   - Good: "Does the first sentence end with a question mark?"
   - Bad: "Is the story relatable?"
   - Good: "Does the post describe a specific moment from the author's own working life?"
3. **Atomic.** One characteristic per question. Do not join two separate ideas with "and" or "or".
4. **Varies across posts.** Skip properties that nearly every LinkedIn post has or that almost none would have. Aim for properties that would be true of roughly 5% to 70% of posts.
5. **About content and form, not identity.** Do not ask about the author's name, title, follower count, or a particular company, product, or person. Generalize instead, e.g. "Does the post promote a product or service the author sells?"
6. **Not about timing.** Do not ask about posting time, date, or engagement numbers. The attached media type is fine to use.
7. **Precise boundaries.** Give a one-sentence definition of what counts as "yes", plus a short yes example and a short no example. Write the examples yourself (20 words or fewer each). Do not copy text from the posts.

## Where to look

Within each pair, compare the two posts closely. Because the author is the same, what differs between them is about the content. Also propose properties that vary across pairs.

Report the differences you actually see, wherever they are: in what the post says, how it is built, how it sounds, or what it asks of the reader. There is no checklist of categories to fill. If several of the most visible differences are of the same kind, propose several properties of that kind. Prefer properties a reader would notice over generic ones that could be written without reading these posts.

## Output

Return a JSON object with exactly $n_properties properties, each different from the others:

```json
{
  "properties": [
    {
      "name": "short_snake_case_name",
      "question": "Yes/no question about a single post?",
      "definition": "One sentence defining when the answer is yes.",
      "yes_example": "Short invented snippet where the answer is yes.",
      "no_example": "Short invented snippet where the answer is no."
    }
  ]
}
```

Return only the JSON object.
