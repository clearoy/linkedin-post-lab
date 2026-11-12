You are curating hypotheses for a study of what makes LinkedIn posts get more reactions.

You will get a list of proposed yes/no post properties. Many overlap. Produce a final list of at most $n distinct properties:

- Merge near-duplicates into one property, keeping the clearest wording.
- Drop any property that restates one under "Already measured", is evaluative (good, strong, engaging, authentic, ...), joins two ideas with "and"/"or", is about the author's identity rather than the post's content, or would be true of almost every or almost no post.
- Prefer properties a writer could act on, and keep a spread of kinds (what the post is about, how it opens, its stance, who it centers, what it asks of the reader). Do not keep more than three properties that are near-variants of each other.
- Keep the field format exactly as given (name, question, definition, yes_example, no_example, hypothesis). Questions end with "?".

Return a JSON object `{"properties": [...]}` and nothing else.
