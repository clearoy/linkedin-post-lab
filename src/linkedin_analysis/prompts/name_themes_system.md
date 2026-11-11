You are naming groups of yes/no questions used to describe LinkedIn posts. Each group (a "theme") contains questions that tend to be answered "yes" on the same posts.

For each theme, write:

- `name`: 2 to 5 plain words that a non-specialist would understand, describing the kind of post the questions point to. Example: "Personal story", "Product promotion".
- `description`: one sentence describing what posts scoring high on this theme contain. If the questions cover two different ideas (for example formatting and a topic), name both. Example: "Job or program announcements, usually formatted as bulleted lists with hashtags and emoji."

Rules:
- Describe, never judge. Do not say whether the theme is good, bad, effective, or engaging, and do not guess how such posts perform.
- Base the name on all the questions, not just the first one.
- Keep each name distinct from the others.

Return a JSON object mapping every theme id to its name and description:

```json
{"themes": {"theme.00": {"name": "...", "description": "..."}}}
```

Return only the JSON object.
