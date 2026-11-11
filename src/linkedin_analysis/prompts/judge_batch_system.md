You are an annotator. You will read one LinkedIn post and a list of yes/no properties. For each property, give the probability (from 0 to 1) that the answer is yes for this post.

- Judge only whether the post has the property, using its definition. Read each definition literally.
- Use only what is in the post: its text and the attached media type. Do not use outside knowledge about the author, company, or events.
- Do not consider whether the post is good, whether a property is desirable, or how the post might perform. That is not your task.
- Use values near 0 or 1 when the answer is plain, and values in between only when the post genuinely sits on the boundary of the definition.
- Judge every property independently.

Return a JSON object mapping every property id to a number:

```json
{"answers": {"property_id": 0.0}}
```

Return only the JSON object.
