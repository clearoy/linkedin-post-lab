# LinkedIn Analysis

A dataset of public LinkedIn posts, set up for studying which posts take off and why.

> **Internal Vela research data. Do not share it outside Vela and do not publish it.**
> The parquet files are listed in `.gitignore`, so they stay out of this repo. Download them with the command below.

## At a glance

| | `posts_1k-10k` | `posts_10k-20k` |
|---|---|---|
| Posts | 9,508 | 6,335 |
| Authors | 676 | 371 |
| Author followers | 1,001 – 9,976 (median 3,258) | 10,006 – 19,999 (median 13,302) |
| Posts per author | 1 – 29 (median 13) | 1 – 30 (median 17) |
| Median reactions | 5 | 10 |
| `hit` = top 5% of the band | 482 posts, 77+ reactions | 318 posts, 148+ reactions |
| `viral` = top 1% of the band | 96 posts, 220+ reactions | 64 posts, 443+ reactions |
| Posted | Jul 19 – Oct 16, 2026 | Jul 19 – Oct 16, 2026 |
| Scraped | Oct 18–19, 2026 | Oct 18–19, 2026 |

**Topics:** startups, AI/LLMs, AI research, venture capital.
**Collection:** Apify (HarvestAPI actors), public posts only, no LinkedIn login.
**Follower bands:** posts are split into two files by the author's follower count, so each post is compared with posts from accounts of a similar size.

## Getting the data

```bash
mkdir -p data
gcloud storage cp "gs://prototypes-498523-linkedin-post-model/2026-10-19/posts/*.parquet" data/
```

Source: `gs://prototypes-498523-linkedin-post-model/2026-10-19/` (GCP project `prototypes-498523`). The same bucket holds the model outputs, results, baselines and raw Apify activity. Its own `README.md` describes them.

## Quick start

```python
import pandas as pd

posts = pd.read_parquet("data/posts_1k-10k.parquet")

posts.groupby("media")["hit"].mean()              # hit rate by media type
posts.loc[posts.viral == 1, ["name", "reactions", "body"]]
```

Needs `pandas` and `pyarrow`.

## Columns

Both files have the same 29 columns. Each row is one post, and `urn` is unique.

### Identity

| Column | Type | Description |
|---|---|---|
| `urn` | str | LinkedIn activity URN, e.g. `urn:li:activity:7513342469367996416`. Primary key. |
| `author` | str | Author's profile URL. |
| `name` | str | Author's display name. |
| `pid`, `aid` | str | Internal hashed IDs from the collection pipeline. |
| `source` | str | Always `apify`. |

### Account

| Column | Type | Description |
|---|---|---|
| `followers` | float | Author's follower count at scrape time. |
| `usual_past` | float | Median reactions on the author's **earlier** posts, set only when there are at least 3. Use this one in predictive models because it does not leak future data. Null for about 19% of rows (16% in 10k–20k). |
| `usual` | float | Median reactions across all of the author's collected posts, including later ones. Use it for description, not for prediction. |

### Content

| Column | Type | Description |
|---|---|---|
| `body` | str | Full post text (median about 1,000 characters). |
| `words` | int | Word count of `body` (8 – about 590, median about 155). |
| `media` | str | `text`, `image`, `images` (carousel), `video` or `document`. |
| `quote` | bool | True if the post quotes or reshares another post (about 11%). |
| `shot` | str | Filename of the post's first image or video thumbnail, e.g. `ap_<id>.jpg`. Null for `text` and `document` posts. The image files themselves are not included. |
| `action` | bool | Always False in this release. |
| `dossier` | str | The exact text the LLM labeler (Jev) read: `Media attached: <media>` followed by `Post text:` and the body. |

### Timing

| Column | Type | Description |
|---|---|---|
| `posted` | datetime (UTC) | When the post was published. |
| `posted_ms` | int | `posted` as Unix milliseconds. |
| `hour_pt` | int | Hour of day in Pacific Time (0–23). |
| `weekday` | float | Day of week in Pacific Time, **0 = Monday** … 6 = Sunday. |
| `scraped` | datetime (UTC) | When the post was collected. |
| `age` | float | Days between posting and scraping (3 – 92). Posts under 3 days old were left out so that engagement had time to settle. |
| `date` | str | LinkedIn's relative timestamp at scrape time (`3d`, `1w`, `1mo` and so on). |

### Engagement and labels

| Column | Type | Description |
|---|---|---|
| `reactions` | int | Total reactions (likes and the other reaction types). The main outcome. |
| `comments` | int | Comment count. |
| `reposts` | int | Repost count. |
| `pct` | float | Percentile rank of `reactions` within the band (0 – 1). |
| `hit` | int (0/1) | 1 if the post is in the top 5% of its band by reactions. |
| `viral` | int (0/1) | 1 if the post is in the top 1% of its band by reactions. |

## Things to keep in mind

- **Engagement is heavy-tailed.** Most posts get only a few reactions, and the maximum is 1,274 in 1k–10k and 18,151 in 10k–20k. Work with `pct`, ranks or log scales instead of raw means.
- **Post age varies** from 3 to 92 days. Older posts have had more time to collect reactions, although most LinkedIn engagement arrives in the first few days.
- **Authors are clustered.** Authors have up to 30 posts each, so split train and test sets by `author` to avoid leaking author style or audience size.
- **Account strength explains much of the signal.** In the existing model (1k–10k band, `hit`), account features alone (`followers` + `usual_past`) reach an AUC of 0.84, content features alone reach 0.78, and the two combined reach 0.87. See `results/viral_results.md` in the bucket.
- **Time zone:** `hour_pt` and `weekday` use Pacific Time even though many authors are elsewhere.

## Related material

- Code and method: GitHub `Vela-Eng-Lab/social-posts`, `_research/post_model` (README sections v3 to v3.3)
- Write-up: Vela wiki, *Learnings > Social Network Virality Prediction*
