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

## Project layout

```
linkedin-analysis/
├── data/
│   ├── raw/                 # posts_<band>.parquet from GCS (gitignored)
│   └── processed/           # generated outputs (gitignored)
├── src/linkedin_analysis/
│   ├── paths.py             # folders, bands, GCS location
│   ├── data.py              # load_posts / load_all
│   ├── labels.py            # author-relative labels
│   ├── llm.py               # DeepSeek / OpenRouter client, cache
│   ├── discovery.py         # propose + merge properties (DeepSeek)
│   ├── judge.py             # P(yes) per post x feature (Jev decisions API)
│   ├── splits.py            # 80/20 author split + CV folds
│   ├── model.py             # controls baseline, lasso, RuleFit, evaluation
│   ├── predict.py           # score a new post from the saved model bundle
│   └── prompts/             # all LLM prompts, as editable .md files
├── scripts/
│   ├── build_labels.py      # writes data/processed/labels.parquet
│   ├── build_splits.py      # writes data/processed/splits.parquet
│   ├── discover_features.py # writes data/processed/features/feature_bank.json
│   ├── judge_features.py    # writes data/processed/judgments*/features.parquet
│   └── train_models.py      # writes data/processed/models/
├── app/dashboard.py         # Streamlit: evaluate a post + pipeline overview
├── tests/                   # pytest tests (uv run pytest)
└── docs/
    ├── labels.md            # label definitions, choices, checks
    ├── features.md          # feature discovery and judging method
    └── models.md            # models, first results
```

## Pipeline

| Step | Status | Command | Output |
|---|---|---|---|
| 1. Labels | done | `uv run python scripts/build_labels.py` | `data/processed/labels.parquet` |
| 2. Author split (80/20, 5 folds) | done | `uv run python scripts/build_splits.py` | `data/processed/splits.parquet` |
| 3. LLM feature discovery (DeepSeek) | done: 205 features | `uv run python scripts/discover_features.py` | `data/processed/features/feature_bank.json` |
| 4. Feature judging (Jev) | done: every labeled post, 147 general features ($11.24) | `uv run python scripts/judge_features.py --bank data/processed/features/feature_bank_general.json --split train --max-cost 13 --out data/processed/judgments_full_train` (and `--split test --max-cost 3.5 --out data/processed/judgments_full_test`) | `data/processed/judgments_full_*/features.parquet` |
| 5. Theme model / RuleFit | done | `uv run python scripts/train_models.py --judgments data/processed/judgments_full_train data/processed/judgments_full_test --bank data/processed/features/feature_bank_general.json --themes 15`, then `scripts/name_themes.py` | `data/processed/models/` |
| 5b. Baselines and learning curve | done | `scripts/compare_baselines.py --train data/processed/judgments_full_train --test data/processed/judgments_full_test`; `scripts/learning_curve.py --judgments` (same two folders) | `models/baselines.json`, `models/learning_curve.json` |
| 6. Dashboard | done | `uv run streamlit run app/dashboard.py` | http://localhost:8501 |

Results are in [docs/models.md](docs/models.md). The models use content
features only. On all 165 held-out authors the theme model ranks same-author
posts correctly 61.3% of the time [59.3, 63.4] (chance is 50%), vs 57.9% for
media, length and time alone.

**Labels:** each post is compared only with the same author's other posts, to
remove the author's influence. The main target is `lift` = log of reactions
relative to the author's usual. The labels also include `win` (≥2× usual) and
`tier` (top/middle/bottom third, for contrast pairs). Definitions and
checks are in [docs/labels.md](docs/labels.md).

```python
import pandas as pd
from linkedin_analysis.data import load_all

labels = pd.read_parquet("data/processed/labels.parquet")
posts = load_all().merge(labels[["urn", "lift", "lift_pct", "win", "tier"]], on="urn")
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Put your API keys in `.env` (gitignored; template in `.env.example`):
`DEEPSEEK_API_KEY` for discovery and `OPENROUTER_API_KEY` for the Jev judge.
All LLM responses are cached in `data/processed/llm_cache/`, so re-runs are free.

```bash
gcloud storage cp "gs://prototypes-498523-linkedin-post-model/2026-10-19/posts/*.parquet" data/raw/
```

Source: `gs://prototypes-498523-linkedin-post-model/2026-10-19/` (GCP project `prototypes-498523`). The same bucket holds the model outputs, results, baselines and raw Apify activity. Its own `README.md` describes them.

## Quick start

```python
from linkedin_analysis.data import load_all, load_posts

posts = load_posts("1k-10k")   # one band
everything = load_all()        # both bands, with a `band` column

posts.groupby("media")["hit"].mean()              # hit rate by media type
posts.loc[posts.viral == 1, ["name", "reactions", "body"]]
```

**Imports on this Mac:** Python 3.13+ skips `.pth` files flagged as hidden, and
macOS keeps re-flagging the ones uv creates for the editable install. So don't
rely on that install. Scripts in `scripts/` add `src/` to `sys.path` themselves,
and pytest is configured with `pythonpath = ["src"]`. For a REPL or notebook:

```bash
PYTHONPATH=src uv run python
```

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
