# Models and first results

Code: [`model.py`](../src/linkedin_analysis/model.py),
[`train_models.py`](../scripts/train_models.py), dashboard scoring in
[`predict.py`](../src/linkedin_analysis/predict.py).

## Setup

- **Target:** `lift` (see [labels.md](labels.md)): log reactions relative to the author's usual.
- **Inputs: content only.** Jev P(yes) for 200 features:
  - 205 discovered;
  - 3 dropped for being almost always yes or no;
  - 2 dropped for describing the attachment rather than the content ("has at least one image
    attached", "attached media is a video").

  **No media type, length, or posting-time inputs.** Pass `--controls` to add them back.
- **Data (first run, Jev under $5):** 115 train authors (2,148 posts) and 65 test authors
  (1,315 posts), sampled from the 80/20 author split.
- **Tuning:** 5 author-grouped folds inside train. The test set is scored once with the chosen
  settings.

| Model | What it is |
|---|---|
| Sparse linear | **Relaxed lasso**: the lasso picks at most 10 content features (CV picks how many), then an unpenalized refit on those |
| RuleFit | Rules from 100 depth-3 boosted trees on content features, then lasso (one-standard-error penalty) |
| *Format reference* | *Ridge on media type, length, and posting time. Used only as a comparison row, never as a model input* |

**Why a relaxed lasso.** The plain lasso kept 26 features. Penalizing it harder until 10 were
left also shrank the survivors toward zero. The relaxed lasso separates choosing features from
estimating their effects, so the kept effects keep their full size.

## Results on held-out authors (content only)

Pairwise accuracy: for two posts by the same author whose lift differs by at least 0.1, the
share the model ranks correctly (0.5 = chance). 95% CIs come from resampling the 65 test authors.

| Model | Terms | CV | Test |
|---|---|---|---|
| Sparse linear | 2 features | 0.584 | 0.574 [0.540, 0.607] |
| RuleFit | 41 rules | 0.587 | 0.594 [0.558, 0.630] |
| *Format reference (not a model input)* | — | 0.579 | 0.581 [0.547, 0.614] |

**Reading it:**
- Both content-only models rank same-author posts better than chance.
- Content alone does about as well as format alone. The differences are within noise with 65
  test authors.
- Scoring more authors with Jev (about $0.024 per author) is the cheapest way to sharpen this.

## Current model: content themes (text-only Jev)

What changed from the first run:
- **Text only:** Jev reads only the post text; no attached-media field.
- **Broadly applicable features:** 147 of 205 features, each true in 5–95% of posts and for at least
  20% of authors.
- **New, smaller Jev subset:** 80 train and 45 test authors ($2.09).
- **Themes:** features were clustered on train posts into 15 themes by how often they are true
  together. Themes with fewer than 3 questions are too niche and are not modeled.
- **Theme names:** written by DeepSeek from each theme's full question list (`scripts/name_themes.py`).

The sparse linear model on themes keeps 5: CV 0.607, test **0.623** [0.586, 0.660] on 45 held-out
authors.

| Theme | Questions | Typical effect* | Same sign in CV folds |
|---|---|---|---|
| Personal experience story | 13 | 1.21× | 5/5 |
| Job or opportunity listing | 24 | 1.19× | 5/5 |
| Recognition and milestones | 12 | 1.11× | 5/5 |
| Funding and news roundup | 4 | 1.03× | 3/5 |
| Product or event promotion | 15 | 0.87× | 5/5 |

\* Typical effect: moving a post's theme score from the 25th to the 75th percentile of training posts.

**How to read a theme.**
- A theme score is the share of its checklist a post ticks (the mean P(yes) of its questions).
- Every post is scored on every theme. A post that doesn't touch a theme scores near 0 there, so that
  theme barely changes its prediction.
- Associations within authors, not proven causes.

## Full-data run: every author scored (2026-11-11)

Jev scored all 147 general features on **every labeled post**: 667 train authors (12,299 posts)
and 165 test authors (3,032 posts). $11.24, about 5.5 minutes, no failed answers. Same pipeline as above:
15 themes clustered on train posts, themes with fewer than 3 questions set aside, relaxed lasso, ≤ 10 themes.

### Results on all 165 test authors (3,032 posts)

| Model | CV | Test | Test 95% CI |
|---|---|---|---|
| Ridge on all 147 Jev questions | 0.593 | 0.615 | [0.593, 0.636] |
| **Themes, sparse linear (7 themes)** | 0.588 | **0.613** | [0.593, 0.634] |
| Gradient boosting on Jev questions | 0.594 | 0.599 | [0.578, 0.619] |
| TF-IDF + ridge | 0.587 | 0.596 | [0.577, 0.616] |
| RuleFit on themes | 0.579 | 0.593 | [0.573, 0.612] |
| *Format reference (not a model input)* | 0.585 | 0.579 | [0.557, 0.600] |

- **The confidence range is about half as wide** as with 45 test authors (±0.02 instead of ±0.04).
- **The earlier 0.623 was partly test-set luck.** Trained on 80 authors and tested on all 165, the theme model
  averages 0.596 (learning curve below), not 0.623.
- **7 themes do almost as well as all 147 questions** (0.613 vs 0.615), with 7 readable weights instead of 147.

### Learning curve (same 165 test authors at every size)

Random samples of whole train authors; every model refit from scratch (themes re-clustered, settings re-tuned).
Mean test pairwise accuracy over 5 samples (1 run at 667).

| Train authors (posts) | Themes linear | Jev ridge | Jev GBM | TF-IDF ridge |
|---|---|---|---|---|
| 40 (≈700) | 0.590 | 0.586 | 0.557 | 0.551 |
| 80 (≈1,500) | 0.596 | 0.598 | 0.571 | 0.563 |
| 160 (≈3,000) | 0.600 | 0.607 | 0.593 | 0.581 |
| 320 (≈5,900) | 0.598 | 0.612 | 0.592 | 0.587 |
| 667 (12,299) | 0.613 | 0.616 | 0.601 | 0.596 |

**Reading it:**
- More data helps every model, by about +0.005 to +0.01 per doubling of authors.
- Jev features beat raw words at every size. They matter most with little data (0.59 vs 0.55 at 40 authors).
- The theme model is already near its level with few authors. Its jump at 667 is one run, so part of it may be noise.
- No model is close to flattening out on more data, but the gains are small. Doubling the data again would
  likely add about 0.01, so better properties (new discovery) are the bigger lever now.

### What the model keeps (trained on all 667 authors)

| Theme (DeepSeek name) | Questions | Weight | Typical effect* | Same sign in CV folds |
|---|---|---|---|---|
| Personal story | 13 | +0.463 | 1.16× | 5/5 |
| Milestone and gratitude | 18 | +0.451 | 1.10× | 5/5 |
| Job opportunity listing | 22 | +0.367 | 1.10× | 5/5 |
| Funding and numbers | 4 | +0.106 | 1.03× | 5/5 |
| Product promotion and news | 12 | −0.093 | 0.97× | 4/5 |
| Cited sources | 8 | −0.133 | 0.98× | 5/5 |
| How-to guidance | 9 | −0.188 | 0.94× | 5/5 |

\* 25th → 75th percentile of training posts.

The story is the same as the small run: personal stories, milestones/gratitude and job listings go with
above-usual reactions. Advice-style posts, cited sources and promotion go with slightly below-usual reactions.
New with more data: how-to guidance and cited sources now show up as small negatives, and promotion's
negative weight is smaller. The dashboard asks Jev 86 questions per post (the members of these 7 themes).
The small-run model is kept in `data/processed/models_small_80authors/` for comparison.

## How many questions are needed? Compact model

Code: [`compact_model.py`](../scripts/compact_model.py). All points are trained on all 667 train authors and
tested on the same 165 test authors. No extra Jev calls: every question was already answered on every post.

| Setup | Questions | CV | Test | Test 95% CI |
|---|---|---|---|---|
| Ridge on all questions | 147 | 0.593 | 0.615 | [0.593, 0.636] |
| 7 themes, all members (full model) | 86 | 0.588 | 0.613 | [0.593, 0.634] |
| 7 themes × top 5 | 34 | 0.581 | 0.607 | [0.586, 0.628] |
| **7 themes × top 2 (compact model)** | **14** | 0.583 | **0.602** | [0.580, 0.625] |
| Best 10 single questions (forward selection) | 10 | 0.589 | 0.598 | [0.578, 0.618] |
| Best 5 single questions | 5 | 0.587 | 0.597 | [0.575, 0.618] |
| 7 themes × top 1 | 7 | 0.577 | 0.592 | [0.570, 0.614] |
| Best single question | 1 | 0.557 | 0.580 | [0.559, 0.600] |

- **10–15 questions keep about 85–90% of the gain over chance.** The loss (about 0.01) shows in CV too, so it is real.
- **Averaging still helps:** 7 themes × 2 questions beat 14 individually chosen questions.
- **Single-question picks are not stable.** In each CV fold the relaxed lasso with 10 questions always picks
  the same 5; the rest swap among 17 overlapping questions.
- **Compact themes are renamed.** A theme's top 2 questions are the ones most correlated with its full score on
  train posts, and they can be narrower than the full theme. For example, "Job opportunity listing" is measured by
  two list-format questions, so its compact name is "List post". DeepSeek names each compact theme from its own
  2 questions.

The dashboard's Evaluate tab offers both: **Full (86 questions, test 0.613)** and **Compact (14 questions,
test 0.602)**. The full model's answers also cover the compact one.
