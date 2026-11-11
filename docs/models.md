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
