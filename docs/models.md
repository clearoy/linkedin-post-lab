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

## What the sparse linear model keeps

CV tried caps of 1–10 features and a cap of 3 scored best. The lasso path jumps from 2 active
features to 4, so the model has 2:

| Feature | Weight | Multiplier | Same sign in CV folds |
|---|---|---|---|
| Does the post explicitly mention an emotion felt by the author or their team? | +0.250 | 1.28× | 5/5 |
| Does the post explicitly thank or express appreciation to a person, group, or organization? | +0.233 | 1.26× | 3/5 |

A weight is the change in lift when P(yes) goes from 0 to 1; the multiplier is e^weight.
These are associations within authors, not proven causes.

## Note on media in Jev's input

Jev was given each post as `{media_attached, post_text}`, and DeepSeek saw "Media attached: …"
during discovery. The models no longer use media in any form. Jev's answers to content questions
could still be slightly influenced by that line. Removing it fully means re-scoring the posts
without it (about $4.32 for the same subset).
