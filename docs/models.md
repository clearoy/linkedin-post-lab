# Models and first results

Code: [`model.py`](../src/linkedin_analysis/model.py),
[`train_models.py`](../scripts/train_models.py), dashboard scoring in
[`predict.py`](../src/linkedin_analysis/predict.py).

## Setup

- **Target:** `lift` (see [labels.md](labels.md)): log reactions relative to the author's usual.
- **Inputs:** Jev P(yes) for 202 features (3 of 205 dropped as almost always yes or no), plus
  controls: media type, log word count, weekend, and hour bucket (PT).
- **Data (first run, Jev under $5):** 115 train authors (2,148 posts) and 65 test authors
  (1,315 posts), sampled from the 80/20 author split.
- **Tuning:** 5 author-grouped folds inside train. The test set was scored once with the
  chosen settings.

| Model | What it is |
|---|---|
| Controls | Ridge on controls only: the baseline content must beat |
| Linear | Lasso on standardized inputs, penalty with lowest CV error; coefficients reported in raw units |
| RuleFit | Rules from 100 depth-3 boosted trees plus scaled linear terms, then lasso (one-standard-error penalty) |

**Penalty choice, made on train CV only.** For the linear model, the
one-standard-error rule removed every term (CV 0.519). The minimum-error
penalty kept 29 terms (CV 0.589). For RuleFit, the one-standard-error rule
did better (CV 0.580 vs 0.573 with 350 terms, which overfit).

## Results on held-out authors

Pairwise accuracy: for two posts by the same author whose lift differs by at
least 0.1, the share the model ranks correctly (0.5 = chance). 95% CIs come from
resampling the 65 test authors.

| Model | CV | Test | Gain over controls |
|---|---|---|---|
| Controls | 0.579 | 0.581 [0.547, 0.614] | — |
| Linear | 0.589 | **0.607** [0.566, 0.645] | +0.026 [−0.009, +0.059] |
| RuleFit | 0.580 | 0.595 [0.559, 0.628] | +0.014 [−0.016, +0.042] |

**Reading it:**
- Every model ranks posts better than chance.
- The content models do better than controls alone, but with 65 test authors the gain's
  interval still includes zero.
- Scoring more authors with Jev (about $0.024 per author) is the cheapest way to confirm it.

## What the linear model keeps (strongest first)

Most of these keep the same sign in 4–5 of the 5 CV folds:
- **Up:** announcing a new job or role (1.41×), a business-deal story (1.18×), lists of
  external opportunities (1.11×), naming an emotion (1.10×), thanking someone (1.09×), a
  specific person's journey (1.08×), family mentions (1.07×), an image attached (1.07×).
- **Down:** templates or exercises for the reader (0.89×), describing a problem that affects
  a group of people (0.97×).

These are associations within authors, not proven causes.
