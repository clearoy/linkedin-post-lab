# Labels: how well a post did *for its author*

The goal is to learn which **content** works. Raw reactions mostly measure the
author: about two thirds of the variance in log reactions lies between authors
(68% in 1k–10k, 65% in 10k–20k), and the dataset's own band percentile `pct`
has a rank correlation of 0.79 with the author's median reactions. So every label
here compares a post only with the **same author's other posts**.

Code: [`src/linkedin_analysis/labels.py`](../src/linkedin_analysis/labels.py).
Build: `uv run python scripts/build_labels.py`.

## Definitions

For each author with at least `min_posts` posts, with posts in time order:

| Column | Definition | Use it for |
|---|---|---|
| `baseline` | Median reactions of the author's `window` posts nearest in time, **excluding this post** | Reference point |
| `lift` | `log((reactions + s) / (baseline + s))`, with `s = smoothing`. 0 = typical for the author, +0.69 = 2×, −0.69 = ½× | **Main target** for regression or RuleFit |
| `lift_pct` | Within-author percentile of `lift`, `(rank − 0.5) / n`, between 0 and 1 | Robust target, plots |
| `win` | `lift ≥ log(win_ratio)`: the post got at least 2× the author's usual | Binary classification, readable rules |
| `tier` | `top` / `middle` / `bottom` third of the author's posts by `lift_pct` | Same-author contrast pairs (top vs bottom) for LLM feature discovery |
| `n_posts` | Author's post count | Weighting, filtering |

Defaults (`LabelConfig`): `min_posts=5`, `window=12`, `smoothing=2`,
`win_ratio=2`, `tier_share=1/3`. Each one can be overridden on the command
line, for example `--window 8`.

## Why these choices

| Choice | Reason | Alternatives tested |
|---|---|---|
| Compare within each author instead of top x% of the band | Removes audience size | — |
| Continuous `lift` instead of top x% per author | A top-20% cutoff lands on tied reaction counts for 34% of 1k–10k authors, and top x% throws away how big the win was | Within-author top x% |
| Leave the post out of its own baseline | Otherwise a big hit raises its own baseline and shrinks its lift | — |
| **Local** baseline (12 nearest posts in time) | Authors' reach drifts. Correlation of lift with timeline position: −0.07 with the all-posts median, −0.045 with 12 neighbors, −0.03 with 8 (but 8 makes the baseline noisier) | All-posts median; 8 neighbors |
| `smoothing = 2` | With no smoothing, going from 1 to 3 reactions counts as a 2× win. Win rate for the smallest authors: 0.23 with `s=1`, 0.14 with `s=2`, 0.10 with `s=3`, against about 0.19 for larger authors. `s=2` sits in the middle | 1, 3, 5 |
| `min_posts = 5` | Keeps 832 of 1,047 authors and 97% of posts. Below 5 posts the baseline is too noisy | — |

## Sanity checks (written to `labels_summary.json`)

Full run with the default settings: 15,331 posts and 832 authors labeled.

| Check | Value | Want |
|---|---|---|
| Rank corr. of raw band `pct` with author median | 0.79 | (the problem being fixed) |
| Rank corr. of `lift` with author median | −0.06 | about 0 |
| Rank corr. of `lift_pct` with author median | 0.00 | 0 |
| Rank corr. of `lift` with position in the author's timeline | −0.045 | about 0 |
| `win` rate by author median (≤3 / 4–10 / 11–30 / >30) | 0.14 / 0.18 / 0.19 / 0.19 | flat |
| Overall `win` rate | 17% (1k–10k 16%, 10k–20k 19%) | — |

## Caveats for modeling

- **Authors still differ in how many posts they contribute** (5–30). Weight by
  `1 / n_posts` if each author should count equally.
- **Split train and test sets by `author`.** Labels are relative to the author's
  other posts, so splitting at random leaks information between the sets.
- **Small leave-one-out effect.** One post's reactions are part of its
  neighbors' baselines, so an author's lifts are slightly negatively
  correlated with each other. This is harmless for feature analysis.
- **`lift` removes the author's general level, not author-by-content
  interactions** (a topic that works only for founders, say). Interactions stay
  in the signal, which is intended.
- **Post age and timing are not removed.** Within an author they matter little
  (rank correlations ≤ 0.05), but keep `age`, `hour_pt`, `weekday` as control
  features in the model.
