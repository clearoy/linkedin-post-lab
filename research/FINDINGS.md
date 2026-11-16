# What content gets more reactions on LinkedIn: findings

> First-pass findings. The full report, with comments/reposts, kinds of post, combinations and the scorecard behind the dashboard, is [REPORT.md](REPORT.md).

832 authors, 15,331 posts. Content only: the post text, never the attached media. Method and scripts: [README.md](README.md).
Every table row: [results/tables.md](results/tables.md). Jev spend: **$4.19 of the $5 budget** (ledger below).

## How to read the numbers

* **"1.30×"** = posts with the property got about 30% more reactions **than the same author's usual post**.
  Every comparison is within one author, so a popular author who often writes a certain kind of post cannot create an effect.
* **Confirmed** = significant on the 667 train authors after correcting for testing many properties (Benjamini-Hochberg q < 0.05),
  **and** pointing the same way on 165 held-out authors who were never used to choose or design anything (BH-adjusted one-sided p < 0.05).
* **Win rate** = for a random pair of posts by the same author, one with the property and one without, how often the one with it got more reactions. 0.50 is a coin flip.
* **Unique** = the effect left once the other properties are held fixed (one joint model), because many properties appear together.
* This is observational. "Confirmed" means a reliable association for the same author, not proof that adding the property to any post will cause more reactions.

## The short version

**196 properties were tested; 78 were confirmed** (60 go with more reactions, 18 with fewer). They tell one consistent story:

> On LinkedIn, reactions follow **people and real moments**: your news, the people around you, and what actually happened.
> Generic advice, frameworks, links and engagement tricks do not help, and several hurt.

A 7–9-item checklist built from these results (chosen on train authors only) sorts **held-out authors'** posts cleanly:

| Checklist items in the post | Share of posts | Reactions vs the author's usual (95% CI) |
|---|---|---|
| 0 | 14% | **0.83×** [0.77, 0.89] |
| 1 | 25% | 0.88× [0.83, 0.92] |
| 2 | 27% | 0.96× [0.91, 1.01] |
| 3 | 20% | 1.08× [1.01, 1.16] |
| 4 or more | 14% | **1.48×** [1.34, 1.66] |

Checklist items (from the train-only joint model): own career milestone · credits others for your success · names 3+ people ·
mentions a place · story about a specific person · personal anecdote · opens with a named person, company or product · mentions a money amount · 3+ hashtags.
The last two were picked on train but did **not** replicate on their own (see "not replicated" below), so the guidance below leaves them out.

## Guidance for writers, with the evidence

### 1. Put real people in the post (strongest, most consistent effects)

| Do this | Reactions | Held-out authors | Win rate | Unique |
|---|---|---|---|---|
| Credit specific people for helping you reach something | **1.87×** [1.70, 2.05] | 1.85× | 0.71 | 1.21× |
| Thank a specific person or group | **1.65×** [1.54, 1.77] | 1.56× | 0.68 | (grouped with the line above) |
| Name three or more specific people | **1.59×** [1.47, 1.72] | 1.40× | 0.65 | 1.22× |
| Tell a story about one individual | 1.50× [1.43, 1.58] | 1.66× | 0.65 | 1.13× |
| Publicly acknowledge a person or team | 1.22× [1.17, 1.28] | 1.15× | 0.58 | |
| Mention a collaboration | 1.37× [1.29, 1.45] | 1.30× | 0.62 | |
| Describe something you learned from a specific person | 1.14× [1.06, 1.22] | 1.21× | 0.56 | |

These hold within every kind of post tested: advice, product promotion, events, industry analysis
(for example, thanking specific people inside an *advice* post: 1.60× [1.42, 1.80]).
**Practical:** instead of "Thanks to the team", write who, and what they did.

### 2. Share your own news and real moments

| Do this | Reactions | Held-out | Win rate |
|---|---|---|---|
| Announce a milestone in your own career (new job, promotion, anniversary, leaving) | **2.00×** [1.75, 2.29] | 2.34× | 0.71 |
| Describe a turning point in your career or life | 1.52× [1.33, 1.73] | 1.60× | 0.65 |
| Say you're proud of something | 1.53× [1.42, 1.65] | 1.57× | 0.66 |
| Describe an event you attended | 1.49× [1.40, 1.58] | 1.53× | 0.65 |
| Announce that you're taking part in something (speaking, judging, attending) | 1.45× [1.36, 1.55] | 1.45× | 0.64 |
| Say how long you've been at it (years in the field, months on a project) | 1.40× [1.30, 1.50] | 1.58× | 0.61 |
| Write about something happening today | 1.30× [1.22, 1.39] | 1.42× | 0.59 |
| Say what you're excited about next | 1.28× [1.20, 1.37] | 1.41× | 0.60 |
| Report a number you or your team achieved | 1.24× [1.16, 1.32] | 1.17× | 0.59 |

The career milestone is the largest single effect, and it has a confirmed unique effect (1.42× [1.25, 1.61]) even after everything else is held fixed.
Most people cannot make milestones happen on demand. The usable lesson: **when you have news, post it, and frame other posts around what actually happened to you.**

### 3. Write as a person, not a brand

| Do this | Reactions | Held-out | Win rate |
|---|---|---|---|
| Include a personal anecdote | 1.42× [1.36, 1.49] | 1.51× | 0.63 |
| Mention life outside work (family, health, hobbies) | 1.45× [1.31, 1.61] | 1.59× | 0.62 |
| Admit doubt, fear or insecurity | 1.37× [1.22, 1.55] | 1.39× | 0.61 |
| Express an emotion | 1.36× [1.31, 1.42] | 1.42× | 0.61 |
| Make a joke at your own expense | 1.27× [1.15, 1.39] | 1.29× | 0.60 |
| Admit a mistake you made | 1.19× [1.07, 1.32] | 1.19× | 0.58 |
| Write in the first person ("I") | 1.29× [1.24, 1.35] | 1.35× | 0.60 |

Humor in general did **not** replicate (1.08×, held-out 1.02×). Only humor aimed at yourself did.

### 4. Be concrete

| Do this | Reactions | Held-out | Unique |
|---|---|---|---|
| Mention a specific place (city, venue, country) | 1.33× [1.26, 1.39] | 1.32× | 1.14× |
| Open with a named person, company or product | 1.17× [1.12, 1.23] | 1.15× | 1.08× |
| Mention a named company or organization | 1.24× [1.19, 1.29] | 1.11× | |
| Include a specific example | 1.22× [1.17, 1.27] | 1.17× | |

### 5. Give it enough room

| | Reactions | Held-out |
|---|---|---|
| Under 80 words | **0.71×** [0.67, 0.75] | 0.72× |
| One block of text, no paragraph breaks | **0.72×** [0.67, 0.76] | 0.71× |
| Over 250 words | 1.17× [1.11, 1.22] | 1.18× |

Some of this is because longer posts hold more of the items above. With everything held fixed, length still adds about 1.12× per
2.7-fold increase in words, and single-block posts still lose (0.87×). Short paragraphs and bullet lists on their own made no difference.

## What goes with *fewer* reactions

| Avoid / be aware | Reactions | Held-out | Win rate |
|---|---|---|---|
| Asking readers to reply with a keyword or number from a list | **0.83×** [0.74, 0.94] | 0.79× | 0.40 |
| Advice not tied to anything you experienced | **0.81×** [0.78, 0.85] | 0.80× | 0.42 |
| Opening with a question | 0.86× [0.81, 0.90] | 0.79× | 0.43 |
| Hypothetical examples ("imagine a company that…") | 0.89× [0.85, 0.92] | 0.86× | 0.45 |
| Opening with a bold one-line claim | 0.89× [0.85, 0.92] | 0.92× | 0.46 |
| Including an external link | 0.90× [0.84, 0.95] | 0.84× | 0.47 |
| Summarizing an article or report | 0.91× [0.86, 0.96] | 0.84× | 0.46 |
| Asking the reader a question | 0.92× [0.88, 0.96] | 0.87× | 0.48 |
| Step-by-step processes or frameworks | 0.92× [0.88, 0.96] | 0.88× | 0.47 |
| Opening with a contradiction ("Everyone says X. They're wrong.") | 0.92× [0.89, 0.96] | 0.86× | 0.47 |
| Problem-then-solution structure | 0.93× [0.90, 0.96] | 0.89× | 0.48 |

The advice penalty largely disappears once the post is personal: in the joint model, "advice without experience" adds nothing beyond
the missing personal content (1.00×). **Practical:** if you share advice, anchor it in something that happened to you or to a named person.

## Common LinkedIn advice that showed **no** effect

| Tactic | Estimate (95% CI) | Status |
|---|---|---|
| Curiosity-gap opener ("Here's what nobody tells you…") | 0.94× [0.90, 0.99] | not confirmed; if anything slightly negative |
| Ending with a one-line takeaway or lesson | 0.97× [0.92, 1.02] | no effect |
| Announcing the number of items ("3 reasons…") | 0.97× [0.91, 1.02] | no effect |
| Asking readers to share their own experience | 1.04× [0.98, 1.11] | no effect |
| Asking readers to comment | 0.99× [0.93, 1.05] | no effect |
| Direct call to action | 0.96× [0.93, 1.00] | no effect |
| Challenging a common belief | 0.96× [0.92, 1.00] | no effect |
| Talking about AI | 1.02× [0.99, 1.06] | no effect |
| Short first line, short paragraphs, bullet lists | 1.01–1.03× | no effect |
| Hashtags (3+) | 1.25× overall, but only 1.09× on held-out authors | not replicated |
| Comment-for-a-resource bait, asking for likes or shares, asking to tag someone | 0.88–0.92× (screen only, 1–3% of posts) | no detectable effect, low power |

## Robustness

* **Not about photos:** 75 of 78 confirmed effects stay significant, in the same direction, when only posts with the same media type are compared.
  They shrink by about a quarter (median), so part of what looks like "content" is the photo that tends to go with it, such as event photos.
* **Not just length:** 68 of 78 stay significant with post length held fixed.
* **Same in both follower bands:** only 6 of 78 differ between the 1k–10k and 10k–20k bands, and all 6 point the same way in both
  (for example, gratitude: 1.69× vs 1.47×).
* **Rank-based check:** the win rates agree in direction with every multiplier, so a few viral posts are not driving the averages.

## How the research ran (auto-research loop)

1. **Existing questions (free).** The 147 Jev questions already answered, plus 15 free text checks: 57 confirmed.
2. **Round 1.** DeepSeek read, for 150 *generation* train authors, the post that most beat the joint model and the one that most fell short.
   It proposed 30 new questions; 18 were hand-written to split the biggest effects. Jev screened 46 on 250 *other* train authors ($1.30);
   20 passed and were scored on the remaining train authors and all test authors ($1.03). **13 confirmed.**
3. **Round 2.** The same loop on each author's second-best and second-worst posts: 25 DeepSeek + 10 hand-written questions.
   Screen $0.97. 10 passed, plus 6 popular-advice tactics that looked null and were measured with more data ($0.88). **8 confirmed.**
4. Joint model, checklist (train-only choice, held-out evaluation) and per-post-type breakdowns: free.

Where the new questions came from: the hand-written seeds were the most productive (16 of 21 new confirmations).
DeepSeek's proposals were mostly surface-level or rare, and 5 of them were confirmed (e.g. opening with a named entity, mentioning animals).

| Run | Posts | Questions | Cost |
|---|---|---|---|
| r1 screen | 4,694 | 46 | $1.30 |
| r1 confirm | 7,828 | 20 | $1.03 |
| r2 screen | 4,694 | 32 | $0.97 |
| r2 confirm | 7,828 | 16 | $0.88 |
| **Total** | | | **$4.19 of $5.00** |

## Prediction: compared with the dashboard model

Same 165 held-out authors (3,032 posts), the dashboard metric (how often the model ranks the better of two posts by the same author
higher), each row compared with the dashboard model by a paired bootstrap over authors. Scripts: `compare_dashboard.py`, `subset.py`.

| Model | Test accuracy | vs dashboard (95% CI) |
|---|---|---|
| Dashboard theme model (86 questions, 7 weights) | 0.613 | |
| **Blend: dashboard + ridge on old + new questions + text checks + length** | **0.623** | +0.010 [−0.001, +0.020] |
| Ridge on old + new questions + text checks + length (199) | 0.619 | +0.006 [−0.014, +0.024] |
| Research joint model (71 properties + length) | 0.614 | +0.001 [−0.021, +0.020] |
| 15 questions, chosen on train | 0.611 | −0.003 [−0.024, +0.018] |
| 5 questions, chosen on train | 0.607 | −0.006 [−0.024, +0.011] |
| 3 questions, chosen on train | 0.601 | −0.012 [−0.029, +0.005] |
| Research checklist (count of 9 items) | 0.592 | −0.022 [−0.038, −0.006] |

* **Combining old and new helps a little, not conclusively:** the best is a plain average of the dashboard model and the full ridge model,
  about one point better, with a CI that just touches zero. No weights were tuned.
* **A short list does almost as well.** Forward selection on train authors only (out-of-fold accuracy, 5 author folds) reaches the
  dashboard's level with about 15 questions; the first 5 already get within a point of it. The first five, in order: personal anecdote (1.21×),
  length (1.24× per 2.7-fold more words), points out a problem or gap (0.77×), own career milestone (1.75×), hypothetical example (0.85×).
  Full order and weights: `results/subset_items.csv`. Using only the old questions (all 667 train authors) needs about 20.
* The text-only content signal seems to top out around 0.61–0.62 with these questions.

## Caveats

* **Association, not causation.** A milestone post may do well because milestones are newsworthy, not because of how it is written.
  The within-author design rules out "who is posting", not "what happened that week".
* **Reactions are not reach or business value.** Congratulation-style content collects reactions from people the author already knows. These results
  say nothing about comments, leads or new followers.
* **Jev answers are judgments.** P(yes) ≥ 0.5 counts as "has it". Misjudged posts make effects look smaller, not bigger.
* **Multiple looks at the test authors.** The held-out authors were used once per run (existing questions, round 1, round 2), each with
  its own correction. The earlier theme model also used part of this test sample, but none of these questions were chosen with it.
* **Selection inside train.** The new questions' train estimates include the screening authors they were chosen on, so the train numbers are
  slightly optimistic. The held-out column is the unbiased one.
