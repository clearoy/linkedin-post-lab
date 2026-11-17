# What makes a LinkedIn post do well: research report

**Data.** 15,331 posts from 832 authors with 1k–20k followers, Jul–Oct 2026. The analysis uses the post text only, never attached media.

**Measure.** Every number compares a post with **the same author's** usual post, so audience size and who is posting cannot create an effect.

**Jev spend.** $4.72 of the $5.00 research budget (ledger in [results/jev_ledger.json](results/jev_ledger.json)).

**Deliverables.**
- This report.
- The dashboard (`uv run streamlit run app/dashboard.py`) with four tabs: **Evaluate a post** (choose a 10, 20, 40 or full 64-question scorecard; see each question's weight and contribution), **Scorecard** (accuracy of each size and every weight), **Research findings** (the main results in a few tables), **Pipeline** (the steps from data to scorecard).
- The first-pass write-up with every single-property table: [FINDINGS.md](FINDINGS.md).

---

## 1. The short version

1. **Reactions follow people and real moments.** The largest effects all involve a real person or a real event:
   - your own career milestone: 2.00×
   - crediting people for your success: 1.87×
   - thanking a specific person: 1.65×
   - naming three or more people: 1.59×
   - a turning point: 1.52×
   - an event you took part in: 1.49×
2. **Lecture-style content does worse.** Each of these lowers reactions by 7–19%:
   - advice not tied to anything you lived through
   - frameworks and step-by-step processes
   - made-up examples
   - bold-claim, contradiction or question openers
   - links out

   Advice itself is not the problem. Advice anchored in your own experience or a named person loses nothing.
3. **What you put in the post matters more than what kind of post it is.**
   - Of eight kinds of post, only personal moments stand out (1.46×) and short updates lag (0.83×).
   - Product posts, opinion pieces, job openings and news are all within ±3% of the author's usual post.
   - Inside every kind of post, the same levers work: people, personal experience, specifics.
4. **Length matters up to about 120 words, then stops.**
   - Under 40 words: 0.63×. 80–120 words: 0.90×.
   - Past 120 words, length adds nothing by itself. Long posts do better only because they hold more of the content above.
5. **Comments and reposts follow different rules from reactions.**
   - Questions, problems and risks raise comments *per reaction* (up to 1.21×) while slightly lowering reactions.
   - Thanks and events collect applause, not discussion.
   - Job openings, requirements and application steps get about 1.2× the reposts with no change in reactions.
6. **Specifics stack, and lectures dilute stories.**
   - A personal story that also names three people gets 2.23×; one that names a place gets 1.89×. Both are more than the two parts alone.
   - A personal story mixed with a made-up example, or with arguing for a method, loses part of its boost. (13 confirmed interactions.)
7. **Popular LinkedIn advice mostly does nothing.** None of these had an effect:
   - curiosity-gap openers
   - takeaway endings
   - "3 reasons…" previews
   - asking for comments or stories
   - calls to action
   - challenging a belief
   - AI topics
   - short first lines, short paragraphs, bullets

   Hashtags looked strong on train authors (1.25×) but did not hold up on held-out authors (1.09×).
8. **No sign of fatigue or a novelty bonus.**
   - Effects do not wear off for authors who use them often (0 of 71 properties).
   - A post unlike the author's usual posts does neither better nor worse (0.997× per standard deviation).
9. **A scorecard turns all of this into an evaluator.**
   - **Accuracy:** it ranks two posts by the same held-out author correctly 61.6% of the time, the same as the best black-box model, so it gives up nothing on prediction.
   - **Calibration:** on held-out authors, the top score band got 1.66× the author's usual reactions and the bottom band 0.84×.
   - **Explanations:** every suggestion it makes traces to a tested, confirmed property.

These are associations measured within author, not controlled experiments; see [Caveats](#8-caveats).

---

## 2. How every claim was tested

| Step | What it guards against |
|---|---|
| `lift = log((reactions + 2) / (author's local median + 2))`, and on top of that every regression removes each author's mean | Big accounts, prolific posters, the author's general popularity |
| **Discover** on 667 train authors (Benjamini-Hochberg q < 0.05 over all properties tested) | Fishing through 196 properties |
| **Confirm** on 165 held-out authors never used to choose anything: same direction, BH-adjusted one-sided p < 0.05 | Effects that do not generalise to new people |
| Standard errors clustered by author; win rates (rank-based) next to every multiplier | A few viral posts driving an average |
| Robustness: same media type only, same length, each follower band separately | "It's really the photo", "it's just longer", "only big accounts" |

Where the 196 properties came from:
- **147 existing Jev questions** from the original pipeline.
- **15 free text checks** (links, hashtags, length, paragraphs…).
- **Two auto-research rounds (36 new questions).**
  - DeepSeek read, for 150 *generation* train authors only, the posts that most beat or missed the model, and proposed questions.
  - Hand-written seeds split the biggest effects.
  - Jev screened the candidates on 250 other train authors, and the survivors were measured on everyone else.

Screening had to pass on its own author set before the confirmation authors were scored.

---

## 3. What works: the drivers

Every property that passed discovery on train authors was grouped by meaning into 14 drivers. Each driver below shows the strongest confirmed properties; all multipliers are reactions vs the same author's usual post. Every property's numbers can be searched in the dashboard's **Research findings** tab. The grouping is for this report only; the scorecard and the dashboard work question by question.

| Driver | Direction | Strongest confirmed properties (all authors · held-out authors) |
|---|---|---|
| **Real people** | ▲ | credit others for your success 1.87× · 1.85×; thank a specific person 1.65× · 1.56×; name 3+ people 1.59× · 1.40×; one person's story 1.50× · 1.66× |
| **Your own news and real moments** | ▲ | own career milestone 2.00× · 2.34×; turning point 1.52× · 1.60×; pride 1.53×; event you took part in 1.49× · 1.53×; something happening today 1.30× · 1.42× |
| **Personal voice** | ▲ | life outside work 1.45× · 1.59×; personal anecdote 1.42× · 1.51×; doubt or fear 1.37×; how you felt 1.36×; joke at your own expense 1.27×; a mistake you made 1.19× |
| **Concrete details** | ▲ | a specific place 1.33× · 1.32×; open in a specific moment 1.25× · 1.31×; a named company 1.24×; a concrete example 1.22×; open with a named person or company 1.17× |
| **Company and market news** | ▲ | mostly did not replicate on its own (funding, money amounts, launches); small weights only |
| **Contrast and analogy** | ▲ | contrast two things 1.09×; analogy 1.07× (small) |
| **Hashtags, emoji and emphasis** | ▲ | emoji 1.16×; hashtags 1.25× on train but 1.09× held-out, **not confirmed** |
| **Lists** | ▲ | a list of 3+ items 1.10× (small; also more reposts) |
| **Lecture mode** | ▼ | advice not tied to your experience 0.81× · 0.80×; hypothetical example 0.89×; framework 0.92×; tips for the reader 0.93× |
| **Problem and risk framing** | ▼ | problem/gap 0.94×; group problem 0.92× (train only; raise comments) |
| **Hook-style openers** | ▼ | open with a question 0.86× · 0.79×; bold one-line claim 0.89×; contradiction 0.92× |
| **Sends readers elsewhere** | ▼ | external link 0.90× · 0.84×; summarise an article 0.91× · 0.84× |
| **No paragraph breaks** | ▼ | one block of text 0.72× · 0.71× (still 0.87× with length fixed) |
| **Asks the reader** | ▼ | reply with a keyword or number 0.83× · 0.79×; asks the reader a question 0.92× · 0.87× |

How much each driver can move a post, between a typical low and a typical high post with everything else held fixed:
- **Largest swings:** real people (0.93–1.19×), concrete details (0.90–1.11×) and your own news (0.94–1.11×).
- **Smallest swings:** hooks, asking the reader and contrast (about ±2%).

**Robustness of the 78 confirmed properties:**
- **Same media type:** 75 stay significant comparing only posts with the same media type. They shrink by about a quarter, so part of "event" content is the event photo.
- **Same length:** 68 stay significant with length held fixed.
- **Follower bands:** only 6 differ between the 1k–10k and 10k–20k bands, and all 6 point the same way in both.
- **Win rates:** they agree in direction with every multiplier.

---

## 4. Kinds of post

The 147 original questions were clustered (k-means on train posts), and the eight clusters were named by hand:

| Kind of post | Share | Median words | Reactions (95% CI) | Held-out | Comments | Reposts | Got 2×+ | Got ½ or less |
|---|---|---|---|---|---|---|---|---|
| Personal moment or thank-you | 12% | 151 | **1.46×** [1.40, 1.53] | 1.48× | 1.24× | 1.03× | 35% | 9% |
| Product or service promotion | 12% | 224 | 1.03× [1.00, 1.06] | 1.00× | 1.08× | 1.07× | 17% | 11% |
| Job opening | 7% | 156 | 1.01× [0.98, 1.05] | 1.02× | 0.99× | 1.04× | 18% | 9% |
| Discussion question | 14% | 215 | 0.97× [0.94, 0.99] | 0.95× | 1.03× | 0.99× | 14% | 12% |
| Event or link promotion | 11% | 104 | 0.97× [0.93, 1.01] | 0.91× | 0.90× | 1.09× | 17% | 16% |
| Industry news and analysis | 12% | 225 | 0.97× [0.94, 1.00] | 0.94× | 1.00× | 1.01× | 15% | 14% |
| Opinion and advice | 15% | 184 | 0.97× [0.94, 0.99] | 1.00× | 1.02× | 0.97× | 13% | 11% |
| Short update | 18% | 28 | **0.83×** [0.80, 0.85] | 0.87× | 0.84× | 0.90× | 12% | 21% |

Inside each kind of post, the confirmed properties were re-tested. A lever counts if it is BH-significant within that kind and points the same way on held-out authors.
- **Opinion and advice:** life outside work 1.62×, one person's story 1.47×, your own background 1.43×, how long you've been at it 1.35×. Generic advice 0.85× and bold-claim openers 0.87× hurt.
- **Product or service promotion:** how long you've been at it 1.42×, one person's story 1.37×, thanks 1.35×, naming three people 1.43×. Problem-then-solution 0.82× and asking the reader a question 0.88× hurt.
- **Discussion question:** self-deprecating humour 1.51×, an event you attended 1.38×, a place 1.28×. Hypothetical examples and generic advice hurt (0.85×).
- **Event or link promotion:** thanks 1.64×, pride 1.43×, a personal anecdote 1.37×.

The per-type tables are in `research/results/archetype_levers.csv`.

---

## 5. Length, comments, reposts, combinations, fatigue

**Length.** Each band is compared with a 120–160-word post by the same author:

| Words | 0–40 | 40–80 | 80–120 | 160–200 | 250–300 | 400+ |
|---|---|---|---|---|---|---|
| Reactions | 0.63× | 0.77× | 0.90× | 1.01× | 1.04× | 1.14× |
| Reactions, same content | 0.74× | 0.82× | 0.93× | 1.01× | 1.01× | 1.02× |
| Comments | 0.69× | 0.80× | 0.92× | 1.08× | 1.14× | 1.34× |

**Beyond reactions.** The same discover-then-confirm test was run on four more outcomes:

| Outcome | Confirmed properties |
|---|---|
| Comments | 57 |
| Reposts | 45 |
| 2× hits | 71 |
| Flops | 50 |

- **Comments mostly follow reactions** (the effects correlate 0.85).
- **Reposts follow a different logic** (correlation 0.33 with reactions): they reward content that is useful to pass on.
- **Comments per reaction** are where the trade-off shows:

| Raises comments per reaction | Lowers it |
|---|---|
| Describes a technology limitation or risk 1.21× | Thanks a specific person 0.79× |
| Asks readers to comment 1.16× | References a specific event 0.80× |
| Points out a problem or gap 1.15× | Announces you're taking part 0.80× |
| Ends asking about the reader's experience 1.15× | Describes an event you attended 0.81× |

**2× hits and flops.**
- **Baseline:** 17% of posts get at least double the author's usual reactions, and 13% get half or less.
- **Gratitude to a specific person:** +21 points of hit chance and −11 points of flop chance.
- **Crediting others:** +26 points of hit chance.
- **Under 80 words:** −11 points of hit chance and +12 points of flop chance.

**Combinations.** 219 pairs of the strongest properties were tested for an extra effect when they appear together. 19 passed on train and 13 were confirmed on held-out authors.

| Pattern | Example | Alone | Together |
|---|---|---|---|
| Specifics stack | personal anecdote + names 3+ people | 1.36× and 1.30× | **2.23×** |
| | personal anecdote + a place | 1.33× and 1.22× | **1.89×** |
| A lecture dilutes a story | one person's story + a made-up example | 1.57× and 0.92× | 1.13× |
| | personal anecdote + arguing for a method | 1.57× and 1.00× | 1.29× |
| | one person's story + a client/founder story | 1.62× and 1.12× | 1.28× |

**No fatigue, no novelty bonus.**
- **Fatigue:** no property's effect shrinks significantly for authors who use it in more of their posts. The median of positive effects is 1.27× / 1.29× / 1.26× for authors who use the property rarely, sometimes and often.
- **Novelty:** a post that is unlike the author's usual posts (distance from their average answer profile) does neither better nor worse: 0.997× per standard deviation [0.977, 1.019].

---

## 6. The scorecard: evaluating a new post

**Design.** It is built for being interpretable, being applied to new posts, and helping improve them:
- **One weight per property.** Every one of the 106 properties that passed discovery on train authors gets its own weight on Jev's P(yes), fitted jointly within author.
- **Directions fixed in advance.** A property that helped on train authors can only add and one that hurt can only subtract, so no weight can flip sign because of a correlated neighbour. 64 keep a non-zero weight: that is the full scorecard.
- **Length** enters as log(min(words, 120)).
- **Chosen on train authors only.** The length cap, the ridge strength and a shrink factor (0.95, so out-of-fold predictions are calibrated) were all set on train. The grouping into drivers used meaning only, not test results.
- **Training data.** It trains on all 667 train authors after the 150 generation authors were scored on the 26 new questions it uses ($0.53 of Jev).
- **Smaller sizes.** The 10, 20 and 40-question scorecards keep the questions that move the full score most (|weight| × spread of the answers on train posts) and are refitted on those alone, each with its own shrink factor and calibration. A question whose refitted weight is exactly 0 is swapped for the next one, so every question counts.
- **Testing.** The 165 held-out authors were used once per size.

**Accuracy on held-out authors** (3,032 posts):

| Model | Pairwise accuracy | vs previous dashboard model (paired 95% CI) | Readable |
|---|---|---|---|
| Average of the previous model and ridge on all 199 questions | 0.623 | +0.010 [−0.001, +0.020] | no |
| **Scorecard** | **0.616** [0.594, 0.639] | +0.003 [−0.013, +0.018] | yes, one signed weight per property |
| Ridge on all questions | 0.616–0.619 | +0.002 to +0.006 (within noise) | no |
| Previous dashboard model (7 themes) | 0.613 | — | partly (themes bundle many questions) |
| 5 questions chosen on train | 0.607 | −0.006 [−0.024, +0.011] | yes |
| 9-item checklist count | 0.592 | −0.021 [−0.038, −0.006] | yes |
| Format only (media, length, posting time) | 0.579 | | |

Every content-only model lands between 0.59 and 0.62. Text-only yes/no questions seem to top out there, and the differences between the good models are within noise. The scorecard was chosen because it ties the best while every weight and every suggestion can be traced to a tested property.

**Calibration on held-out authors.** Bands are cut at train score percentiles 10/30/70/90. "Predicted" is the calibrated expectation for one post by an author you know nothing else about.

| Band | Share | Predicted | Actual (95% CI) | Beat author's usual | Got 2×+ | Got ½ or less | Comments | Reposts |
|---|---|---|---|---|---|---|---|---|
| Low | 8% | 0.82× | 0.84× [0.75, 0.92] | 28% | 7% | 16% | 0.96× | 1.00× |
| Below average | 18% | 0.92× | 0.89× [0.82, 0.97] | 38% | 9% | 19% | 1.05× | 1.09× |
| Average | 42% | 1.05× | 0.98× [0.94, 1.03] | 39% | 12% | 14% | 1.09× | 1.07× |
| Above average | 21% | 1.23× | 1.25× [1.16, 1.34] | 58% | 25% | 10% | 1.25× | 1.13× |
| High | 11% | 1.56× | 1.66× [1.46, 1.92] | 71% | 39% | 6% | 1.52× | 1.21× |

**Two drafts by the same author.**
- **Within one author, scores calibrate directly:** a score difference d means about exp(d) times the reactions. The calibration slope on held-out authors is 1.03 [0.88, 1.17].
- **Why a lone post's prediction is smaller:** for a single post by an unknown author the slope is 0.5. Part of any score is the author's habitual style, which an author-relative measure cancels out.
- **How often the higher score wins**, for same-author pairs:

| Score gap as a ratio | < 1.1× | 1.1–1.3× | 1.3–1.65× | 1.65–2.2× | 2.2–3.3× | > 3.3× |
|---|---|---|---|---|---|---|
| Higher score did better | 52% | 55% | 64% | 75% | 85% | 91% |
| Share of pairs | 28% | 27% | 25% | 13% | 6% | 1% |

**Smaller scorecards** (held-out authors, paired difference from the full scorecard over authors):

| Scorecard | Jev questions | Text checks | Pairwise accuracy (95% CI) | vs full (points, 95% CI) | Same-author calibration slope |
|---|---|---|---|---|---|
| 10 questions | 9 | 1 | 59.5% [57.1, 61.8] | −2.1 [−3.5, −0.7] | 0.96 |
| 20 questions | 18 | 2 | 60.9% [58.6, 63.4] | −0.7 [−1.6, +0.2] | 0.99 |
| 40 questions | 36 | 4 | 61.4% [59.2, 63.6] | −0.2 [−0.6, +0.1] | 1.01 |
| 64 questions (full) | 57 | 7 | 61.6% [59.4, 63.9] | — | 1.02 |

20–40 questions keep almost all the accuracy; 10 lose about 2 points. Weights in a small scorecard are larger than in the full one, because each kept question also stands in for related ones that were dropped (the milestone question is +0.47 in the full scorecard and +0.81 in the 10-question one).

**What the Evaluate tab shows for a draft.** Jev answers only the chosen scorecard's questions (up to 57, about $0.0004 per draft; switching to a smaller size re-scores without asking again).
- **Headline:** the score, the expected reactions vs the author's usual post, and the score band with what held-out posts in it got.
- **One table, one row per question:** Jev's answer, the question's weight, and its contribution = weight × (answer − the average post's answer). The contributions add up to the score, so negative rows are what pull the post down: a helpful question answered no, or a harmful one answered yes.
- **Compare mode:** both drafts' answers and contributions side by side, sorted by where they differ most, with the expected ratio and how often a gap that size held up.

**Example** (two drafts written for the test, not from the dataset, full scorecard):
- **A five-year work anniversary** that names and thanks two colleagues and a city: score +1.20, 1.96× the author's usual, High band. The milestone (+0.43) and crediting people (+0.20) contribute most.
- **A generic 4-step onboarding framework** with a question and a link: score −0.69, 0.75×, Low band. The framework (−0.08), the link and pointing readers elsewhere pull it down, and it has none of the people or moments.

---

## 7. Practical guide for writers

1. **Put real people in it.** Name who was involved and what they did. Credit people when you reach something.
2. **Tell what happened to you.** Your news, an event, a first-hand lesson, how it felt, a doubt or a mistake.
3. **Be concrete.** The place, the company, the number, the moment. Open with a person or a moment, not a claim or a question.
4. **If you give advice, earn it.** Tie it to something you or a named person did. Drop made-up examples and frameworks for their own sake.
5. **Give it room.** At least about 120 words, in paragraphs. Longer than that is fine but not a lever by itself.
6. **Keep readers in the post.** Make the point without a link. Skip "comment YES" and keyword-reply bait.
7. **Pick the goal.**
   - For discussion, a question or a problem raises comments per reaction at a small cost in reactions.
   - For reposts, useful specifics (who should apply, how) help.
8. **Ignore the hacks.** Hooks, curiosity gaps, takeaway endings, hashtags, short first lines and bullet formatting showed no reliable effect.

---

## 8. Caveats

- **Association, not causation.** A milestone post may do well because milestones are newsworthy, not because of how it is written. The within-author design rules out "who is posting", not "what happened that week".
- **Reactions are not reach or business value.** Congratulation-style content collects reactions from people the author already knows. The comments and reposts sections help, but nothing here measures leads or followers.
- **Jev answers are judgments.** P(yes) ≥ 0.5 counts as "has it". Misjudged posts make effects look smaller, not bigger.
- **Multiple looks at the held-out authors.** The held-out authors were used once per analysis (single properties, two rounds of new questions, interactions, post types, scorecard), each with its own correction. Nothing was chosen with them.
  - The earlier theme model's design partly used an earlier 45-author test sample, so its test score may be slightly optimistic.
  - The new questions were proposed from the 150 generation authors, who are now also in the scorecard's training data. That affects train fit only, not the held-out scores.
- **A contribution is the scorecard's own arithmetic.** Changing an answer moves the score by its weight, but whether reactions move with it is the association above, not a guarantee.
- **The data covers 1k–20k-follower accounts and four months of 2026.**

---

## 9. Reproduce

New scripts in this round (all free except the one Jev run):

```bash
uv run python research/outcomes.py      # comments, reposts, hits, flops for all 196 properties
uv run python research/shape.py         # length curve, wear-off, combinations, variety
uv run python research/archetypes.py    # kinds of post and levers inside each
uv run python research/score.py --round 1 --sets generate --ids-from <r1 ids> --name r1_generate  # $0.32
uv run python research/score.py --round 2 --sets generate --ids-from <r2 ids> --name r2_generate  # $0.21
uv run python research/scorecard.py     # fit, calibrate, test 10/20/40/64-question scorecards; writes the dashboard bundle
uv run streamlit run app/dashboard.py
```

The `--ids-from` files list the r1_/r2_ question ids that appear in [drivers.json](drivers.json).

**Outputs:**
- Statistics (no post text) are in `research/results/`.
- The dashboard bundle (`data/processed/research/scorecard.json`) and all Jev answers stay in `data/` (gitignored).
