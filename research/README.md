# Auto research: what content gets more reactions?

Goal: find properties of a post's **content** that go with more reactions, each backed by a
statistical test that was decided before the test data was looked at, and turn them into
plain guidance for writers. Content only: attached media is never an input (it appears only in one
robustness check, to see whether an effect is really about photos).

Results: [FINDINGS.md](FINDINGS.md). Raw tables: `results/`.

## How every claim is tested

* **Same author only.** `lift` compares each post with the same author's nearby posts, and on top
  of that every regression removes each author's mean. "1.20×" means about 20% more reactions than
  the same person's usual post. Who the author is cannot create an effect.
* **Discover on train, confirm on test.** A property counts as *confirmed* only if it passes
  Benjamini-Hochberg q < 0.05 on the 667 train authors **and** points the same way on the 165 held-out
  test authors (BH-adjusted one-sided p < 0.05). Standard errors are clustered by author.
* **Win rate.** For a random post with the property and one without, from the same author, how often
  the one with it got more reactions (0.50 = no difference). Rank-based, so a few viral posts cannot drive it.
* **Unique effect.** Properties that co-occur (thanks + pride + event) are grouped, and one joint model
  shows which still matter with the others held fixed.
* **Robustness.** Each effect is re-estimated comparing only posts with the same media type, with
  post length held fixed, and separately for the 1k–10k and 10k–20k follower bands.

This is observational data: a confirmed property is reliably *associated* with more reactions for
the same author. It is not proof that adding it to any post will cause more reactions.

## The research loop

| Step | Script | Jev cost |
|---|---|---|
| 1. Test the 147 existing Jev questions + 15 free text checks | `effects.py --name existing` | $0 (cached answers) |
| 2. Joint model: which effects are unique | `joint.py` | $0 |
| 3. DeepSeek reads the posts the joint model gets most wrong (150 *generation* train authors only) and proposes new questions; hand-written seeds split the biggest effects | `hypotheses.py --round N` | $0 (DeepSeek) |
| 4. Jev scores candidates on 250 *screening* train authors; liberal screen | `score.py`, `screen.py` | paid |
| 5. Jev scores the survivors on the remaining train authors and all test authors; same discovery → confirmation test as step 1 | `score.py`, `effects.py --name new` | paid |
| 6. Joint model, train-only checklist evaluated on test authors, levers by post type | `joint.py`, `playbook.py`, `tables.py` | $0 |

Author sets are disjoint and fixed (seeded, `data/processed/research/authors.json`). Test authors are
never shown to DeepSeek and never used to choose anything. Every paid run is logged in
`results/jev_ledger.json`, and `score.py` refuses a run that would take the total past $5.

```bash
uv run python research/effects.py --name existing
uv run python research/joint.py
uv run python research/hypotheses.py --round 1
uv run python research/score.py --round 1 --sets screen --name r1_screen
uv run python research/screen.py --round 1 --judgments data/processed/research/judgments_r1_screen
uv run python research/score.py --round 1 --sets rest test --ids-from research/results/screen_r1.csv --name r1_confirm
R=data/processed/research
uv run python research/effects.py --name r1 --judgments $R/judgments_r1_screen $R/judgments_r1_confirm --bank $R/bank_round1.json --no-text --min-share 0.01
uv run python research/joint.py --effects existing r1 --name r1 --judgments data/processed/judgments_full_{train,test} $R/judgments_r1_{screen,confirm}
# round 2: same steps with --round 2 (seeds_round2.json), screen.py --joint r1 --base ..., then effects.py --name r2
J=(data/processed/judgments_full_{train,test} $R/judgments_r{1,2}_{screen,confirm})
uv run python research/joint.py --effects existing r1 r2 --name r2 --judgments $J
uv run python research/joint.py --effects existing r1 r2 --name r2_train --select train --judgments $J
uv run python research/playbook.py --effects existing r1 r2 --joint r2_train --judgments $J
uv run python research/tables.py --effects existing r1 r2 --joint r2 > research/results/tables.md
```

Data stays in `data/processed/research/` (gitignored). `results/` holds only statistics and question
wording, no post text.
