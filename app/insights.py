"""Dashboard tabs built from the research results (research/results, no post text): the scorecards, and the research findings."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RESEARCH = Path(__file__).resolve().parents[1] / "research"
RESULTS = RESEARCH / "results"
GREEN, RED = "#2e7d32", "#c62828"
TEXT_LABELS = {  # free text checks that are not in the scorecard
    "text.under_80_words": "Under 80 words", "text.over_250_words": "Over 250 words",
    "text.first_line_under_60_chars": "First line under 60 characters", "text.short_paragraphs": "Short paragraphs",
    "text.has_bullet_or_numbered_list": "Bullet or numbered list", "text.ends_with_question": "Ends with a question",
}

MYTHS = {  # popular LinkedIn advice that was tested and did not hold up
    "r2_auto.uses_curiosity_gap_opening": "Curiosity-gap opener (\"Here's what nobody tells you...\")",
    "r2_seed.ends_with_takeaway_sentence": "End with a one-line takeaway",
    "r2_auto.states_number_of_items_before_list": "Announce the number of items (\"3 reasons...\")",
    "r2_seed.asks_readers_to_share_their_story": "Ask readers to share their own story",
    "call_to_action.asks_reader_to_comment": "Ask readers to comment",
    "call_to_action.includes_direct_call_to_action": "Direct call to action",
    "argument_claims.challenges_common_belief_or_practice": "Challenge a common belief",
    "topic.mentions_ai_technology": "Talk about AI",
    "text.first_line_under_60_chars": "Short first line",
    "text.short_paragraphs": "Short paragraphs",
    "text.has_bullet_or_numbered_list": "Bullet or numbered list",
    "structure_format.has_three_or_more_hashtags": "Three or more hashtags",
}


@st.cache_data
def load() -> dict:
    """Every research table the tabs use."""
    spec = json.loads((RESEARCH / "drivers.json").read_text())
    labels = dict(TEXT_LABELS)
    for v in spec["drivers"].values():
        for i, label, _ in v["members"]:
            labels[i] = label
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id")
    eff["label"] = eff["id"].map(labels).fillna(eff["question"])
    return {
        "effects": eff,
        "types": pd.read_csv(RESULTS / "archetypes.csv"),
        "length": pd.read_csv(RESULTS / "shape_length.csv"),
        "combos": pd.read_csv(RESULTS / "shape_combos.csv"),
        "variety": json.loads((RESULTS / "shape_variety.json").read_text()),
        "models": json.loads((RESULTS / "blend.json").read_text()) if (RESULTS / "blend.json").exists() else None,
    }


MODELS = [  # rows of the model comparison: (key in blend.json, shown name, what it reads, readable)
    ("Scorecard (64 questions)", "Scorecard (64 questions)", "Jev's answers + length", "yes, one weight per question"),
    ("Ridge, all 173 Jev questions", "Ridge regression", "Jev's answers to 173 questions", "no"),
    ("Gradient boosting, all 173 Jev questions", "Gradient boosting", "Jev's answers to 173 questions", "no"),
    ("TF-IDF words + ridge", "TF-IDF + ridge", "the words themselves", "no"),
    ("Scorecard + TF-IDF (stacked)", "Scorecard + TF-IDF", "both", "partly"),
    ("All six models (stacked)", "All models combined", "everything above", "no"),
]


def models_section(res: dict) -> None:
    """Scorecard vs standard ML models and blends, on the same held-out authors (research/blend.py)."""
    st.markdown("#### How the scorecard compares with standard ML models")
    rows = []
    for key, name, reads, readable in MODELS:
        r = res[key]
        d = r["vs_scorecard"]
        rows.append({"model": name, "held-out accuracy": 100 * r["test_pairwise"],
                     "vs scorecard (points)": "—" if d is None else
                     f"{100 * d['mean']:+.1f} ({100 * d['ci95'][0]:+.1f} to {100 * d['ci95'][1]:+.1f})",
                     "95% range": f"{r['test_ci95'][0]:.1%} – {r['test_ci95'][1]:.1%}",
                     "reads": reads, "readable": readable})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"held-out accuracy": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("All fitted on the same 667 train authors and scored once on the same 165 held-out authors. Accuracy: for two "
               "posts by the same author, how often the higher-scoring one got more reactions (50% = coin flip). vs scorecard: "
               "paired difference per author, with its 95% range from resampling authors.")
    st.markdown(
        "- **The scorecard beats TF-IDF** by 2.0 points (range excludes 0) and gradient boosting by 1.4, and ties ridge, "
        "while every weight stays readable.\n"
        "- **Combining adds at most about 1 point.** Scorecard + TF-IDF reached 62.4% (+0.8, range just above 0), but it "
        "was the best of 10 blends tried; the blend chosen on train authors alone gained only +0.3, within noise. TF-IDF "
        "helps a little because it is the only model that reads the words rather than Jev's answers.")


# ------------------------------------------------------------------ Research findings


def effect_table(t: pd.DataFrame, first: str) -> None:
    t = t.rename(columns={"label": first, "train_mult": "train authors", "test_mult": "held-out authors", "share": "share of posts"})
    st.dataframe(t[[first, "train authors", "held-out authors", "share of posts"]], hide_index=True, use_container_width=True,
                 column_config={"train authors": st.column_config.NumberColumn(format="%.2f×"),
                                "held-out authors": st.column_config.NumberColumn(format="%.2f×"),
                                "share of posts": st.column_config.NumberColumn(format="percent")})


def what_works_tab() -> None:
    d = load()
    eff = d["effects"]
    conf = eff[eff["confirmed"]].sort_values("train_mult", ascending=False).drop_duplicates("label")
    st.subheader("Research findings")
    st.markdown(
        f"**{len(eff)}** properties of the post text were tested and **{int(eff['confirmed'].sum())}** held up: significant on 667 "
        "train authors and again on 165 held-out authors. Every number compares a post with **the same author's** usual post, so "
        "audience size cannot explain it. These are associations, not proof that adding something causes more reactions.")
    st.info("**In one line:** reactions follow **people and real moments**: your news, the people around you, what actually "
            "happened. Generic advice, frameworks, links and engagement tricks do not help, and several hurt.")

    st.markdown("#### What goes with more reactions")
    effect_table(conf[conf["train_mult"] > 1].head(12), "property")
    st.markdown("#### What goes with fewer reactions")
    effect_table(conf[conf["train_mult"] < 1].sort_values("train_mult").head(10), "property")
    st.caption("Reactions compared with the same author's usual post: 1.50× = 50% more. Train authors: the 667 authors the "
               "properties were found on. Held-out authors: the 165 authors never used to choose anything, the independent "
               "check. A property counts as present when Jev's P(yes) is 0.5 or more.")

    st.markdown("#### Popular advice that did not hold up")
    e = eff.set_index("id")
    myths = pd.DataFrame([{"label": name, "train_mult": e.loc[i, "train_mult"], "test_mult": e.loc[i, "test_mult"], "share": e.loc[i, "share"]}
                          for i, name in MYTHS.items() if i in e.index])
    effect_table(myths, "advice")
    st.caption("None of these was confirmed: either no clear effect on train authors, or it did not repeat on held-out authors "
               "(hashtags: 1.29× on train, only 1.09× held-out).")

    st.markdown("#### Other findings")
    ln = d["length"].set_index(["outcome", "words"])["mult"]
    t = d["types"].sort_values("lift", ascending=False)
    combos = d["combos"]
    v = d["variety"]["per_sd_unusual"]
    st.markdown(
        f"- **Length:** under 40 words gets {ln[('lift', '0-40')]:.2f}× the reactions of a 120–160 word post, 80–120 words "
        f"{ln[('lift', '80-120')]:.2f}×. Past about 120 words, length adds nothing once the content is the same.\n"
        f"- **Kind of post matters less than what is in it:** of 8 kinds, only personal moments stand out "
        f"({t.iloc[0]['lift']:.2f}×) and short updates lag ({t.iloc[-1]['lift']:.2f}×); the rest are within a few percent.\n"
        "- **Comments and reposts follow different rules:** questions and problems raise comments per reaction but not "
        "reactions; thanks and events raise reactions but not discussion; job details raise reposts.\n"
        f"- **Specifics stack:** {int(combos['confirmed'].sum())} of {len(combos)} pairs tested had an extra effect together, e.g. "
        "a personal story that also names three or more people gets 2.23×. Mixing a story with a made-up example weakens it.\n"
        f"- **No fatigue, no novelty bonus:** effects do not fade for authors who use them often, and a post unlike the author's "
        f"usual ones does no better ({v['mult']:.2f}× per standard deviation).")

    if d["models"]:
        models_section(d["models"])

    with st.expander(f"Search all {len(eff)} tested properties"):
        q = st.text_input("Search", placeholder="e.g. people, link, question")
        t = eff if not q else eff[eff["label"].str.contains(q, case=False) | eff["question"].str.contains(q, case=False)]
        t = t.sort_values("train_mult", ascending=False).rename(columns={"label": "property", "train_mult": "train authors",
                                                                    "test_mult": "held-out authors", "share": "share of posts"})
        st.dataframe(t[["property", "train authors", "held-out authors", "share of posts", "status"]], hide_index=True,
                     use_container_width=True, height=400, column_config={
                         "train authors": st.column_config.NumberColumn(format="%.2f×"),
                         "held-out authors": st.column_config.NumberColumn(format="%.2f×"),
                         "share of posts": st.column_config.NumberColumn(format="percent")})
    st.caption("Full write-up: research/REPORT.md.")


# ------------------------------------------------------------------ Scorecard


def scorecard_tab(b: dict) -> None:
    """The scorecards behind Evaluate a post: accuracy of each size, and every question's weight."""
    common = b["common"]
    st.subheader("The scorecards")
    st.markdown(
        f"Each scorecard is one weight per yes/no question plus length, added up. Weights were fitted on "
        f"{common['train_authors']} train authors, comparing each author's posts only with their own; a question that helped there can "
        f"only add, one that hurt can only subtract. The 5 and 10-question scorecards are the best "
        f"combinations of common questions (each in at least 10% of train posts) found by searching on train authors (add the question that helps most, then swap questions while it "
        f"improves); 20 and 40 keep the questions that move the full score most. Each is refitted on its own questions. Accuracy is measured once, on {common['test_authors']} held-out authors "
        f"({common['test_posts']:,} posts) never used to fit or choose anything.")
    sizes = sorted(b["sizes"], key=int)
    rows = []
    for k in sizes:
        m = b["sizes"][k]["metrics"]
        lo, hi = m["vs_full"]["ci95"]
        rows.append({"scorecard": f"{k} questions" + (" (full)" if k == b["default"] else ""), "Jev questions": m["jev_questions"],
                     "text checks": m["questions"] - m["jev_questions"], "held-out accuracy": 100 * m["test_pairwise"],
                     "95% range": f"{m['test_pairwise_ci95'][0]:.1%} – {m['test_pairwise_ci95'][1]:.1%}",
                     "vs full (points)": "—" if k == b["default"] else f"{100 * m['vs_full']['mean']:+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "held-out accuracy": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Held-out accuracy: for two posts by the same author, how often the higher-scoring one got more reactions (50% = coin "
               "flip). Every content-only model tried lands at 59–62%; the previous dashboard model got 61.3%. vs full: the paired "
               "difference from the full scorecard, with its 95% range over authors.")

    pick = st.selectbox("Show the weights of", sizes, index=sizes.index(b["default"]),
                        format_func=lambda k: f"{k} questions" + (" (full)" if k == b["default"] else ""))
    c = b["sizes"][pick]
    ln = c["length"]
    w = pd.DataFrame([{"Question": i["label"] + (" (text check)" if i["id"].startswith("text.") else ""), "Weight": i["weight"],
                       "Jev's exact question": "Counted from the text" if i["id"].startswith("text.") else i["question"]}
                      for i in c["items"]]
                     + [{"Question": f"Post length, not a question (per doubling of words, up to {ln['cap']})", "Weight": ln["weight"] * np.log(2),
                         "Jev's exact question": "Counted from the text"}])
    w = w.sort_values("Weight", ascending=False)
    st.dataframe(w.style.format({"Weight": "{:+.2f}"}).map(lambda v: f"color: {GREEN if v > 0 else RED}", subset=["Weight"]),
                 hide_index=True, use_container_width=True, height=min(38 + 35 * len(w), 700))
    st.caption("Weight: how much the score changes from a clear no to a clear yes, for the same author. A weight of +0.25 means "
               "about 1.28× the reactions (exp of the weight); −0.10 means about 0.90×. Text checks (hashtags, emoji, links…) are "
               "counted directly from the text, not asked of Jev.")

    st.markdown("#### Does a higher score mean more reactions? (held-out authors)")
    bd = pd.DataFrame(c["bands"])
    st.dataframe(bd[["name", "share", "predicted", "actual", "above_usual", "hit_rate", "flop_rate"]], hide_index=True,
                 use_container_width=True, column_config={
                     "name": "score band", "share": st.column_config.NumberColumn("share of posts", format="percent"),
                     "predicted": st.column_config.NumberColumn("predicted", format="%.2f×"),
                     "actual": st.column_config.NumberColumn("actual", format="%.2f×"),
                     "above_usual": st.column_config.NumberColumn("beat the author's usual", format="percent"),
                     "hit_rate": st.column_config.NumberColumn("got 2× or more", format="percent"),
                     "flop_rate": st.column_config.NumberColumn("got ½ or less", format="percent")})
    st.caption("Reactions vs the author's usual post. Bands are cut at the 10th, 30th, 70th and 90th percentile of train scores.")

    st.markdown("#### Comparing two drafts by the same author")
    g = pd.DataFrame(c["pair_gap"])
    g["score gap"] = g.apply(lambda r: f"{r['lo']:.2f} or more" if pd.isna(r["hi"]) else f"{r['lo']:.2f} – {r['hi']:.2f}", axis=1)
    g["as a ratio"] = g.apply(lambda r: f"{np.exp(r['lo']):.2f}× or more" if pd.isna(r["hi"])
                              else f"{np.exp(r['lo']):.2f}× – {np.exp(r['hi']):.2f}×", axis=1)
    st.dataframe(g[["score gap", "as a ratio", "share", "accuracy"]], hide_index=True, use_container_width=True, column_config={
        "share": st.column_config.NumberColumn("share of pairs", format="percent"),
        "accuracy": st.column_config.NumberColumn("higher score did better", format="percent")})
