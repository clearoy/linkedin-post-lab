"""Dashboard tabs built from the research results (research/results, no post text): what works, and the evidence."""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RESEARCH = Path(__file__).resolve().parents[1] / "research"
RESULTS = RESEARCH / "results"
GREEN, RED, GREY = "#2e7d32", "#c62828", "#9e9e9e"
LOG_AXIS = alt.Axis(values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], format=".2~f")
TEXT_LABELS = {  # free text checks that are not in drivers.json
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


def x(m: float) -> str:
    return f"{m:.2f}×"


@st.cache_data
def load() -> dict:
    """Every research table the tabs use."""
    spec = json.loads((RESEARCH / "drivers.json").read_text())
    labels, driver_of = dict(TEXT_LABELS), {}
    for d, v in spec["drivers"].items():
        for i, label, _ in v["members"]:
            labels[i], driver_of[i] = label, v["name"]
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id")
    eff["label"] = eff["id"].map(labels).fillna(eff["question"])
    eff["driver"] = eff["id"].map(driver_of).fillna("—")
    out = pd.read_csv(RESULTS / "outcomes.csv")
    for o in ["comments", "reposts", "talk"]:
        g = out[out["outcome"] == o].set_index("id")
        eff[o] = eff["id"].map(g["mult"])
        eff[o + "_confirmed"] = eff["id"].map(g["confirmed"])
    for o in ["hit", "flop"]:
        eff[o] = eff["id"].map(out[out["outcome"] == o].set_index("id")["points"])
    return {
        "spec": spec, "labels": labels, "effects": eff,
        "score": json.loads((RESULTS / "scorecard.json").read_text()),
        "types": pd.read_csv(RESULTS / "archetypes.csv"),
        "levers": pd.read_csv(RESULTS / "archetype_levers.csv"),
        "length": pd.read_csv(RESULTS / "shape_length.csv"),
        "combos": pd.read_csv(RESULTS / "shape_combos.csv"),
        "wearoff": pd.read_csv(RESULTS / "shape_wearoff.csv"),
        "variety": json.loads((RESULTS / "shape_variety.json").read_text()),
        "outcomes": json.loads((RESULTS / "outcomes_summary.json").read_text()),
        "compare": json.loads((RESULTS / "compare_dashboard.json").read_text()),
        "subset": json.loads((RESULTS / "subset.json").read_text()),
        "ledger": json.loads((RESULTS / "jev_ledger.json").read_text()),
    }


def label(d: dict, i: str) -> str:
    return d["labels"].get(i) or d["effects"].set_index("id")["question"].get(i, i)


# ------------------------------------------------------------------ What works


def what_works_tab() -> None:
    d = load()
    eff = d["effects"]
    conf = eff[eff["confirmed"]]
    st.subheader("What content goes with more reactions")
    st.markdown(
        f"Every number compares posts **by the same author**, so audience size and who is posting cannot create an effect. "
        f"**{len(eff)}** content properties were tested; **{len(conf)}** were confirmed: significant on 667 train authors "
        "after correcting for multiple tests, and in the same direction on 165 held-out authors. Text only: attached media is "
        "never an input. These are associations, not proof that adding something will cause more reactions.")

    c = st.columns(4)
    c[0].metric("Own career milestone", "2.00×", help="Reactions vs the same author's usual post. Held-out authors: 2.34×.")
    c[1].metric("Credit people for your success", "1.87×", help="Held-out authors: 1.85×.")
    c[2].metric("Advice not tied to your experience", "0.81×", help="Held-out authors: 0.80×.")
    c[3].metric("Under 80 words", "0.71×", help="Held-out authors: 0.72×.")
    st.info("**The one-line version:** reactions follow **people and real moments**: your news, the people around you, and "
            "what actually happened. Generic advice, frameworks, links and engagement tricks do not help, and several hurt.")

    drivers_section(d)
    types_section(d)
    length_section(d)
    beyond_reactions_section(d)
    combos_section(d)
    myths_section(d)
    explorer_section(d)


def drivers_section(d: dict) -> None:
    st.markdown("#### The drivers")
    st.caption("The tested properties grouped by meaning into 14 drivers plus length. The bar shows how much each driver "
               "moves the scorecard between a typical low post and a typical high post (10th to 90th percentile of train "
               "posts), with everything else held fixed. Wider bar = matters more.")
    df = pd.DataFrame(d["score"]["drivers"])
    ln = d["score"]["length"]
    df = pd.concat([df, pd.DataFrame([{"id": "length", "name": "Length", "direction": "+", "low": ln["low"], "high": ln["high"],
                                       "summary": f"More words help up to about {ln['cap']}.", "advice": "",
                                       "items": 1, "items_nonzero": 1}])])
    df["swing"] = df["high"] / df["low"]
    df = df.sort_values("swing", ascending=False)
    df["kind"] = np.where(df["direction"] == "+", "helps", "hurts")
    chart = alt.Chart(df).mark_bar(height=14).encode(
        x=alt.X("low:Q", title="Effect on reactions vs the average post (×), typical low to typical high",
                axis=LOG_AXIS, scale=alt.Scale(type="log", nice=False, domain=[0.85, 1.25])),
        x2="high:Q",
        y=alt.Y("name:N", sort=list(df["name"]), title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["helps", "hurts"], range=[GREEN, RED]), title=None),
        tooltip=["name", "summary", alt.Tooltip("low:Q", format=".2f"), alt.Tooltip("high:Q", format=".2f"),
                 alt.Tooltip("items:Q", title="properties"), alt.Tooltip("items_nonzero:Q", title="with a weight")])
    rule = alt.Chart(pd.DataFrame({"v": [1.0]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(x="v:Q")
    st.altair_chart((chart + rule).properties(height=alt.Step(26)), use_container_width=True)

    eff = d["effects"].set_index("id")
    for _, r in df[df["id"] != "length"].iterrows():
        v = d["spec"]["drivers"][r["id"]]
        members = [m for m in v["members"] if m[0] in eff.index]
        conf = [m for m in members if eff.loc[m[0], "confirmed"]]
        with st.expander(f"{'▲' if r['direction'] == '+' else '▼'} **{r['name']}** · {r['summary']}"):
            st.markdown(f"**Advice:** {v['advice']}")
            t = pd.DataFrame([{"property": m[1], "type": {"write": "writing choice", "occasion": "only if true",
                                                            "descr": "describes the post"}[m[2]],
                               "share of posts": eff.loc[m[0], "share"], "reactions": eff.loc[m[0], "mult"],
                               "held-out authors": eff.loc[m[0], "test_mult"], "win rate": eff.loc[m[0], "win_rate"],
                               "status": eff.loc[m[0], "status"]} for m in members]).sort_values("reactions", ascending=r["direction"] == "-")
            st.dataframe(t, hide_index=True, use_container_width=True, column_config={
                "share of posts": st.column_config.NumberColumn(format="percent"),
                "reactions": st.column_config.NumberColumn(format="%.2f×"),
                "held-out authors": st.column_config.NumberColumn(format="%.2f×"),
                "win rate": st.column_config.NumberColumn(format="%.2f")})
            st.caption(f"{len(conf)} of {len(members)} confirmed on their own. Win rate: for a post with the property and one "
                       "without by the same author, how often the one with it got more reactions (0.50 = coin flip).")


def types_section(d: dict) -> None:
    st.markdown("#### Kinds of post")
    st.caption("Eight kinds of post, found by clustering Jev's answers to 147 questions on train posts (names by hand). "
               "Reactions are against the same author's other posts; error bars resample authors.")
    t = d["types"].sort_values("lift", ascending=False)
    base = alt.Chart(t).encode(y=alt.Y("type:N", sort=list(t["type"]), title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(height=16).encode(
        x=alt.X("lift:Q", title="Reactions vs the author's usual (×)", axis=LOG_AXIS, scale=alt.Scale(type="log", nice=False, domain=[0.75, 1.6])),
        x2=alt.datum(1.0),
        color=alt.condition(alt.datum.lift > 1, alt.value(GREEN), alt.value(RED)),
        tooltip=["type", "description", alt.Tooltip("share:Q", format=".0%"), alt.Tooltip("lift:Q", format=".2f"),
                 alt.Tooltip("lift_test:Q", format=".2f", title="held-out authors"),
                 alt.Tooltip("comments:Q", format=".2f"), alt.Tooltip("reposts:Q", format=".2f"),
                 alt.Tooltip("hit_rate:Q", format=".0%", title="got 2× usual"), alt.Tooltip("flop_rate:Q", format=".0%", title="got ½ usual")])
    err = base.mark_rule(color=GREY).encode(x="lift_lo:Q", x2="lift_hi:Q")
    st.altair_chart((bars + err).properties(height=alt.Step(28)), use_container_width=True)
    show = t[["type", "description", "share", "median_words", "lift", "lift_test", "comments", "reposts", "hit_rate", "flop_rate"]]
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={
        "share": st.column_config.NumberColumn(format="percent"), "median_words": "median words",
        "lift": st.column_config.NumberColumn("reactions", format="%.2f×"),
        "lift_test": st.column_config.NumberColumn("held-out", format="%.2f×"),
        "comments": st.column_config.NumberColumn(format="%.2f×"), "reposts": st.column_config.NumberColumn(format="%.2f×"),
        "hit_rate": st.column_config.NumberColumn("got 2×+ usual", format="percent"),
        "flop_rate": st.column_config.NumberColumn("got ½ or less", format="percent")})
    st.caption("Only personal moments stand out (1.46×); short updates lag (0.83×). Every other kind of post is about average, "
               "so **what you put inside the post matters more than which kind it is.** Job openings and event promotions "
               "get more reposts; discussion questions get slightly more comments.")

    pick = st.selectbox("What works inside one kind of post", list(t["type"]), index=list(t["type"]).index("Opinion and advice"))
    lv = d["levers"]
    lv = lv[(lv["type"] == pick) & lv["significant"]].copy()
    lv["label"] = lv["id"].map(lambda i: label(d, i))
    lv = lv.sort_values("mult", ascending=False)
    if lv.empty:
        st.write("No property is significant within this kind of post.")
        return
    chart = alt.Chart(lv).mark_point(filled=True, size=70).encode(
        x=alt.X("mult:Q", title=f"Reactions within '{pick}' posts (×)", axis=LOG_AXIS, scale=alt.Scale(type="log", nice=False)),
        y=alt.Y("label:N", sort=list(lv["label"]), title=None, axis=alt.Axis(labelLimit=380)),
        color=alt.condition(alt.datum.mult > 1, alt.value(GREEN), alt.value(RED)),
        tooltip=["label", alt.Tooltip("mult:Q", format=".2f"), alt.Tooltip("lo:Q", format=".2f"), alt.Tooltip("hi:Q", format=".2f"),
                 alt.Tooltip("test_mult:Q", format=".2f", title="held-out authors"),
                 alt.Tooltip("share_in_type:Q", format=".0%", title="share of these posts")])
    err = alt.Chart(lv).mark_rule().encode(x="lo:Q", x2="hi:Q", y=alt.Y("label:N", sort=list(lv["label"])),
                                           color=alt.condition(alt.datum.mult > 1, alt.value(GREEN), alt.value(RED)))
    rule = alt.Chart(pd.DataFrame({"v": [1.0]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(x="v:Q")
    st.altair_chart((err + chart + rule).properties(height=alt.Step(20)), use_container_width=True)
    st.caption("Significant within this kind of post after correcting for multiple tests, and in the same direction on held-out authors.")


def length_section(d: dict) -> None:
    st.markdown("#### Length")
    ln = d["length"]
    ln = ln.melt(id_vars=["outcome", "words"], value_vars=["mult", "content_fixed_mult"], var_name="v", value_name="m")
    ln["series"] = ln["v"].map({"mult": "as is", "content_fixed_mult": "same content"})
    ln["outcome"] = ln["outcome"].map({"lift": "reactions", "comments": "comments"})
    order = list(d["length"]["words"].drop_duplicates())
    chart = alt.Chart(ln).mark_line(point=True).encode(
        x=alt.X("words:N", sort=order, title="Words in the post"),
        y=alt.Y("m:Q", title="vs a 120–160 word post by the same author (×)", axis=LOG_AXIS,
                scale=alt.Scale(type="log", nice=False, domain=[0.58, 1.45])),
        color=alt.Color("outcome:N", title=None, scale=alt.Scale(domain=["reactions", "comments"], range=["#1565c0", "#ef6c00"])),
        strokeDash=alt.StrokeDash("series:N", title=None, legend=alt.Legend(symbolStrokeColor=GREY)),
        tooltip=["outcome", "series", "words", alt.Tooltip("m:Q", format=".2f")])
    st.altair_chart(chart.properties(height=300), use_container_width=True)
    st.caption("Posts under about 120 words get fewer reactions (under 40 words: 0.63×). Past 120 words, length on its own adds "
               "nothing for reactions: long posts do better only because they hold more of the content above (\"same content\" line). "
               "Comments keep rising with length a little longer.")


def beyond_reactions_section(d: dict) -> None:
    st.markdown("#### Comments and reposts follow different rules")
    o = d["outcomes"]
    eff = d["effects"]
    e = eff[eff["confirmed"] | eff["talk_confirmed"].fillna(False)].copy()
    e["group"] = np.select([e["confirmed"] & e["talk_confirmed"].fillna(False), e["confirmed"]],
                           ["both", "reactions only"], "comments per reaction only")
    chart = alt.Chart(e).mark_circle(size=60, opacity=0.8).encode(
        x=alt.X("mult:Q", title="Reactions (×)", axis=LOG_AXIS, scale=alt.Scale(type="log", nice=False)),
        y=alt.Y("talk:Q", title="Comments per reaction (×)", axis=LOG_AXIS, scale=alt.Scale(type="log", nice=False)),
        color=alt.Color("group:N", title="confirmed for"),
        tooltip=["label", alt.Tooltip("mult:Q", format=".2f", title="reactions"), alt.Tooltip("comments:Q", format=".2f"),
                 alt.Tooltip("talk:Q", format=".2f", title="comments per reaction"), alt.Tooltip("reposts:Q", format=".2f")])
    rules = alt.Chart(pd.DataFrame({"v": [1.0]}))
    st.altair_chart((chart + rules.mark_rule(color=GREY).encode(x="v:Q") + rules.mark_rule(color=GREY).encode(y="v:Q"))
                    .properties(height=360), use_container_width=True)
    st.markdown(
        f"* Same test as above, run on comments and reposts. **{o['confirmed_by_outcome']['comments']}** properties are confirmed for "
        f"comments and **{o['confirmed_by_outcome']['reposts']}** for reposts. {o['zero_comment_share']:.0%} of posts have no comments and "
        f"{o['zero_repost_share']:.0%} no reposts.\n"
        "* **Questions and problems start conversations, not applause.** Asking the reader something, pointing out a problem or a risk, "
        "and even generic advice raise comments *per reaction* (up to 1.21×) while lowering reactions slightly (top-left of the chart).\n"
        "* **Thanks and events collect applause.** Gratitude, events and pride raise reactions most but comments per reaction fall (0.79–0.81×).\n"
        "* **Reposts are about usefulness to others:** job openings, requirements and application instructions get 1.2× the reposts with no "
        "change in reactions. Naming several people and crediting others raise all three.")


def combos_section(d: dict) -> None:
    st.markdown("#### Combinations")
    cb = d["combos"]
    c = cb[cb["confirmed"]].copy()
    c["pair"] = c["a"].map(lambda i: label(d, i)) + "  +  " + c["b"].map(lambda i: label(d, i))
    c = c.sort_values("extra_when_both", ascending=False)
    st.dataframe(c[["pair", "share_both", "a_alone", "b_alone", "extra_when_both", "both_total", "test_extra"]], hide_index=True,
                 use_container_width=True, column_config={
                     "share_both": st.column_config.NumberColumn("posts with both", format="percent"),
                     "a_alone": st.column_config.NumberColumn("first alone", format="%.2f×"),
                     "b_alone": st.column_config.NumberColumn("second alone", format="%.2f×"),
                     "extra_when_both": st.column_config.NumberColumn("extra when together", format="%.2f×"),
                     "both_total": st.column_config.NumberColumn("together", format="%.2f×"),
                     "test_extra": st.column_config.NumberColumn("extra, held-out", format="%.2f×")})
    st.caption(f"{len(cb)} pairs of the strongest properties were tested for an extra effect when they appear together; {len(c)} were "
               "confirmed on held-out authors. Two patterns: **specifics stack** (a personal story that also names people or a place does "
               "better than either alone: 2.23× and 1.89×), and **mixing a story with a lecture dampens it** (a personal story plus a "
               "made-up example, or plus arguing for a method, loses part of the story's boost).")


def myths_section(d: dict) -> None:
    st.markdown("#### Popular advice that did not hold up")
    eff = d["effects"].set_index("id")
    rows = [{"advice": name, "reactions": eff.loc[i, "mult"], "low": eff.loc[i, "lo"], "high": eff.loc[i, "hi"],
             "held-out": eff.loc[i, "test_mult"], "status": eff.loc[i, "status"], "share of posts": eff.loc[i, "share"]}
            for i, name in MYTHS.items() if i in eff.index]
    t = pd.DataFrame(rows)
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        "reactions": st.column_config.NumberColumn(format="%.2f×"), "low": st.column_config.NumberColumn("95% low", format="%.2f×"),
        "high": st.column_config.NumberColumn("95% high", format="%.2f×"), "held-out": st.column_config.NumberColumn(format="%.2f×"),
        "share of posts": st.column_config.NumberColumn(format="percent")})
    w, v = d["wearoff"], d["variety"]
    st.caption(
        "Hashtags looked strong on train authors (1.25×) but only 1.09× on held-out authors, so they are not counted as confirmed. "
        f"Two more null results: effects do **not** wear off for authors who use them often (0 of {len(w)} properties show it after "
        "correction), and a post that is **unlike the author's usual posts** does no better or worse "
        f"({v['per_sd_unusual']['mult']:.2f}× per standard deviation, 95% {v['per_sd_unusual']['lo']:.2f}–{v['per_sd_unusual']['hi']:.2f}).")


def explorer_section(d: dict) -> None:
    st.markdown("#### Every property tested")
    eff = d["effects"]
    status = st.multiselect("Status", ["confirmed", "not replicated", "no clear effect"], default=["confirmed"])
    q = st.text_input("Search", placeholder="e.g. question, people, link")
    t = eff[eff["status"].isin(status)]
    if q:
        t = t[t["label"].str.contains(q, case=False) | t["question"].str.contains(q, case=False)]
    t = t.sort_values("mult", ascending=False)
    st.dataframe(t[["label", "driver", "share", "mult", "lo", "hi", "test_mult", "win_rate", "comments", "reposts", "hit", "flop",
                    "same_media_mult", "length_adj_mult", "status", "question"]],
                 hide_index=True, use_container_width=True, height=420, column_config={
                     "label": "property", "share": st.column_config.NumberColumn(format="percent"),
                     "mult": st.column_config.NumberColumn("reactions", format="%.2f×"),
                     "lo": st.column_config.NumberColumn("95% low", format="%.2f×"),
                     "hi": st.column_config.NumberColumn("95% high", format="%.2f×"),
                     "test_mult": st.column_config.NumberColumn("held-out", format="%.2f×"),
                     "win_rate": st.column_config.NumberColumn("win rate", format="%.2f"),
                     "comments": st.column_config.NumberColumn(format="%.2f×"), "reposts": st.column_config.NumberColumn(format="%.2f×"),
                     "hit": st.column_config.NumberColumn("2× hit (pts)", format="%+.1f"),
                     "flop": st.column_config.NumberColumn("flop (pts)", format="%+.1f"),
                     "same_media_mult": st.column_config.NumberColumn("same media", format="%.2f×"),
                     "length_adj_mult": st.column_config.NumberColumn("same length", format="%.2f×")})
    st.caption("Reactions: vs the same author's usual post, all authors, 95% CI. Held-out: the 165 authors never used to choose anything. "
               "2× hit / flop: change in the chance of at least 2× / at most ½ the author's usual reactions, in percentage points. "
               "Same media / same length: the estimate comparing only posts with the same attachment type, or holding length fixed.")


# ------------------------------------------------------------------ Evidence


def evidence_tab() -> None:
    d = load()
    s = d["score"]
    m = s["metrics"]
    st.subheader("How well the scorecard works")
    st.markdown(
        f"The scorecard behind **Evaluate a post** gives each of the {m['items']} properties that passed discovery its own weight, "
        f"fitted together within author with the direction fixed in advance (a helpful property can only add, a harmful one only "
        f"subtract), plus length up to {m['length_cap']} words. {m['items_nonzero']} properties keep a non-zero weight. Everything was "
        f"chosen on {m['train_authors']} train authors; the {m['test_authors']} held-out authors ({m['test_posts']:,} posts) were used once.")
    c = st.columns(4)
    c[0].metric("Pairwise accuracy, held-out", f"{m['test_pairwise']:.1%}",
                help="For two posts by the same held-out author, how often the higher-scoring one got more reactions. 50% = chance.")
    c[1].metric("vs previous dashboard model", f"{m['vs_dashboard']['mean']:+.3f}",
                help=f"Paired over authors, 95% CI {m['vs_dashboard']['ci95']}. A tie: the scorecard's gain is interpretability and advice.")
    c[2].metric("Calibration slope, same author", f"{m['test_calibration_slope']:.2f}",
                help=f"1.00 = a score difference of d between two posts by the same author means exp(d) times the reactions. "
                     f"95% CI {m['test_calibration_slope_ci95']}.")
    c[3].metric("Cost per post", "$0.0011", help="Jev answers about 173 yes/no questions; the text checks are free.")

    st.markdown("#### Does a higher score mean more reactions? (held-out authors)")
    b = pd.DataFrame(s["bands"])
    long = b.melt(id_vars=["name"], value_vars=["predicted", "actual"], var_name="kind", value_name="m")
    order = list(b["name"])
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("name:N", sort=order, title="Score band"), xOffset="kind:N",
        y=alt.Y("m:Q", title="Reactions vs the author's usual (×)"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["predicted", "actual"], range=["#90caf9", "#1565c0"]), title=None),
        tooltip=["name", "kind", alt.Tooltip("m:Q", format=".2f")])
    err = alt.Chart(b).mark_rule(color=GREY, strokeWidth=2).encode(x=alt.X("name:N", sort=order), y="actual_lo:Q", y2="actual_hi:Q")
    st.altair_chart((bars + err).properties(height=300), use_container_width=True)
    st.dataframe(b[["name", "share", "predicted", "actual", "median_ratio", "above_usual", "hit_rate", "flop_rate", "comments", "reposts"]],
                 hide_index=True, use_container_width=True, column_config={
                     "name": "band", "share": st.column_config.NumberColumn("share of posts", format="percent"),
                     "predicted": st.column_config.NumberColumn(format="%.2f×"), "actual": st.column_config.NumberColumn(format="%.2f×"),
                     "median_ratio": st.column_config.NumberColumn("median post", format="%.2f×"),
                     "above_usual": st.column_config.NumberColumn("beat the author's usual", format="percent"),
                     "hit_rate": st.column_config.NumberColumn("got 2×+", format="percent"),
                     "flop_rate": st.column_config.NumberColumn("got ½ or less", format="percent"),
                     "comments": st.column_config.NumberColumn(format="%.2f×"), "reposts": st.column_config.NumberColumn(format="%.2f×")})
    st.caption("Bands are cut at the 10th, 30th, 70th and 90th percentile of train scores. Predicted: the calibrated expectation for a post "
               "by an author you know nothing else about. Actual: what held-out posts in the band got, with a 95% range over authors. "
               "Posts in the top band beat the author's usual 71% of the time and got at least double it 39% of the time; the bottom band 28% and 7%.")

    st.markdown("#### Comparing two drafts by the same author")
    g = pd.DataFrame(s["pair_gap"])
    g["gap"] = g.apply(lambda r: f"{r['lo']:.2f}+" if pd.isna(r["hi"]) else f"{r['lo']:.2f}–{r['hi']:.2f}", axis=1)
    g["ratio"] = g.apply(lambda r: f"≥ {np.exp(r['lo']):.2f}×" if pd.isna(r["hi"]) else f"{np.exp(r['lo']):.2f}–{np.exp(r['hi']):.2f}×", axis=1)
    chart = alt.Chart(g).mark_bar(color="#1565c0").encode(
        x=alt.X("ratio:N", sort=list(g["ratio"]), title="Score gap between the two posts (as a ratio)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", title="Higher score got more reactions", scale=alt.Scale(domain=[0.4, 1]), axis=alt.Axis(format="%")),
        tooltip=["ratio", alt.Tooltip("accuracy:Q", format=".0%"), alt.Tooltip("share:Q", format=".0%", title="share of pairs"), "pairs"])
    half = alt.Chart(pd.DataFrame({"v": [0.5]})).mark_rule(color=RED, strokeDash=[4, 4]).encode(y="v:Q")
    st.altair_chart((chart + half).properties(height=260), use_container_width=True)
    st.caption("All pairs of posts by the same held-out author (whose reactions differ by at least 10%). Small score gaps are close to a coin "
               "flip; when one draft scores at least 1.65× higher it did better 75% of the time, and above 2.2× 85–91% of the time.")

    st.markdown("#### Compared with other models (same held-out authors)")
    rows = [{"model": "Scorecard (this dashboard)", "accuracy": m["test_pairwise"], "vs previous": m["vs_dashboard"]["mean"],
             "ci": m["vs_dashboard"]["ci95"], "readable": "yes: one weight per property, signs fixed"}]
    for k, v in d["compare"].items():
        if k == "new_questions_gain_in_ridge":
            continue
        rows.append({"model": k[0].upper() + k[1:], "accuracy": v["pairwise"],
                     "vs previous": (v["vs_dashboard"] or {}).get("mean", 0.0), "ci": (v["vs_dashboard"] or {}).get("ci95"),
                     "readable": "yes" if "dashboard" in k or "checklist" in k or "joint" in k else "no"})
    for k, v in d["subset"].items():
        if k.startswith("blend: dashboard + ridge"):
            rows.append({"model": "Average of previous model and ridge on all questions", "accuracy": v["pairwise"],
                         "vs previous": v["vs_dashboard"]["mean"], "ci": v["vs_dashboard"]["ci95"], "readable": "no"})
    t = pd.DataFrame(rows)
    t["95% CI vs previous"] = t["ci"].map(lambda c: "—" if not c else f"[{c[0]:+.3f}, {c[1]:+.3f}]")
    st.dataframe(t[["model", "accuracy", "vs previous", "95% CI vs previous", "readable"]].sort_values("accuracy", ascending=False),
                 hide_index=True, use_container_width=True, column_config={
                     "accuracy": st.column_config.NumberColumn(format="%.3f"), "vs previous": st.column_config.NumberColumn(format="%+.3f")})
    st.caption("Every content-only model lands at 0.59–0.62. Text-only yes/no questions seem to top out there; the differences between the "
               "good models are within noise. The scorecard was chosen because it ties the best while every weight is readable and every "
               "suggestion can be traced to a tested property.")

    st.markdown("#### How the research ran")
    led = d["ledger"]
    st.markdown(
        "1. **Existing questions (free).** The 147 Jev questions already answered for every post, plus 15 free text checks.\n"
        "2. **Two rounds of new questions.** DeepSeek read, for 150 *generation* train authors only, the posts that most beat or missed the "
        "model, and proposed new questions; others were hand-written to split the biggest effects. Jev screened them on 250 other train "
        "authors, and the survivors were scored on the remaining train authors and all held-out authors.\n"
        "3. **Confirmation.** Each property: Benjamini-Hochberg q < 0.05 on train authors, then the same direction on held-out authors "
        "(BH-adjusted one-sided p < 0.05). Robustness: same media type only, same length, each follower band.\n"
        "4. **More outcomes and structure (free).** Comments, reposts, 2× hits and flops; kinds of post; length curve; combinations; wear-off; variety.\n"
        "5. **Scorecard.** The 150 generation authors were then scored on the 26 new questions in the scorecard, so it trains on all 667 train authors.")
    st.dataframe(pd.DataFrame(led["runs"])[["name", "posts", "questions", "cost_usd"]], hide_index=True, use_container_width=True,
                 column_config={"cost_usd": st.column_config.NumberColumn("Jev cost", format="$%.2f")})
    st.caption(f"Total Jev spend for the research: ${led['spent_usd']:.2f} of the ${led['budget_usd']:.2f} budget. Every run is logged "
               "in research/results/jev_ledger.json. Full write-up: research/REPORT.md.")
