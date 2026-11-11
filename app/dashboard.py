"""LinkedIn post dashboard: evaluate a draft with Jev, and a short pipeline overview.

Run:
    uv run streamlit run app/dashboard.py

Needs OPENROUTER_API_KEY in .env and the outputs of the pipeline scripts
(data/processed/models/model_bundle.json in particular).
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # see README "Imports on this Mac"

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from linkedin_analysis import judge  # noqa: E402
from linkedin_analysis.llm import DecisionsClient, LLMSettings  # noqa: E402
from linkedin_analysis.model import MEDIA  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402
from linkedin_analysis.predict import (  # noqa: E402
    build_row,
    load_bundle,
    multiplier,
    predict_linear,
    predict_rulefit,
)

MAX_COST_PER_SESSION = 0.25  # USD, Jev spend cap for one browser session

st.set_page_config(page_title="LinkedIn Post Lab", page_icon="📝", layout="wide")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource
def bundle() -> dict:
    return load_bundle()


@st.cache_resource
def jev() -> DecisionsClient:
    return DecisionsClient(LLMSettings.from_env("openrouter"), max_cost_usd=MAX_COST_PER_SESSION, max_connections=8)


def describe(column: str, questions: dict[str, str]) -> str:
    """Readable name for a model column."""
    if column in questions:
        return questions[column]
    names = {"ctl.log_words": "Post length (log words)", "ctl.weekend": "Posted on a weekend",
             "ctl.hour_night": "Posted at night (PT)", "ctl.hour_afternoon": "Posted in the afternoon (PT)",
             "ctl.hour_evening": "Posted in the evening (PT)"}
    if column.startswith("ctl.media_"):
        return f"Media: {column.removeprefix('ctl.media_')}"
    return names.get(column, column)


def describe_condition(col: str, op: str, t: float, questions: dict[str, str]) -> str:
    if col in questions:
        # Thresholds near 0 or 1 mean "not a clear no/yes" rather than a plain yes/no.
        if op == ">":
            answer = "yes" if t >= 0.5 else "not a clear no"
        else:
            answer = "no" if t < 0.5 else "not a clear yes"
        return f"{questions[col]} → **{answer}** (P(yes) {op} {t:.2f})"
    if col == "ctl.log_words":
        return f"{'more' if op == '>' else 'at most'} than ~{math.expm1(t):.0f} words"
    is_on = op == ">"
    return f"{describe(col, questions)}: {'yes' if is_on else 'no'}"


def fmt_x(lift: float) -> str:
    m = multiplier(lift)
    return f"{m:.2f}×"


# Tab 1: evaluate a post


def evaluate_tab() -> None:
    b = bundle()
    questions = {f["id"]: f["question"] for f in b["features"]}
    m = b["metrics"]

    st.subheader("Evaluate a draft post")
    st.caption(
        f"Jev scores the draft on {len(b['features'])} content properties (about "
        f"${len(b['features']) * judge.USD_PER_QUESTION:.3f} per post). Two models trained on real posts turn "
        "those scores into an expected result **relative to the author's usual post**, so it applies to any account size."
    )
    left, right = st.columns([3, 1])
    text = left.text_area("Post text", height=260, placeholder="Paste a LinkedIn post draft…")
    media = right.selectbox("Attached media", ["text", *MEDIA],
                            help="Passed to Jev only, the same way training posts were scored. "
                                 "The models use content features only, not media.")
    go = right.button("Evaluate with Jev", type="primary", disabled=not text.strip(), use_container_width=True)

    if go:
        client = jev()
        post = {"urn": "draft", "media": media, "body": text}
        with st.spinner("Asking Jev…"):
            try:
                rows, skipped = judge.judge(client.decide, [post], b["features"], workers=8)
            except Exception as e:  # network or API errors
                st.error(f"Jev call failed: {e}")
                return
        if skipped:
            st.error(f"Session spending cap (${MAX_COST_PER_SESSION:.2f}) reached. Restart the app to reset it.")
            return
        st.session_state["result"] = {"rows": rows, "media": media, "text": text,
                                      "spent": client.usage.cost_usd}

    result = st.session_state.get("result")
    if not result:
        return
    p_yes = {r["feature_id"]: r["p_yes"] for r in result["rows"]}
    row = build_row(b, p_yes, result["media"], result["text"])
    lin_pred, contrib = predict_linear(b, row)
    rf_pred, fired = predict_rulefit(b, row)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Linear model", fmt_x(lin_pred), help="Expected reactions vs. the author's usual post")
    c2.metric("RuleFit", fmt_x(rf_pred), help="Expected reactions vs. the author's usual post")
    c3.metric("Test pairwise accuracy", f"{m['linear']['test']['pairwise_accuracy']:.0%} / "
              f"{m['rulefit']['test']['pairwise_accuracy']:.0%}",
              help="Linear / RuleFit. For two posts by the same held-out author, how often the model "
                   "ranks the better one higher. 50% = chance.")
    st.caption(f"1.00× = a typical post for this author. Jev spend this session: ${result['spent']:.4f}.")

    # Full width: the property labels are long.
    st.markdown("**What moves the linear model's prediction**")
    top = [c for c in contrib if abs(c["contribution"]) > 1e-4][:12]
    if top:
        df = pd.DataFrame({
            "property": [describe(c["column"], questions) for c in top],
            "effect": [c["contribution"] for c in top],
            "P(yes)": [c["value"] if c["column"] in questions else None for c in top],
        })
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("effect:Q", title="Push on log-lift vs. average post"),
            y=alt.Y("property:N", sort=None, title=None, axis=alt.Axis(labelLimit=520)),
            color=alt.condition(alt.datum.effect > 0, alt.value("#2e7d32"), alt.value("#c62828")),
            tooltip=["property", alt.Tooltip("effect:Q", format="+.3f"), alt.Tooltip("P(yes):Q", format=".2f")],
        ).properties(height=alt.Step(30))
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No property moves this prediction away from average.")

    st.markdown("**RuleFit rules that apply to this post** (strongest first)")
    if fired:
        for r in fired[:8]:
            conds = "<br>".join("• " + describe_condition(c, op, t, questions) for c, op, t in r["conditions"])
            sign = "🟢" if r["coef"] > 0 else "🔴"
            st.markdown(f"{sign} **{r['coef']:+.3f}**<br>{conds}", unsafe_allow_html=True)
    else:
        st.write("No rules fire for this post.")

    with st.expander(f"All {len(p_yes)} Jev answers"):
        table = pd.DataFrame([{"property": questions[k], "P(yes)": v, "dimension": k.split(".")[0]}
                              for k, v in p_yes.items()]).sort_values("P(yes)", ascending=False)
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"P(yes)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})


# Tab 2: pipeline


def pipeline_tab() -> None:
    labels = read_json(PROCESSED / "labels_summary.json") or {}
    splits = read_json(PROCESSED / "splits_summary.json") or {}
    disc = read_json(PROCESSED / "features" / "run_meta.json") or {}
    jt = read_json(PROCESSED / "judgments_train" / "run_meta.json") or {}
    jte = read_json(PROCESSED / "judgments_test" / "run_meta.json") or {}
    metrics = read_json(PROCESSED / "models" / "metrics.json") or {}
    data = metrics.get("data", {})

    st.subheader("How it works")
    steps = [
        ("1 · Data", f"{labels.get('posts_in', 0):,} public LinkedIn posts from {labels.get('authors_in', 0):,} "
                     "authors (1k–20k followers), Jul–Oct 2026."),
        ("2 · Label", f"Each post is compared only with the **same author's** other posts: "
                      f"lift = log(reactions ÷ author's usual). Removes audience size. "
                      f"{labels.get('posts_labeled', 0):,} posts from {labels.get('authors_labeled', 0):,} authors "
                      "with 5+ posts."),
        ("3 · Split", "80/20 **by author**, stratified by band and size; 5 CV folds inside train. "
                      f"Test: {splits.get('by_split', {}).get('test', {}).get('authors', '–')} authors, never seen."),
        ("4 · Discover", f"DeepSeek V4 Pro read {disc.get('pairs', '–')} unlabeled same-author pairs "
                         "(best vs worst, random order) and proposed neutral yes/no content properties; "
                         f"{disc.get('raw_proposals', '–')} proposals merged into **{disc.get('features', '–')} "
                         "features**. It never saw performance."),
        ("5 · Judge", f"Jev 1.13 scored P(yes) for every feature on {jt.get('posts', 0) + jte.get('posts', 0):,} "
                      f"posts ({jt.get('authors', '–')} train + {jte.get('authors', '–')} test authors) for "
                      f"${jt.get('usage', {}).get('cost_usd', 0) + jte.get('usage', {}).get('cost_usd', 0):.2f}."),
        ("6 · Model", f"A sparse linear model (≤10 features) and RuleFit predict lift from "
                      f"{data.get('features_used', '–')} **content features only**: no media, length, or posting-time "
                      "inputs, and attachment questions excluded. Tuned with author-grouped CV, then tested once."),
    ]
    for title, body in steps:
        st.markdown(f"**{title}** — {body}")

    if not metrics:
        return
    st.markdown(f"#### Results on {data.get('test_authors', '–')} held-out authors ({data.get('test_posts', 0):,} posts)")
    names = {"linear": "Sparse linear (relaxed lasso)", "rulefit": "RuleFit",
             "controls": "Reference: format only (media, length, time), not used in any model"}
    res = pd.DataFrame([{
        "model": names[k],
        "pairwise accuracy (test)": metrics[k]["test"]["pairwise_accuracy"],
        "pairwise accuracy (CV)": metrics[k]["cv_train"]["pairwise_accuracy"],
        "per-author Spearman (test)": metrics[k]["test"]["author_spearman"],
        "vs format reference (95% CI)": (
            "—" if k == "controls" else
            "{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]".format(mean=metrics[k]["test"]["gain_over_controls"]["mean"],
                                                        lo=metrics[k]["test"]["gain_over_controls"]["ci95"][0],
                                                        hi=metrics[k]["test"]["gain_over_controls"]["ci95"][1])),
    } for k in names])
    st.dataframe(res, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in res.columns[1:4]})
    st.caption("Pairwise accuracy: for two posts by the same author, how often the model ranks the one that did "
               "better higher. 0.5 = chance. Both content-only models beat chance; content alone ranks posts about as "
               "well as format alone (the reference row), and the difference is within noise at this sample size.")

    coefs_path = PROCESSED / "models" / "linear_coefs.csv"
    if coefs_path.exists():
        coefs = pd.read_csv(coefs_path)
        coefs = coefs[(coefs["coef"] != 0) & ~coefs["column"].str.startswith("ctl.")].head(12)
        if len(coefs):
            st.markdown("#### Content properties the linear model keeps")
            coefs["label"] = coefs["question"]
            lo, hi = min(coefs["effect_x"].min(), 1) - 0.05, max(coefs["effect_x"].max(), 1) + 0.05
            chart = alt.Chart(coefs).mark_bar().encode(
                x=alt.X("effect_x:Q", title="Reactions multiplier when present vs. absent (1.0 = no effect)",
                        scale=alt.Scale(domain=[lo, hi], zero=False)),
                x2=alt.datum(1.0),
                y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
                color=alt.condition(alt.datum.coef > 0, alt.value("#2e7d32"), alt.value("#c62828")),
                tooltip=["label", alt.Tooltip("effect_x:Q", format=".2f"),
                         alt.Tooltip("sign_stability:Q", format=".0%", title="same sign in CV folds")],
            ).properties(height=alt.Step(30))
            st.altair_chart(chart, use_container_width=True)
            st.caption("Associations, not causes. Sign stability = share of CV folds with the same direction.")


tab1, tab2 = st.tabs(["Evaluate a post", "Pipeline"])
with tab1:
    if (PROCESSED / "models" / "model_bundle.json").exists():
        evaluate_tab()
    else:
        st.info("Train the models first: scripts/train_models.py")
with tab2:
    pipeline_tab()
