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
from linkedin_analysis.paths import PROCESSED  # noqa: E402
from linkedin_analysis.predict import (  # noqa: E402
    build_row,
    load_bundle,
    multiplier,
    linear_features,
    predict_linear,
)

MAX_COST_PER_SESSION = 0.25  # USD, Jev spend cap for one browser session

st.set_page_config(page_title="LinkedIn Post Lab", page_icon="📝", layout="wide")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _mtime(name: str) -> float:
    path = PROCESSED / "models" / name
    return path.stat().st_mtime if path.exists() else 0.0


VARIANTS = {"full": "model_bundle.json", "compact": "model_bundle_compact.json"}  # compact: scripts/compact_model.py


@st.cache_resource
def _bundle(name: str, mtime: float) -> dict:
    return load_bundle(PROCESSED / "models" / name)


def bundle(variant: str = "full") -> dict:
    """A trained model; reloaded automatically when its file is rewritten."""
    return _bundle(VARIANTS[variant], _mtime(VARIANTS[variant]))


def has_variant(variant: str) -> bool:
    return (PROCESSED / "models" / VARIANTS[variant]).exists()


@st.cache_resource
def _theme_names(mtime: tuple[float, float]) -> dict[str, dict]:
    names = read_json(PROCESSED / "models" / "theme_names.json") or {}
    themes = bundle().get("themes", {})
    return {t: v for t, v in names.items() if t in themes and v.get("members") == themes[t]["members"]}


def theme_names() -> dict[str, dict]:
    """DeepSeek-written names (scripts/name_themes.py), used only if the theme's questions still match."""
    return _theme_names((_mtime("model_bundle.json"), _mtime("theme_names.json")))


def theme_title(t: str, b: dict | None = None) -> str:
    """Theme name: the bundle's own name (compact model), else the DeepSeek name, else the central question."""
    if b and b["themes"][t].get("name"):
        return b["themes"][t]["name"]
    named = theme_names().get(t)
    return named["name"] if named else bundle()["themes"][t]["label"]


def theme_description(t: str, b: dict | None = None) -> str:
    if b and b["themes"][t].get("name"):
        return b["themes"][t].get("description", "")
    named = theme_names().get(t)
    return named["description"] if named else ""


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


def fmt_x(lift: float) -> str:
    m = multiplier(lift)
    return f"{m:.2f}×"


# Tab 1: evaluate a post


def variant_label(variant: str) -> str:
    b = bundle(variant)
    acc = b["metrics"]["linear"]["test"]["pairwise_accuracy"]
    kind = "Full" if variant == "full" else "Compact"
    return f"{kind}: {len(linear_features(b))} questions · test {acc:.1%}"


def evaluate_tab() -> None:
    st.subheader("Evaluate a draft post")
    variants = ["full"] + (["compact"] if has_variant("compact") else [])
    variant = st.radio("Model", variants, format_func=variant_label, horizontal=True,
                       help="Full: each theme is the average of all its questions. Compact: each theme is measured by "
                            "its 2 most representative questions only. Cheaper and easier to read, about 1 point "
                            "less accurate.")
    b = bundle(variant)
    questions = {f["id"]: f["question"] for f in b["features"]}
    used = {t: v for t, v in b.get("themes", {}).items() if t in b["linear"]["coef"]}  # themes the model uses
    questions.update({t: theme_title(t, b) for t in used})
    ask = linear_features(b)  # only the questions those themes need
    m = b["metrics"]

    st.caption(
        f"The linear model uses {len(b['linear']['coef'])} content themes. Jev answers the {len(ask)} yes/no "
        f"questions behind them (about ${len(ask) * judge.USD_PER_QUESTION:.4f} per post), and the model turns "
        "those answers into an expected result **relative to the author's usual post**, for any account size."
    )
    left, right = st.columns([4, 1])
    text = left.text_area("Post text", height=260, placeholder="Paste a LinkedIn post draft…")
    right.caption("Text only: attached media is not used anywhere.")
    go = right.button("Evaluate with Jev", type="primary", disabled=not text.strip(), use_container_width=True)

    if go:
        client = jev()
        post = {"urn": "draft", "media": "text", "body": text}  # Jev sees only the text
        with st.spinner(f"Asking Jev {len(ask)} questions…"):
            try:
                rows, skipped = judge.judge(client.decide, [post], ask, workers=8)
            except Exception as e:  # network or API errors
                st.error(f"Jev call failed: {e}")
                return
        if skipped:
            st.error(f"Session spending cap (${MAX_COST_PER_SESSION:.2f}) reached. Restart the app to reset it.")
            return
        st.session_state["result"] = {"rows": rows, "text": text, "spent": client.usage.cost_usd}

    result = st.session_state.get("result")
    if not result:
        return
    p_yes = {r["feature_id"]: r["p_yes"] for r in result["rows"]}
    missing = [f for f in ask if f["id"] not in p_yes]
    if missing:  # evaluated with the compact model, then switched to full
        st.info(f"This post was evaluated with fewer questions. Click **Evaluate with Jev** to ask the "
                f"{len(missing)} missing ones for this model.")
        return
    p_yes = {f["id"]: p_yes[f["id"]] for f in ask}  # the full model's answers also cover the compact one
    row = build_row(b, p_yes, "text", result["text"])
    lin_pred, contrib = predict_linear(b, row)

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Expected reactions vs. author's usual", fmt_x(lin_pred),
              help="Linear model on content themes. 1.00× = a typical post for this author.")
    c2.metric("Model test accuracy", f"{m['linear']['test']['pairwise_accuracy']:.0%}",
              help="For two posts by the same held-out author, how often the model ranks the better one higher. "
                   "50% = chance.")
    st.caption(f"Jev answered {len(p_yes)} questions. Spend this session: ${result['spent']:.4f}.")

    st.markdown("**What moves the prediction**")
    top = [c for c in contrib if abs(c["contribution"]) > 1e-4]
    if top:
        df = pd.DataFrame({
            "theme": [describe(c["column"], questions) for c in top],
            "effect": [c["contribution"] for c in top],
            "theme score": [c["value"] for c in top],
            "average post": [b["column_means"][c["column"]] for c in top],
            "weight": [c["coef"] for c in top],
        })
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("effect:Q", title="Push on log-lift vs. average post"),
            y=alt.Y("theme:N", sort=None, title=None, axis=alt.Axis(labelLimit=520)),
            color=alt.condition(alt.datum.effect > 0, alt.value("#2e7d32"), alt.value("#c62828")),
            tooltip=["theme", alt.Tooltip("effect:Q", format="+.3f"),
                     alt.Tooltip("theme score:Q", format=".2f", title="this post's score"),
                     alt.Tooltip("average post:Q", format=".2f", title="average post's score"),
                     alt.Tooltip("weight:Q", format="+.3f")],
        ).properties(height=alt.Step(30))
        st.altair_chart(chart, use_container_width=True)
        st.caption("Each bar compares this post with an average post: effect = weight × (this post's score − "
                   "average post's score). A red bar on a helpful theme means the post is just below average on it, "
                   "a missed boost, not a penalty. Theme scores themselves are always between 0 and 1.")
    else:
        st.write("No theme moves this prediction away from average.")

    with st.expander(f"How this post scored on the {len(used)} themes"):
        for t, v in used.items():
            st.markdown(f"**{theme_title(t, b)}** — this post: **{row[t]:.0%}** of the checklist "
                        f"(typical post {v['q25']:.0%}–{v['q75']:.0%})")
            st.caption(theme_description(t, b))
            st.markdown("\n".join(f"- {'✅' if p_yes.get(f, 0) >= 0.5 else '▫️'} {q} ({p_yes.get(f, float('nan')):.2f})"
                                    for f, q in zip(v["members"], v["member_questions"])))

    with st.expander(f"All {len(p_yes)} Jev answers"):
        table = pd.DataFrame([{"question": questions[k], "P(yes)": v} for k, v in p_yes.items()]
                             ).sort_values("P(yes)", ascending=False)
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"P(yes)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})


# Tab 2: pipeline


def pipeline_tab() -> None:
    labels = read_json(PROCESSED / "labels_summary.json") or {}
    splits = read_json(PROCESSED / "splits_summary.json") or {}
    disc = read_json(PROCESSED / "features" / "run_meta.json") or {}
    full = (PROCESSED / "judgments_full_train" / "run_meta.json").exists()
    jdir = "judgments_full" if full else "judgments_text"
    jt = read_json(PROCESSED / f"{jdir}_train" / "run_meta.json") or {}
    jte = read_json(PROCESSED / f"{jdir}_test" / "run_meta.json") or {}
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
        ("5 · Judge", f"Kept {jt.get('features', '–')} broadly applicable features (true in 5–95% of posts and "
                      "for at least 20% of authors; attachment questions removed). Jev 1.13 read **the text only** "
                      f"and scored P(yes) for each on {jt.get('posts', 0) + jte.get('posts', 0):,} "
                      f"posts ({jt.get('authors', '–')} train + {jte.get('authors', '–')} test authors) for "
                      f"${jt.get('usage', {}).get('cost_usd', 0) + jte.get('usage', {}).get('cost_usd', 0):.2f}."),
        ("6 · Model", f"Questions that tend to be true together were grouped into **{data.get('themes', '–')} "
                      "themes** (checklists); themes with fewer than 3 questions were set aside as too niche. "
                      "A sparse linear model picks the themes that predict lift, **content only**: no media, length, "
                      "or posting time. Tuned with author-grouped CV, then tested once."),
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

    baselines_section(data)
    learning_curve_section()
    question_budget_section()
    themes_section()


BASELINES = {  # key in baselines.json: (label, input, trained on)
    "themes_linear (current)": ("Themes, sparse linear (current)", "Jev theme scores", "subset"),
    "tfidf_ridge_all": ("TF-IDF + ridge, all data", "Raw words and word pairs", "all"),
    "jev_ridge": ("Ridge on all Jev questions", "All single Jev P(yes)", "subset"),
    "jev_gbm": ("Gradient boosting on Jev questions", "All single Jev P(yes), non-linear", "subset"),
    "tfidf_ridge_subset": ("TF-IDF + ridge", "Raw words and word pairs", "subset"),
    "format_reference": ("Format reference (not content)", "Media, length, posting time", "subset"),
}


def baselines_section(data: dict) -> None:
    res = read_json(PROCESSED / "models" / "baselines.json")
    if not res:
        return
    st.markdown("#### Compared with standard ML baselines")
    subset = f"{data.get('train_authors', '–')} authors ({data.get('train_posts', 0):,} posts)"
    rows = []
    for key, (label, inputs, trained) in BASELINES.items():
        r = res.get(key)
        if not r:
            continue
        ci = r["test"].get("pairwise_accuracy_ci95") or [None, None]
        rows.append({
            "model": label, "input": inputs,
            "train data": (f"{r.get('train_authors', '–')} authors ({r.get('train_posts', 0):,} posts)"
                           if trained == "all" else subset),
            "CV": r["cv_train"]["pairwise_accuracy"], "test": r["test"]["pairwise_accuracy"],
            "low": ci[0], "high": ci[1],
        })
    df = pd.DataFrame(rows).sort_values("test", ascending=False)
    df["test 95% CI"] = df.apply(lambda r: f"[{r['low']:.3f}, {r['high']:.3f}]" if r["low"] is not None else "—", axis=1)
    st.dataframe(df[["model", "input", "train data", "CV", "test", "test 95% CI"]], hide_index=True,
                 use_container_width=True,
                 column_config={"CV": st.column_config.NumberColumn(format="%.3f"),
                                "test": st.column_config.NumberColumn(format="%.3f")})

    order = list(df["model"])
    base = alt.Chart(df).encode(y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)))
    whiskers = base.mark_rule(strokeWidth=2, color="#888").encode(
        x=alt.X("low:Q", title="Test pairwise accuracy (dot) with 95% range", scale=alt.Scale(zero=False)),
        x2="high:Q")
    dots = base.mark_circle(size=120).encode(
        x="test:Q",
        color=alt.condition(alt.datum.model == "Themes, sparse linear (current)", alt.value("#2e7d32"), alt.value("#555")),
        tooltip=["model", alt.Tooltip("test:Q", format=".3f"), alt.Tooltip("CV:Q", format=".3f"), "train data"])
    chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="#c62828").encode(x="x:Q")
    st.altair_chart((whiskers + dots + chance).properties(height=alt.Step(34)), use_container_width=True)
    st.caption(
        f"All models predict the same target (lift vs. the author's usual), are tuned with the same author-grouped "
        f"folds, and are tested on the same {data.get('test_authors', '–')} held-out authors "
        f"({data.get('test_posts', 0):,} posts). The ranges overlap, so the differences are tendencies, not proven gaps. "
        "Red dashed line = chance (0.5)."
    )


LC_NAMES = {"themes_linear": "Themes, sparse linear (current)", "jev_ridge": "Ridge on all Jev questions",
            "jev_gbm": "Gradient boosting on Jev questions", "tfidf_ridge": "TF-IDF + ridge"}


def learning_curve_section() -> None:
    lc = read_json(PROCESSED / "models" / "learning_curve.json")
    if not lc:
        return
    st.markdown("#### Does more training data help?")
    df = pd.DataFrame(lc["summary"])
    df["model"] = df["model"].map(LC_NAMES).fillna(df["model"])
    df["low"], df["high"] = df["mean"] - df["sd"], df["mean"] + df["sd"]
    base = alt.Chart(df).encode(
        x=alt.X("train_authors:Q", title="Training authors (log scale)", scale=alt.Scale(type="log"),
                axis=alt.Axis(values=sorted(df["train_authors"].unique().tolist()))),
        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="bottom", columns=2, labelLimit=320)))
    band = base.mark_area(opacity=0.12).encode(y=alt.Y("low:Q", scale=alt.Scale(zero=False),
                                                       title="Test pairwise accuracy"), y2="high:Q",
                                               color=alt.Color("model:N", legend=None))
    line = base.mark_line(point=True).encode(
        y="mean:Q",
        tooltip=["model", "train_authors", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("sd:Q", format=".3f"),
                 "runs"])
    st.altair_chart((band + line).properties(height=320), use_container_width=True)
    st.caption(
        f"Each point trains the model from scratch on a random sample of training authors (themes re-clustered, "
        f"settings re-tuned) and scores it on the same {lc['test_authors']} test authors ({lc['test_posts']:,} posts). "
        "Smaller sizes are averaged over 5 random samples; the band is ±1 standard deviation across them. "
        "The rightmost point uses every training author once.")


def question_budget_section() -> None:
    qb = read_json(PROCESSED / "models" / "question_budget.json")
    if not qb:
        return
    st.markdown("#### How many questions does Jev need to answer?")
    rows = pd.DataFrame(qb["rows"])
    curves = rows[rows["method"] != "all_ridge"].copy()
    curves["series"] = curves["method"].map({"themes": "7 themes, top questions of each",
                                             "single": "Single questions, best added one at a time"})
    curves["point"] = curves.apply(
        lambda r: f"{r['per_theme']} per theme" if r["method"] == "themes" else f"{int(r['questions'])} questions", axis=1)
    x = alt.X("questions:Q", title="Jev questions per post (log scale)", scale=alt.Scale(type="log"),
              axis=alt.Axis(values=[1, 2, 5, 7, 10, 14, 21, 34, 86, 147]))
    top = max(rows["test"].max(), 0.0)
    y = alt.Y("test:Q", title="Test pairwise accuracy",
              scale=alt.Scale(domain=[float(rows["test"].min()) - 0.005, float(top) + 0.01]))
    series = ["7 themes, top questions of each", "Single questions, best added one at a time"]
    lines = alt.Chart(curves).mark_line(point=True).encode(
        x=x, y=y, color=alt.Color("series:N", title=None, scale=alt.Scale(domain=series, range=["#2e7d32", "#ef6c00"]),
                                  legend=alt.Legend(orient="bottom", labelLimit=320)),
        tooltip=["series", "point", "questions", alt.Tooltip("test:Q", format=".3f"), alt.Tooltip("cv:Q", format=".3f")])
    layers = [lines]
    marks = curves[(curves["method"] == "themes") & curves["per_theme"].isin([qb["per_theme_compact"], "all"])].copy()
    marks["label"] = marks["per_theme"].map(lambda m: "Full model" if m == "all" else "Compact model")
    layers.append(alt.Chart(marks).mark_text(dy=-12, fontWeight="bold", color="#2e7d32").encode(x=x, y=y, text="label:N"))
    ridge = rows[rows["method"] == "all_ridge"]
    if len(ridge):
        r = ridge.iloc[0]
        layers.append(alt.Chart(pd.DataFrame({"y": [r["test"]], "label": [f"All {int(r['questions'])} questions (ridge)"]}))
                      .mark_rule(strokeDash=[4, 4], color="#888").encode(y="y:Q", tooltip=["label", alt.Tooltip("y:Q", format=".3f")]))
        layers.append(alt.Chart(pd.DataFrame({"y": [r["test"]], "x": [1], "label": [f"all {int(r['questions'])} questions"]}))
                      .mark_text(align="left", dy=-7, color="#888").encode(x=alt.X("x:Q", scale=alt.Scale(type="log")), y="y:Q", text="label:N"))
    st.altair_chart(alt.layer(*layers).properties(height=320), use_container_width=True)
    st.caption(
        f"Every point is trained on all training authors and scored on the same {qb['test_authors']} test authors. "
        "Compact: each of the 7 themes is measured by its 2 questions that best track the full theme score, chosen on "
        "training posts. About 10–15 questions keep most of the accuracy (~0.60 vs 0.613); averaging two questions per "
        "theme beats picking 14 single questions. Dashed line: ridge on all questions. The single-question picks are "
        "not stable: rerun on other training folds, only about 5 of 10 are always chosen.")

    if has_variant("compact"):
        cb = bundle("compact")
        with st.expander(f"The compact model's {len(cb['themes'])} themes and {len(linear_features(cb))} questions"):
            st.caption("With only 2 questions, a theme can mean something narrower than its full version, so each "
                       "compact theme is renamed from its own questions (full theme in brackets).")
            for t, v in sorted(cb["themes"].items(), key=lambda kv: -kv[1]["coef"]):
                arrow = "helps" if v["coef"] > 0 else "hurts"
                st.markdown(f"**{v['name']}** ({v.get('full_name', '')}) — typical effect "
                            f"**{v['realistic_x']:.2f}×** ({arrow})\n"
                            + "\n".join(f"- {q}" for q in v["member_questions"]))


def themes_section() -> None:
    if not (PROCESSED / "models" / "model_bundle.json").exists():
        return
    b = bundle()
    themes = b.get("themes", {})
    used = sorted((t for t in themes if themes[t].get("coef")), key=lambda t: -themes[t]["realistic_x"])
    if not used:
        return

    st.markdown(f"#### What the model learned: {len(used)} content themes")
    st.markdown(
        "A **theme** is a checklist of yes/no questions that tend to be true on the same posts. "
        "Every post gets a score on every theme: the share of that checklist it ticks. "
        "A post that doesn't touch a theme scores near 0 there, so that theme barely changes its prediction."
    )

    df = pd.DataFrame([{"theme": theme_title(t), "effect": themes[t]["realistic_x"],
                        "direction": "helps" if themes[t]["coef"] > 0 else "hurts",
                        "folds": themes[t]["sign_stability"], "questions": len(themes[t]["members"])} for t in used])
    lo, hi = min(df["effect"].min(), 1) - 0.03, max(df["effect"].max(), 1) + 0.03
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("effect:Q", title="Typical effect on reactions vs. the author's usual (1.0 = none)",
                scale=alt.Scale(domain=[lo, hi], zero=False)),
        x2=alt.datum(1.0),
        y=alt.Y("theme:N", sort=None, title=None, axis=alt.Axis(labelLimit=300, labelFontSize=13)),
        color=alt.condition(alt.datum.effect > 1, alt.value("#2e7d32"), alt.value("#c62828")),
        tooltip=["theme", alt.Tooltip("effect:Q", format=".2f", title="typical effect"),
                 alt.Tooltip("folds:Q", format=".0%", title="same direction in CV folds"), "questions"],
    ).properties(height=alt.Step(34))
    st.altair_chart(chart, use_container_width=True)
    st.caption("Typical effect: moving a post from a typical low to a typical high score on the theme "
               "(25th to 75th percentile of training posts). Associations, not causes.")

    for t in used:
        v = themes[t]
        with st.container(border=True):
            left, mid, right = st.columns([3, 1, 1])
            left.markdown(f"**{theme_title(t)}**  \n{theme_description(t)}")
            mid.metric("Typical effect", f"{v['realistic_x']:.2f}×")
            if v["coef"] > 0:
                mid.badge("helps", icon=":material/arrow_upward:", color="green")
            else:
                mid.badge("hurts", icon=":material/arrow_downward:", color="red")
            right.metric("Consistency", f"{round(v['sign_stability'] * 5)}/5 folds",
                         help="In how many of the 5 cross-validation folds the theme pushed the same way.")
            with st.expander(f"{len(v['members'])} questions in this checklist"):
                qs = pd.DataFrame({"question": v["member_questions"],
                                   "yes on % of training posts": v["member_yes_rate"]})
                st.dataframe(qs, hide_index=True, use_container_width=True,
                             column_config={"yes on % of training posts": st.column_config.ProgressColumn(
                                 min_value=0, max_value=1, format="percent")})

    unused = [t for t in themes if t not in used]
    with st.expander(f"{len(unused)} other themes the model does not use"):
        st.caption("Either they didn't predict performance in cross-validation, or they have fewer than 3 "
                   "questions (too niche to apply to most posts).")
        st.dataframe(pd.DataFrame([{"theme": theme_title(t), "description": theme_description(t),
                                    "questions": len(themes[t]["members"])} for t in unused]),
                     hide_index=True, use_container_width=True)


tab1, tab2 = st.tabs(["Evaluate a post", "Pipeline"])
with tab1:
    if (PROCESSED / "models" / "model_bundle.json").exists():
        evaluate_tab()
    else:
        st.info("Train the models first: scripts/train_models.py")
with tab2:
    pipeline_tab()
