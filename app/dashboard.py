"""LinkedIn post dashboard: evaluate a draft with the research scorecard, what works, the evidence, and the pipeline.

Run:
    uv run streamlit run app/dashboard.py

Needs OPENROUTER_API_KEY in .env and the outputs of the pipeline scripts
(data/processed/research/scorecard.json from research/scorecard.py, and
data/processed/models/model_bundle.json for the Pipeline tab).
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # see README "Imports on this Mac"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from insights import evidence_tab, what_works_tab  # noqa: E402
from linkedin_analysis import judge, scorecard  # noqa: E402
from linkedin_analysis.llm import DecisionsClient, LLMSettings  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402
from linkedin_analysis.predict import (  # noqa: E402
    build_row,
    load_bundle,
    linear_features,
    multiplier,
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


def fmt_x(lift: float) -> str:
    m = multiplier(lift)
    return f"{m:.2f}×"


# Tab 1: evaluate a post (research scorecard)


@st.cache_resource
def _card(mtime: float) -> dict:
    return scorecard.load()


def card() -> dict:
    """The research scorecard (research/scorecard.py); reloaded when the file is rewritten."""
    return _card(scorecard.BUNDLE.stat().st_mtime)


def ask_jev(texts: list[str], features: list[dict]) -> list[dict[str, float]] | None:
    client = jev()
    posts = [{"urn": f"draft{i}", "media": "text", "body": t} for i, t in enumerate(texts)]  # Jev sees only the text
    with st.spinner(f"Asking Jev {len(features)} questions per draft…"):
        try:
            rows, skipped = judge.judge(client.decide, posts, features, workers=8)
        except Exception as e:  # network or API errors
            st.error(f"Jev call failed: {e}")
            return None
    if skipped:
        st.error(f"Session spending cap (${MAX_COST_PER_SESSION:.2f}) reached. Restart the app to reset it.")
        return None
    st.session_state["spent"] = client.usage.cost_usd
    return [{r["feature_id"]: r["p_yes"] for r in rows if r["urn"] == p["urn"]} for p in posts]


def evaluate_tab() -> None:
    st.subheader("Evaluate a draft post")
    b = card()
    feats = scorecard.jev_features(b)
    m = b["metrics"]
    st.caption(
        f"Jev answers {len(feats)} yes/no questions about the text (about ${len(feats) * judge.USD_PER_QUESTION:.4f} per draft) and the "
        f"scorecard adds up the {m['items_nonzero']} properties that matter, each tested on {m['train_authors']} authors. You get the expected "
        "reactions **relative to the author's usual post**, the drivers behind it, and the edits that should raise it. Text only: "
        "attached media is not used anywhere.")
    compare = st.toggle("Compare two drafts", help="Score two versions of a post and see which one the scorecard expects to do better.")
    n = 2 if compare else 1
    cols = st.columns(n)
    texts = [cols[i].text_area(f"Draft {'AB'[i]}" if compare else "Post text", height=240, key=f"draft{i}",
                               placeholder="Paste a LinkedIn post draft…") for i in range(n)]
    if st.button("Evaluate with Jev", type="primary", disabled=not all(t.strip() for t in texts)):
        answers = ask_jev(texts, feats)
        if answers is None:
            return
        st.session_state["eval"] = {"texts": texts, "answers": answers}

    ev = st.session_state.get("eval")
    if not ev or len(ev["texts"]) != n:
        return
    if ev["texts"] != texts:
        st.warning("The text changed since the last evaluation. Click **Evaluate with Jev** to score the new version.")
    reports = []
    for text, p_yes in zip(ev["texts"], ev["answers"]):
        values = scorecard.post_values(b, p_yes, text)
        result = scorecard.score(b, values)
        reports.append({"text": text, "p_yes": p_yes, "values": values, "result": result,
                        "suggest": scorecard.suggestions(b, result), "type": scorecard.post_type(b, values)})
    st.caption(f"Jev spend this session: ${st.session_state.get('spent', 0):.4f} (repeat evaluations of the same text are free).")
    st.divider()

    if compare:
        a, c = reports
        gap = a["result"]["score"] - c["result"]["score"]
        conf = scorecard.pair_confidence(b, gap)
        better, worse = ("A", "B") if gap >= 0 else ("B", "A")
        k = st.columns(3)
        k[0].metric("Draft A", f"{a['result']['multiplier']:.2f}×", a["result"]["band"]["name"], delta_color="off")
        k[1].metric("Draft B", f"{c['result']['multiplier']:.2f}×", c["result"]["band"]["name"], delta_color="off")
        k[2].metric(f"Draft {better} vs Draft {worse}", f"{math.exp(abs(gap)):.2f}×",
                    help="For the same author, the expected ratio of reactions: exp(score difference).")
        st.markdown(f"For two posts by the same held-out author with a gap this size, the higher-scoring one got more reactions "
                    f"**{conf['accuracy']:.0%}** of the time ({conf['share']:.1%} of pairs had a gap this size).")
        if abs(gap) > b["metrics"]["max_gap_seen"]:
            st.caption(f"This gap is larger than any seen between two posts by the same held-out author "
                       f"(up to {math.exp(b['metrics']['max_gap_seen']):.1f}×), so read the ratio as 'much better', not as an exact number.")
        diff = pd.DataFrame([{"driver": da["name"], "change": math.exp(da["contribution"] - dc["contribution"])}
                             for da, dc in zip(a["result"]["drivers"], c["result"]["drivers"])])
        diff = diff[(diff["change"] - 1).abs() > 0.005].sort_values("change", key=lambda s: -(np.log(s).abs()))
        if len(diff):
            st.markdown("**Where Draft A differs from Draft B**")
            st.altair_chart(driver_chart(diff, "change", "Draft A vs Draft B (×)"), use_container_width=True)
        for name, tab in zip("AB", st.tabs(["Draft A", "Draft B"])):
            with tab:
                report(b, reports["AB".index(name)])
    else:
        report(b, reports[0])


def driver_chart(df: pd.DataFrame, col: str, title: str) -> alt.Chart:
    lo, hi = min(df[col].min(), 0.95), max(df[col].max(), 1.05)
    return alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{col}:Q", title=title, scale=alt.Scale(type="log", nice=False, domain=[lo / 1.02, hi * 1.02]),
                axis=alt.Axis(values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], format=".2~f")),
        x2=alt.datum(1.0),
        y=alt.Y("driver:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.condition(f"datum['{col}'] > 1", alt.value("#2e7d32"), alt.value("#c62828")),
        tooltip=["driver", alt.Tooltip(f"{col}:Q", format=".2f")] + (["has"] if "has" in df else []),
    ).properties(height=alt.Step(26))


def report(b: dict, r: dict) -> None:
    res, sug, ptype = r["result"], r["suggest"], r["type"]
    band = res["band"]
    k = st.columns(3)
    k[0].metric("Expected reactions vs the author's usual post", f"{res['multiplier']:.2f}×",
                help="Calibrated on held-out authors for an author you know nothing else about. 1.00× = a typical post for that author.")
    k[1].metric("Score band", band["name"], help="Bands cut at the 10th, 30th, 70th and 90th percentile of train posts.")
    k[2].metric("Kind of post", ptype["name"], help=ptype["description"])
    st.markdown(f"Held-out posts in the **{band['name'].lower()}** band got **{band['actual']:.2f}×** their author's usual reactions on "
                f"average, beat it **{band['above_usual']:.0%}** of the time, got at least double it **{band['hit_rate']:.0%}** of the time "
                f"and half or less **{band['flop_rate']:.0%}** of the time.")

    st.markdown("**What moves this post's score** (vs an average post)")
    drv = pd.DataFrame([{"driver": d["name"], "effect": math.exp(d["contribution"]),
                         "has": "; ".join(i["label"] for i in d["has"])[:300]} for d in res["drivers"]])
    drv = drv[(drv["effect"] - 1).abs() > 0.005].sort_values("effect", key=lambda s: -(np.log(s).abs()))
    if len(drv):
        st.altair_chart(driver_chart(drv, "effect", "Effect on expected reactions (×)"), use_container_width=True)
        st.caption("A red bar on a helpful driver means the post has less of it than an average post: a missed boost, not a penalty.")

    st.markdown("**Edits that should help**")
    proven = {lv["id"]: lv for lv in ptype.get("levers", [])}
    if sug["edits"]:
        t = pd.DataFrame([{
            "do": ("Add: " if s["action"] == "add" else "Remove: ") + s["label"],
            "expected gain": s["gain_multiplier"],
            "evidence (alone, all authors)": s["single"]["mult"] if s["single"] else None,
            "held-out authors": s["single"]["test_mult"] if s["single"] else None,
            "within this kind of post": "✓ " + f"{proven[s['id']]['mult']:.2f}×" if s["id"] in proven else "",
        } for s in sug["edits"][:10]])
        st.dataframe(t, hide_index=True, use_container_width=True, column_config={
            "expected gain": st.column_config.NumberColumn(format="%.2f×"),
            "evidence (alone, all authors)": st.column_config.NumberColumn(format="%.2f×"),
            "held-out authors": st.column_config.NumberColumn(format="%.2f×")})
        st.caption("Only properties that were confirmed on held-out authors on their own and that a writer can choose. Expected gain is the "
                   "scorecard's weight: for the same author, about that many times the reactions, if the association holds for your post. "
                   "Gains overlap, so two related edits add less than their product. ✓ = also significant within this kind of post. "
                   "Use what fits the post, then evaluate again.")
    else:
        st.write("No confirmed edit would add much. The post already has what the scorecard rewards.")
    if sug["if_true"]:
        st.markdown("**If it's true, say it**: real occasions that go with more reactions")
        st.markdown("\n".join(f"- {s['label']} (about {s['gain_multiplier']:.2f}×)" for s in sug["if_true"][:4]))

    levers = [lv for lv in ptype.get("levers", []) if lv["mult"] > 1][:6]
    if levers:
        with st.expander(f"What works best in '{ptype['name']}' posts"):
            st.caption(f"{ptype['description']} This kind of post gets {ptype['reactions']:.2f}× the author's usual reactions on average. "
                       "Properties significant within this kind of post and in the same direction on held-out authors:")
            st.markdown("\n".join(f"- {lv['label']}: **{lv['mult']:.2f}×** (held-out {lv['test_mult']:.2f}×)" for lv in levers))

    with st.expander("What the post already does"):
        for d in res["drivers"]:
            if d["has"]:
                st.markdown(f"**{d['name']}**: " + "; ".join(f"{i['label']} ({i['value']:.2f})" for i in d["has"]))

    if (PROCESSED / "models" / "model_bundle.json").exists():
        with st.expander("Previous dashboard model (theme model), from the same answers"):
            tb = bundle()
            row = build_row(tb, r["p_yes"], "text", r["text"])
            pred, _ = predict_linear(tb, row)
            st.metric("Theme model: expected reactions vs the author's usual", fmt_x(pred))
            st.caption("The 7-theme linear model from the Pipeline tab. It ranks posts about as well as the scorecard "
                       "(0.613 vs 0.616 on held-out authors) but gives no edit-level advice.")

    with st.expander(f"All {len(r['p_yes'])} Jev answers"):
        labels = {i["id"]: i["label"] for i in b["items"]}
        q = {f["id"]: f["question"] for f in b["features"]}
        table = pd.DataFrame([{"question": labels.get(k) or q.get(k, k), "P(yes)": v} for k, v in r["p_yes"].items()]
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
    jev_cost = sum((read_json(PROCESSED / f"{d}_{s}" / "run_meta.json") or {}).get("usage", {}).get("cost_usd", 0)
                   for d in ({"judgments_full", "judgments_text"} if full else {"judgments_text"}) for s in ("train", "test"))
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
                      f"${jev_cost:.2f} in total (answers from an earlier subset run were reused from cache)."),
        ("6 · Model", f"Questions that tend to be true together were grouped into **{data.get('themes', '–')} "
                      "themes** (checklists); themes with fewer than 3 questions were set aside as too niche. "
                      "A sparse linear model picks the themes that predict lift, **content only**: no media, length, "
                      "or posting time. Tuned with author-grouped CV. Caveat: the theme design (15 themes, ≥ 3 "
                      "questions, ≤ 10 themes) was settled while looking at an earlier 45-author test sample, which is "
                      "part of today's 165 test authors, so the test score may be slightly optimistic."),
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
        "vs format reference (paired 95% CI)": (
            "—" if k == "controls" else
            "{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]".format(mean=metrics[k]["test"]["gain_over_controls"]["mean"],
                                                        lo=metrics[k]["test"]["gain_over_controls"]["ci95"][0],
                                                        hi=metrics[k]["test"]["gain_over_controls"]["ci95"][1])),
        "pairwise accuracy (CV)": metrics[k]["cv_train"]["pairwise_accuracy"],
        "per-author Spearman (test)": metrics[k]["test"]["author_spearman"],
    } for k in names])
    st.dataframe(res, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in res.columns
                                if c.startswith(("pairwise", "per-author"))})
    st.caption("Pairwise accuracy: for two posts by the same author, how often the model ranks the one that did "
               "better higher. 0.5 = chance. The 'vs format reference' column compares each model with the format "
               "reference on the same authors (paired). The sparse linear model on content themes is clearly better than format alone "
               "(the range excludes 0); RuleFit's lead over format is within noise.")

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
        d = r["test"].get("diff_vs_current")
        if d:
            lo, hi = d["ci95"]
            verdict = "no clear difference" if lo <= 0 <= hi else ("better" if lo > 0 else "worse")
            if verdict != "no clear difference" and min(abs(lo), abs(hi)) < 0.002:
                verdict += " (borderline)"
            vs = f"{d['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] · {verdict}"
        else:
            vs = "—"
        rows.append({
            "model": label, "input": inputs,
            "train data": (f"{r.get('train_authors', '–')} authors ({r.get('train_posts', 0):,} posts)"
                           if trained == "all" else subset),
            "CV": r["cv_train"]["pairwise_accuracy"], "test": r["test"]["pairwise_accuracy"],
            "low": ci[0], "high": ci[1], "vs current model (paired 95% CI)": vs,
        })
    df = pd.DataFrame(rows).sort_values("test", ascending=False)
    df["test 95% CI"] = df.apply(lambda r: f"[{r['low']:.3f}, {r['high']:.3f}]" if r["low"] is not None else "—", axis=1)
    st.dataframe(df[["model", "test", "test 95% CI", "vs current model (paired 95% CI)", "CV", "input", "train data"]],
                 hide_index=True,
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
        color=alt.condition(alt.datum.model == "Themes, sparse linear (current)", alt.value("#2e7d32"), alt.value("#9e9e9e")),
        tooltip=["model", alt.Tooltip("test:Q", format=".3f"), alt.Tooltip("CV:Q", format=".3f"), "train data"])
    chance = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="#c62828").encode(x="x:Q")
    st.altair_chart((whiskers + dots + chance).properties(height=alt.Step(34)), use_container_width=True)
    st.caption(
        f"All models predict the same target (lift vs. the author's usual), are tuned with the same author-grouped "
        f"folds (ridge strength, number of trees), and are tested on the same {data.get('test_authors', '–')} held-out "
        f"authors ({data.get('test_posts', 0):,} posts). Overlapping ranges are not the right test, so the 'vs current model' column "
        "compares each model with the current one author by author (paired). Reading it: the theme model ties ridge on "
        "all 147 questions, is clearly better than format alone, and its lead over TF-IDF and gradient boosting is "
        "small (about 0.02) and at the edge of noise. Its value is matching the best baseline with 7 readable weights. "
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
        "theme is slightly better than 14 single questions (0.602 vs 0.595, within noise). Dashed line: ridge on all "
        "questions. Single-question picks are not stable: a 10-question sparse model rerun on other training folds "
        "always keeps only about 5 of the same questions.")

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


tab1, tab2, tab3, tab4 = st.tabs(["Evaluate a post", "What works", "Evidence", "Pipeline"])
with tab1:
    if scorecard.BUNDLE.exists():
        evaluate_tab()
    else:
        st.info("Build the scorecard first: uv run python research/scorecard.py")
with tab2:
    what_works_tab()
with tab3:
    evidence_tab()
with tab4:
    st.caption("The original pipeline and the 7-theme model that the first version of this dashboard used.")
    pipeline_tab()
