"""LinkedIn post dashboard: evaluate a draft with a research scorecard, the scorecards' weights and accuracy, research findings, and the pipeline.

Run:
    uv run streamlit run app/dashboard.py

Needs OPENROUTER_API_KEY in .env and data/processed/research/scorecard.json (research/scorecard.py).
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # see README "Imports on this Mac"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from insights import scorecard_tab, what_works_tab  # noqa: E402
from linkedin_analysis import judge, scorecard  # noqa: E402
from linkedin_analysis.llm import DecisionsClient, LLMSettings  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402

MAX_COST_PER_SESSION = 0.25  # USD, Jev spend cap for one browser session

st.set_page_config(page_title="LinkedIn Post Lab", page_icon="📝", layout="wide")


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_resource
def jev() -> DecisionsClient:
    return DecisionsClient(LLMSettings.from_env("openrouter"), max_cost_usd=MAX_COST_PER_SESSION, max_connections=8)


# Tab 1: evaluate a post (research scorecard)


@st.cache_resource
def _card(mtime: float) -> dict:
    return scorecard.load()


def card() -> dict:
    """The research scorecards (research/scorecard.py); reloaded when the file is rewritten."""
    return _card(scorecard.BUNDLE.stat().st_mtime)


def size_name(b: dict, size: str) -> str:
    return f"{size} questions (full)" if size == b["default"] else f"{size} questions"


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


def answers_for(texts: list[str], feats: list[dict]) -> list[dict[str, float]] | None:
    """Jev's answers for each text, asking only for questions not already answered in this session."""
    cache = st.session_state.setdefault("answers", {})
    todo = [t for t in dict.fromkeys(texts) if any(f["id"] not in cache.get(t, {}) for f in feats)]
    if todo:
        missing = [f for f in feats if any(f["id"] not in cache.get(t, {}) for t in todo)]
        new = ask_jev(todo, missing)
        if new is None:
            return None
        for t, a in zip(todo, new):
            cache[t] = {**cache.get(t, {}), **a}
    return [cache[t] for t in texts]


def known(texts: list[str], feats: list[dict]) -> list[dict[str, float]] | None:
    """Answers already in this session for every text, or None (never calls Jev)."""
    cache = st.session_state.get("answers", {})
    if all(t in cache and all(f["id"] in cache[t] for f in feats) for t in texts):
        return [cache[t] for t in texts]
    return None


def evaluate_tab() -> None:
    b = card()
    st.subheader("Evaluate a post")
    sizes = scorecard.sizes(b)
    size = st.radio("Scorecard", sizes, index=sizes.index(b["default"]), horizontal=True,
                    format_func=lambda s: size_name(b, s))
    c = b["sizes"][size]
    m = c["metrics"]
    feats = scorecard.jev_features(b, size)
    st.caption(f"{m['jev_questions']} Jev questions + {m['questions'] - m['jev_questions']} text checks + length · "
               f"held-out accuracy **{m['test_pairwise']:.1%}** (50% = coin flip) · about "
               f"${len(feats) * judge.USD_PER_QUESTION:.4f} per draft. Text only: attached media is not used.")

    compare = st.toggle("Compare two drafts")
    n = 2 if compare else 1
    cols = st.columns(n)
    texts = [cols[i].text_area(f"Draft {'AB'[i]}" if compare else "Post text", height=220, key=f"draft{i}",
                               placeholder="Paste a LinkedIn post draft…") for i in range(n)]
    clicked = st.button("Evaluate", type="primary", disabled=not all(t.strip() for t in texts))
    if clicked:
        if answers_for(texts, feats) is None:
            return
        st.session_state["eval"] = texts

    ev = st.session_state.get("eval")
    if not ev or len(ev) != n:
        return
    if ev != texts:
        st.warning("The text changed since the last evaluation. Click **Evaluate** to score the new version.")
    answers = known(ev, feats)
    if answers is None:
        st.info(f"The {size_name(b, size)} scorecard needs answers Jev has not given yet. Click **Evaluate**.")
        return
    results = [scorecard.score(c, scorecard.post_values(c, a, t)) for t, a in zip(ev, answers)]
    st.caption(f"Jev spend this session: ${st.session_state.get('spent', 0):.4f}.")
    st.divider()
    if compare:
        compare_report(c, *results)
    else:
        report(c, results[0])


EXPLAIN = ("**Answer**: Jev's probability that the answer is yes (text checks and length are counted directly). "
           "**Weight**: how much the score changes from a clear no to a clear yes. "
           "**Contribution** = weight × (answer − the average post's answer); the contributions add up to the score, so "
           "negative rows are what pull this post down. For two posts by the same author, a score 0.10 higher means about 10% "
           "more reactions.")


def answer_text(row: dict) -> str:
    if row["id"] == "length":
        return f"{int(row['answer'])} words"
    if row["id"].startswith("text."):
        return "yes" if row["answer"] >= 0.5 else "no"
    return f"{'yes' if row['answer'] >= 0.5 else 'no'} ({row['answer']:.2f})"


def fmt_signed(v: float) -> str:
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}"


def signed(v: float) -> str:
    return "color: #2e7d32" if v > 0.005 else "color: #c62828" if v < -0.005 else ""


def report(c: dict, r: dict) -> None:
    band = r["band"]
    k = st.columns(3)
    k[0].metric("Score", f"{r['score']:+.2f}", help="Sum of the contributions below. 0 = an average post.")
    k[1].metric("Expected reactions vs usual", f"{r['multiplier']:.2f}×",
                help="Expected reactions compared with the author's usual post, calibrated on held-out authors. About half the score shows up here, because part of any score "
                     "is the author's usual style, which is already in their usual reactions.")
    k[2].metric("Score band", band["name"])
    st.caption(f"Held-out posts in the **{band['name'].lower()}** band got {band['actual']:.2f}× their author's usual "
               f"reactions on average and beat it {band['above_usual']:.0%} of the time.")
    t = pd.DataFrame([{"Question": row["label"], "Answer": answer_text(row), "Weight": row["weight"],
                       "Contribution": row["contribution"]} for row in r["rows"]]).sort_values("Contribution", ascending=False)
    st.dataframe(t.style.format({"Weight": fmt_signed, "Contribution": fmt_signed}).map(signed, subset=["Weight", "Contribution"]),
                 hide_index=True, use_container_width=True, height=min(38 + 35 * len(t), 700))
    st.caption(EXPLAIN)


def compare_report(c: dict, a: dict, bb: dict) -> None:
    gap = a["score"] - bb["score"]
    conf = scorecard.pair_confidence(c, gap)
    better, worse = ("A", "B") if gap >= 0 else ("B", "A")
    k = st.columns(3)
    k[0].metric("Draft A score", f"{a['score']:+.2f}")
    k[1].metric("Draft B score", f"{bb['score']:+.2f}")
    k[2].metric(f"Draft {better} vs Draft {worse}", f"{math.exp(abs(gap)):.2f}×",
                help="For the same author, the expected ratio of reactions: exp(score difference).")
    st.caption(f"Expected reactions vs the author's usual post: Draft A {a['multiplier']:.2f}× ({a['band']['name'].lower()} band), "
               f"Draft B {bb['multiplier']:.2f}× ({bb['band']['name'].lower()} band).")
    if conf["accuracy"] is not None:
        st.caption(f"For two posts by the same held-out author with a score gap this size, the higher-scoring one got more "
                   f"reactions **{conf['accuracy']:.0%}** of the time.")
    if abs(gap) > c["metrics"]["max_gap_seen"]:
        st.caption(f"This gap is larger than any seen between two posts by the same held-out author "
                   f"(up to {math.exp(c['metrics']['max_gap_seen']):.1f}×): read it as 'much better', not as an exact number.")
    t = pd.DataFrame([{"Question": ra["label"], "Weight": ra["weight"], "Answer A": answer_text(ra), "Answer B": answer_text(rb),
                       "Contribution A": ra["contribution"], "Contribution B": rb["contribution"],
                       "A − B": ra["contribution"] - rb["contribution"]} for ra, rb in zip(a["rows"], bb["rows"])])
    t = t.sort_values("A − B", key=lambda s: -s.abs())
    num = ["Weight", "Contribution A", "Contribution B", "A − B"]
    st.dataframe(t.style.format({k: fmt_signed for k in num}).map(signed, subset=num),
                 hide_index=True, use_container_width=True, height=min(38 + 35 * len(t), 700))
    st.caption(EXPLAIN + " **A − B**: where the two drafts differ, largest first.")


# Tab 4: pipeline


def count(path: Path) -> int:
    d = read_json(path)
    return len(d["features"]) if d else 0


def pipeline_tab() -> None:
    labels = read_json(PROCESSED / "labels_summary.json") or {}
    split = (read_json(PROCESSED / "splits_summary.json") or {}).get("by_split", {})
    tr, te = split.get("train", {}), split.get("test", {})
    proposed = count(PROCESSED / "features" / "feature_bank.json")
    kept = count(PROCESSED / "features" / "feature_bank_general.json")
    new = count(PROCESSED / "research" / "bank_round1.json") + count(PROCESSED / "research" / "bank_round2.json")
    base_cost = sum((read_json(PROCESSED / f"judgments_full_{s}" / "run_meta.json") or {}).get("usage", {}).get("cost_usd", 0)
                    for s in ("train", "test"))
    ledger = read_json(ROOT / "research" / "results" / "jev_ledger.json") or {}
    eff = pd.concat([pd.read_csv(ROOT / "research" / "results" / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]])
    passed = int(eff["status"].isin(["confirmed", "not replicated"]).sum())
    b = card() if scorecard.BUNDLE.exists() else None
    sizes = ", ".join(scorecard.sizes(b)) if b else "–"
    full = b["sizes"][b["default"]]["metrics"] if b else {}

    st.subheader("Pipeline")
    steps = [
        ("1. Collect", "Public LinkedIn posts (text only) from authors with 1k–20k followers, Jul–Oct 2026.",
         f"{labels.get('posts_in', 0):,} posts, {labels.get('authors_in', 0):,} authors"),
        ("2. Label", "Compare each post with the same author's usual post: lift = log(reactions ÷ author's usual).",
         f"{labels.get('posts_labeled', 0):,} posts from {labels.get('authors_labeled', 0):,} authors with 5+ posts"),
        ("3. Split", "Split by author. Test authors are never used to choose anything.",
         f"{tr.get('authors', '–')} train authors, {te.get('authors', '–')} test authors"),
        ("4. Ask questions", "DeepSeek proposes yes/no questions about the content; free code checks are added "
         "(hashtags, links, length…).", f"{kept} of {proposed} first-round questions kept, {new} more in two research rounds, 15 text checks"),
        ("5. Answer", "Jev reads each post's text and gives P(yes) for every question.",
         f"${base_cost:.2f} + ${ledger.get('spent_usd', 0):.2f} of Jev"),
        ("6. Test", "Keep a question only if it matters on train authors after correcting for many tests; "
         "'confirmed' if it also holds on test authors.", f"{len(eff)} tested, {passed} kept, {int(eff['confirmed'].sum())} confirmed"),
        ("7. Scorecard", "One weight per kept question plus length, fitted on train authors; direction fixed in advance. "
         "Smaller versions keep the questions that matter most.", f"sizes {sizes}; held-out accuracy {full.get('test_pairwise', 0):.1%}"),
        ("8. Evaluate", "For a new post, Jev answers the chosen scorecard's questions and the score is the weighted sum.",
         "about $0.0004 per draft"),
    ]
    st.markdown("| Step | What happens | Result |\n|---|---|---|\n"
                + "\n".join(f"| **{a}** | {b} | {c} |" for a, b, c in steps))
    st.caption("Held-out accuracy: for two posts by the same test author, how often the higher-scoring one got more reactions "
               "(50% = coin flip). The first dashboard used a 7-theme model (61.3%); the scorecard replaced it. "
               "Details: README.md, research/README.md and research/REPORT.md.")


tab1, tab2, tab3, tab4 = st.tabs(["Evaluate a post", "Scorecard", "Research findings", "Pipeline"])
with tab1:
    if scorecard.BUNDLE.exists():
        evaluate_tab()
    else:
        st.info("Build the scorecard first: uv run python research/scorecard.py")
with tab2:
    if scorecard.BUNDLE.exists():
        scorecard_tab(card())
with tab3:
    what_works_tab()
with tab4:
    pipeline_tab()
