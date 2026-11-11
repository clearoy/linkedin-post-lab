"""Compact theme model with only a few Jev questions, plus a question-count curve.

Usage:
    uv run python scripts/compact_model.py \
        --judgments data/processed/judgments_full_train data/processed/judgments_full_test

Run after train_models.py (and name_themes.py). For each theme the full linear
model uses, keeps the --per-theme questions that track the full theme score
most closely on train posts (correlation), scores the theme as their mean, and
refits the linear weights (OLS) on train. CV redoes the question choice inside
each fold. Test authors are scored once.

Writes data/processed/models/:
    model_bundle_compact.json  same layout as model_bundle.json, read by the dashboard
    question_budget.json       test accuracy vs number of Jev questions:
                               - themes with their top-m questions (m = 1, 2, 3, 5, all)
                               - single questions added one at a time (forward selection by CV error)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from linkedin_analysis.model import ATTACHMENT_FEATURES, bootstrap_ci, cv_splits, evaluate, pairwise_by_author  # noqa: E402
from linkedin_analysis.paths import PROCESSED  # noqa: E402

MODELS = PROCESSED / "models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judgments", type=Path, nargs="+", required=True)
    p.add_argument("--per-theme", type=int, default=2, help="questions kept per theme in the compact model")
    p.add_argument("--max-single", type=int, default=15, help="forward selection goes up to this many questions")
    return p.parse_args()


def top_members(themes: dict, rows: pd.DataFrame, m: int | None) -> dict[str, list[str]]:
    """Per theme, the m members most correlated with the full theme score (all members if m is None)."""
    out = {}
    for t, v in themes.items():
        if m is None:
            out[t] = list(v["members"])
            continue
        full = rows[v["members"]].mean(axis=1)
        out[t] = list(rows[v["members"]].corrwith(full).sort_values(ascending=False).index[:m])
    return out


def scores(rows: pd.DataFrame, members: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame({t: rows[c].mean(axis=1) for t, c in members.items()}, index=rows.index)


def metrics(y, pred, authors) -> dict:
    r = evaluate(y, pred, authors)
    r["pairwise_accuracy_ci95"] = bootstrap_ci(pairwise_by_author(y, pred, authors))["ci95"]
    return r


def rename(themes: dict) -> None:
    """With only a few questions a theme can mean something narrower than its full name
    (e.g. a job-listing theme measured by two list-format questions), so DeepSeek renames
    each compact theme from its own questions. Keeps the full theme's name if DeepSeek is unavailable."""
    try:
        from linkedin_analysis.discovery import load_prompt
        from linkedin_analysis.llm import LLMClient, LLMSettings
        client = LLMClient(LLMSettings.from_env("deepseek"))
        reply = client.chat_json(load_prompt("name_themes_system").template,
                                 json.dumps({t: v["member_questions"] for t, v in themes.items()}, indent=1),
                                 temperature=0.0)
    except Exception as e:  # missing key, network, bad JSON
        print(f"Theme renaming skipped ({e}); keeping the full themes' names.")
        return
    for t, v in themes.items():
        got = reply.get("themes", {}).get(t, {})
        if got.get("name"):
            v["full_name"] = v["name"]
            v["name"], v["description"] = str(got["name"]).strip(), str(got.get("description", "")).strip()


def main() -> None:
    args = parse_args()
    full = json.loads((MODELS / "model_bundle.json").read_text())
    names = json.loads((MODELS / "theme_names.json").read_text()) if (MODELS / "theme_names.json").exists() else {}
    used = {t: v for t, v in full["themes"].items() if t in full["linear"]["coef"]}

    wide = pd.concat([pd.read_parquet(d / "features.parquet") for d in args.judgments]).drop_duplicates("urn")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "author", "lift"]]
    splits = pd.read_parquet(PROCESSED / "splits.parquet")[["author", "split", "fold"]]
    df = wide.merge(labels, on="urn").merge(splits, on="author").reset_index(drop=True)
    train = df["split"] == "train"
    q_all = [c for c in wide.columns if c != "urn" and c not in ATTACHMENT_FEATURES]
    prev = df.loc[train, q_all].mean()
    Q = [c for c in q_all if 0.02 <= prev[c] <= 0.98]
    df[Q] = df[Q].fillna(df.loc[train, Q].mean())
    tr, te = df[train].reset_index(drop=True), df[~train].reset_index(drop=True)
    y, folds, a = tr["lift"].to_numpy(), tr["fold"].to_numpy(), tr["author"].to_numpy()
    yt, at = te["lift"].to_numpy(), te["author"].to_numpy()
    fold_idx = cv_splits(folds)

    def theme_model(m: int | None):
        oof, signs = np.zeros(len(y)), []
        for i, v in fold_idx:
            mem = top_members(used, tr.iloc[i], m)
            ols = LinearRegression().fit(scores(tr.iloc[i], mem), y[i])
            oof[v] = ols.predict(scores(tr.iloc[v], mem))
            signs.append(np.sign(ols.coef_))
        mem = top_members(used, tr, m)
        S_tr = scores(tr, mem)
        ols = LinearRegression().fit(S_tr, y)
        pred = ols.predict(scores(te, mem))
        return mem, S_tr, ols, np.array(signs), evaluate(y, oof, a), metrics(yt, pred, at)

    budget = []
    for m in [1, 2, 3, 5, None]:
        mem, _, _, _, cv, test = theme_model(m)
        n = sum(len(c) for c in mem.values())
        budget.append({"method": "themes", "per_theme": m or "all", "questions": n,
                       "cv": cv["pairwise_accuracy"], "test": test["pairwise_accuracy"],
                       "ci95": test["pairwise_accuracy_ci95"]})
        print(f"{len(used)} themes x top-{m or 'all'}: {n:3d} questions  cv={cv['pairwise_accuracy']:.4f} "
              f"test={test['pairwise_accuracy']:.4f} {test['pairwise_accuracy_ci95']}", flush=True)

    # Single questions, added one at a time by out-of-fold squared error (train only).
    F, Ft = tr[Q].to_numpy(), te[Q].to_numpy()

    def oof_ols(cols):
        pred = np.zeros(len(y))
        for i, v in fold_idx:
            pred[v] = LinearRegression().fit(F[i][:, cols], y[i]).predict(F[v][:, cols])
        return pred

    chosen = []
    for _ in range(args.max_single):
        best = min((j for j in range(len(Q)) if j not in chosen), key=lambda j: ((oof_ols(chosen + [j]) - y) ** 2).mean())
        chosen.append(best)
        pred = LinearRegression().fit(F[:, chosen], y).predict(Ft[:, chosen])
        test = metrics(yt, pred, at)
        budget.append({"method": "single", "questions": len(chosen), "added": Q[best],
                       "cv": evaluate(y, oof_ols(chosen), a)["pairwise_accuracy"],
                       "test": test["pairwise_accuracy"], "ci95": test["pairwise_accuracy_ci95"]})
        print(f"single n={len(chosen):2d} test={test['pairwise_accuracy']:.4f} + {Q[best]}", flush=True)

    ridge = json.loads((MODELS / "baselines.json").read_text()).get("jev_ridge") if (MODELS / "baselines.json").exists() else None
    if ridge:
        budget.append({"method": "all_ridge", "questions": len(Q), "cv": ridge["cv_train"]["pairwise_accuracy"],
                       "test": ridge["test"]["pairwise_accuracy"], "ci95": ridge["test"].get("pairwise_accuracy_ci95")})
    (MODELS / "question_budget.json").write_text(json.dumps(
        {"per_theme_compact": args.per_theme, "test_authors": int(te["author"].nunique()), "test_posts": len(te),
         "rows": budget}, indent=2))

    # The compact model itself.
    mem, S_tr, ols, signs, cv, test = theme_model(args.per_theme)
    question = {f["id"]: f for f in full["features"]}
    coef = dict(zip(mem, map(float, ols.coef_)))
    themes = {}
    for j, (t, members) in enumerate(mem.items()):
        q25, q75 = (float(x) for x in S_tr[t].quantile([0.25, 0.75]))
        themes[t] = {
            "members": members, "central": members[0],
            "member_questions": [question[q]["question"] for q in members],
            "coef": coef[t], "q25": q25, "q75": q75,
            "realistic_x": float(np.exp(coef[t] * (q75 - q25))), "full_x": float(np.exp(coef[t])),
            "sign_stability": float((signs[:, j] == np.sign(coef[t])).mean()),
            "member_yes_rate": [float((tr[q] >= 0.5).mean()) for q in members],
            "label": used[t]["label"],
            "name": names.get(t, {}).get("name", used[t]["label"]),
            "description": names.get(t, {}).get("description", ""),
            "full_members": len(used[t]["members"]),
        }
    rename(themes)
    keep = [q for c in mem.values() for q in c]
    bundle = {
        "variant": "compact",
        "per_theme": args.per_theme,
        "features": [question[q] for q in keep],
        "themes": themes,
        "column_means": {t: float(S_tr[t].mean()) for t in mem},
        "linear": {"intercept": float(ols.intercept_), "coef": coef},
        "metrics": {"linear": {"cv_train": cv, "test": test}, "data": full["metrics"]["data"]},
    }
    (MODELS / "model_bundle_compact.json").write_text(json.dumps(bundle, indent=2))
    print(f"\nCompact model: {len(keep)} questions, test {test['pairwise_accuracy']:.4f} "
          f"{test['pairwise_accuracy_ci95']}. Wrote {MODELS / 'model_bundle_compact.json'} and question_budget.json")


if __name__ == "__main__":
    main()
