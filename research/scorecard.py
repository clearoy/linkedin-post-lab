"""Fit, calibrate and test the scorecard that the dashboard uses to evaluate a post.

Usage:
    uv run python research/scorecard.py

Model. Every property that passed discovery on train authors (research/drivers.json)
gets its own weight on Jev's P(yes), fitted jointly within author, with the
direction fixed in advance: a property in a helpful driver can only add, one in a
harmful driver can only subtract (bounded least squares, small ridge penalty).
So no weight can flip sign because of a correlated neighbour, and a property that
adds nothing beyond the others gets exactly 0. Length enters as log(min(words, cap)).

Everything is chosen on train authors: the ridge strength and the length cap by
5-fold author CV (all 667 train authors), and a shrink factor so that out-of-fold predictions are
calibrated (slope 1). Test authors are used once, to report accuracy, the
calibration bands and the score-gap table.

Writes
  data/processed/research/scorecard.json   the bundle the dashboard loads (has Jev question definitions)
  research/results/scorecard.json          accuracy, calibration, bands, drivers (statistics only)
  research/results/scorecard_items.csv     every item's weight and its single-property evidence
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

from compare_dashboard import DIRS as BASE_DIRS  # noqa: E402
from joint import frame  # noqa: E402
from outcomes import add_outcomes  # noqa: E402
from playbook import author_boot  # noqa: E402
from stats import PROCESSED, RESEARCH_DATA, RESULTS, demean, within_ols  # noqa: E402
from subset import Pairs  # noqa: E402

from linkedin_analysis.model import bootstrap_ci, cv_splits, evaluate, pairwise_by_author  # noqa: E402

HERE = Path(__file__).resolve().parent
# the 150 generation authors were scored on the new questions afterwards (judgments_r*_generate), so the
# scorecard trains on all 667 train authors; test authors are unchanged
DIRS = BASE_DIRS + [RESEARCH_DATA / "judgments_r1_generate", RESEARCH_DATA / "judgments_r2_generate"]
CAPS = [None, 100, 120, 160, 200]
RIDGE = [0.0, 10.0, 30.0]
BAND_QUANTILES = [0.10, 0.30, 0.70, 0.90]
BAND_NAMES = ["Low", "Below average", "Average", "Above average", "High"]


def length_col(words: pd.Series, cap: int | None) -> np.ndarray:
    return np.log(np.minimum(words, cap) if cap else words).to_numpy(float)


def bounded_fit(Xd: np.ndarray, yd: np.ndarray, sign: np.ndarray, lam: float) -> np.ndarray:
    """Least squares with each weight held to its sign (+1: >= 0, -1: <= 0) and a ridge penalty."""
    lo, hi = np.where(sign > 0, 0.0, -np.inf), np.where(sign > 0, np.inf, 0.0)
    if lam > 0:
        Xd = np.vstack([Xd, np.sqrt(lam) * np.eye(Xd.shape[1])])
        yd = np.r_[yd, np.zeros(Xd.shape[1])]
    return lsq_linear(Xd, yd, bounds=(lo, hi)).x


def oof_predictions(X, y, authors, folds, sign, lam) -> tuple[np.ndarray, list[np.ndarray]]:
    """Out-of-fold scores, each centred on its own fit's average post (as the dashboard scores a draft)."""
    Xd, yd = demean(X, authors), demean(y, authors)
    out, betas = np.zeros(len(y)), []
    for fit, val in cv_splits(folds):
        b = bounded_fit(Xd[fit], yd[fit], sign, lam)
        out[val] = (X[val] - X[fit].mean(axis=0)) @ b
        betas.append(b)
    return out, betas


def main() -> None:
    spec = json.loads((HERE / "drivers.json").read_text())
    df = add_outcomes(frame(DIRS))
    items = [(m[0], m[1], m[2], d) for d, v in spec["drivers"].items() for m in v["members"]]
    ids = [i[0] for i in items]
    sign = np.array([1.0 if spec["drivers"][i[3]]["direction"] == "+" else -1.0 for i in items] + [1.0])  # last: length
    bundle_dash = json.loads((PROCESSED / "models" / "model_bundle.json").read_text())
    df["dashboard"] = bundle_dash["linear"]["intercept"] + sum(
        c * df[bundle_dash["themes"][t]["members"]].mean(axis=1) for t, c in bundle_dash["linear"]["coef"].items())

    full = df[df[ids].notna().all(axis=1)].reset_index(drop=True)
    tr = full[full["split"] == "train"].reset_index(drop=True)
    te = full[full["split"] == "test"].reset_index(drop=True)
    assert len(te) == (df["split"] == "test").sum()
    y, au, folds = tr["lift"].to_numpy(), tr["author"].to_numpy(), tr["fold"].to_numpy()
    print(f"train {tr['author'].nunique()} authors / {len(tr)} posts; test {te['author'].nunique()} / {len(te)}; {len(ids)} items")

    # 1. choose length cap and ridge strength by author CV on train
    pairs, scores = Pairs(y, au), {}
    for cap in CAPS:
        X = np.column_stack([tr[ids].to_numpy(float), length_col(tr["words"], cap)])
        for lam in RIDGE:
            scores[(cap, lam)] = pairs.per_author(oof_predictions(X, y, au, folds, sign, lam)[0])
    best = max(scores, key=lambda k: scores[k].mean())
    close = [k for k, s in scores.items()
             if (s - scores[best]).mean() >= -(s - scores[best]).std(ddof=1) / np.sqrt(len(s)) or k == best]
    cap, lam = min(close, key=lambda k: (k[0] or 10**9, -k[1]))  # tie-break: capped length, then more shrinkage
    for k, s in scores.items():
        print(f"   cap {str(k[0]):>4}  ridge {k[1]:>4}  cv {s.mean():.4f}{'  <- best' if k == best else ''}{'  <- chosen' if k == (cap, lam) else ''}")

    # 2. fit, then shrink so that out-of-fold predictions have calibration slope 1
    X = np.column_stack([tr[ids].to_numpy(float), length_col(tr["words"], cap)])
    Xt = np.column_stack([te[ids].to_numpy(float), length_col(te["words"], cap)])
    oof, fold_betas = oof_predictions(X, y, au, folds, sign, lam)
    shrink = float(within_ols(y, oof, au)["beta"][0])
    beta = bounded_fit(demean(X, au), demean(y, au), sign, lam) * shrink
    means = X.mean(axis=0)
    stable = np.mean([np.abs(b) > 1e-6 for b in fold_betas], axis=0)
    print(f"out-of-fold calibration slope {shrink:.3f} (weights multiplied by it); "
          f"{int((np.abs(beta[:-1]) > 1e-6).sum())} of {len(ids)} items have a non-zero weight")

    # calibration for a single post by an unknown author: pooled fit of lift on the out-of-fold score.
    # Within one author a score difference of d means about exp(d) (slope 1 by construction); across
    # authors part of the score is the author's habitual style, which lift (already author-relative) removes.
    s_tr = oof * shrink
    cal_b, cal_a = np.polyfit(s_tr, y, 1)
    print(f"pooled calibration on train out-of-fold: lift = {cal_a:.3f} + {cal_b:.3f} * score")

    # 3. test authors, once
    s_te = (Xt - means) @ beta
    yt, at = te["lift"].to_numpy(), te["author"].to_numpy()
    per, dash = pairwise_by_author(yt, s_te, at), pairwise_by_author(yt, te["dashboard"].to_numpy(), at)
    ev = evaluate(yt, s_te, at)
    slope = within_ols(yt, s_te, at)
    metrics = {
        "train_authors": int(tr["author"].nunique()), "train_posts": len(tr), "test_authors": int(te["author"].nunique()),
        "test_posts": len(te), "items": len(ids), "items_nonzero": int((np.abs(beta[:-1]) > 1e-6).sum()),
        "length_cap": cap, "ridge": lam, "shrink": round(shrink, 4),
        "cv_pairwise": round(float(scores[(cap, lam)].mean()), 4),
        "test_pairwise": round(float(per.mean()), 4), "test_pairwise_ci95": bootstrap_ci(per)["ci95"],
        "test_author_spearman": ev["author_spearman"],
        "dashboard_test_pairwise": round(float(dash.mean()), 4), "vs_dashboard": bootstrap_ci(per, dash),
        "test_calibration_slope": round(float(slope["beta"][0]), 3),
        "pooled_calibration": {"a": round(float(cal_a), 4), "b": round(float(cal_b), 4)},
        "test_pooled_slope": round(float(np.polyfit(s_te, yt, 1)[0]), 3),
        "test_calibration_slope_ci95": [round(float(slope["beta"][0] + z * slope["se"][0]), 3) for z in (-1.96, 1.96)],
    }
    print(json.dumps(metrics, indent=1))

    # 4. calibration bands: cut points from train out-of-fold scores, outcomes from test authors
    cuts = [float(q) for q in np.quantile(s_tr, BAND_QUANTILES)]
    te = te.assign(score=s_te, band=np.digitize(s_te, cuts))
    bands = []
    for b, name in enumerate(BAND_NAMES):  # lift, comments, reposts are already relative to the author's usual post
        g = te[te["band"] == b]
        lo, hi = author_boot(g["lift"], g["author"])
        bands.append({"name": name, "lo": cuts[b - 1] if b else None, "hi": cuts[b] if b < len(cuts) else None,
                      "posts": len(g), "share": len(g) / len(te),
                      "predicted": float(np.exp(cal_a + cal_b * g["score"].mean())),
                      "actual": float(np.exp(g["lift"].mean())), "actual_lo": float(np.exp(lo)), "actual_hi": float(np.exp(hi)),
                      "median_ratio": float(((g["reactions"] + 2) / (g["baseline"] + 2)).median()),
                      "above_usual": float((g["lift"] > 0).mean()), "hit_rate": float(g["hit"].mean()),
                      "flop_rate": float(g["flop"].mean()), "comments": float(np.exp(g["comments"].mean())),
                      "reposts": float(np.exp(g["reposts"].mean()))})
    print(pd.DataFrame(bands).drop(columns=["lo", "hi"]).round(3).to_string(index=False))

    # 5. score gap between two posts by the same test author -> how often the higher score won
    tp = Pairs(yt, at)
    gap = np.abs(s_te[tp.i] - s_te[tp.j])
    right = (np.sign(s_te[tp.i] - s_te[tp.j]) == tp.sy).astype(float)
    edges = [0.1, 0.25, 0.5, 0.8, 1.2]
    pair_gap = []
    for k in range(len(edges) + 1):
        m = (gap >= (edges[k - 1] if k else 0)) & (gap < (edges[k] if k < len(edges) else np.inf))
        pair_gap.append({"lo": edges[k - 1] if k else 0.0, "hi": edges[k] if k < len(edges) else None,
                         "pairs": int(m.sum()), "share": float(m.mean()), "accuracy": float(right[m].mean())})
    metrics["max_gap_seen"] = round(float(gap.max()), 3)  # largest same-author score gap among held-out pairs
    print(pd.DataFrame(pair_gap).round(3).to_string(index=False))

    # 6. evidence per item and per driver
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id").set_index("id")
    out = pd.read_csv(RESULTS / "outcomes.csv")
    oc = {o: out[out["outcome"] == o].set_index("id") for o in ["comments", "reposts", "hit", "flop"]}
    rows = []
    for k, (i, label, kind, d) in enumerate(items):
        e = eff.loc[i]
        rows.append({"id": i, "label": label, "question": e["question"] if not i.startswith("text.") else label, "kind": kind,
                     "driver": d, "weight": float(beta[k]), "mean": float(means[k]), "folds_nonzero": float(stable[k]),
                     "share": float(e["share"]),
                     "single": {"mult": float(e["mult"]), "lo": float(e["lo"]), "hi": float(e["hi"]),
                                "test_mult": float(e["test_mult"]), "win_rate": float(e["win_rate"]), "status": e["status"],
                                "comments_mult": float(oc["comments"].loc[i, "mult"]),
                                "comments_confirmed": bool(oc["comments"].loc[i, "confirmed"]),
                                "reposts_mult": float(oc["reposts"].loc[i, "mult"]),
                                "reposts_confirmed": bool(oc["reposts"].loc[i, "confirmed"]),
                                "hit_points": float(oc["hit"].loc[i, "points"]), "flop_points": float(oc["flop"].loc[i, "points"])}})
    contrib = (X[:, :-1] - means[:-1]) * beta[:-1]
    drivers = []
    for d, v in spec["drivers"].items():
        cols = [k for k, it in enumerate(items) if it[3] == d]
        c = contrib[:, cols].sum(axis=1)
        any_yes = (tr[[ids[k] for k in cols]] >= 0.5).any(axis=1).mean()
        drivers.append({"id": d, "name": v["name"], "direction": v["direction"], "summary": v["summary"], "advice": v["advice"],
                        "items": len(cols), "items_nonzero": int((np.abs(beta[cols]) > 1e-6).sum()),
                        "share_any": float(any_yes), "swing": float(np.exp(np.quantile(c, 0.9) - np.quantile(c, 0.1))),
                        "low": float(np.exp(np.quantile(c, 0.1))), "high": float(np.exp(np.quantile(c, 0.9)))})
    lc = (length_col(tr["words"], cap) - means[-1]) * beta[-1]
    length = {"weight": float(beta[-1]), "cap": cap, "mean": float(means[-1]),
              "low": float(np.exp(np.quantile(lc, 0.1))), "high": float(np.exp(np.quantile(lc, 0.9)))}
    print(pd.DataFrame(drivers)[["name", "direction", "items", "items_nonzero", "share_any", "low", "high"]].round(2).to_string(index=False))
    print("length:", {k: round(v, 3) if isinstance(v, float) else v for k, v in length.items()})

    flat = pd.DataFrame([{**{k: v for k, v in r.items() if k != "single"}, **r["single"]} for r in rows])
    flat.sort_values("weight", key=np.abs, ascending=False).round(4).to_csv(RESULTS / "scorecard_items.csv", index=False)
    stats = {"metrics": metrics, "bands": bands, "pair_gap": pair_gap, "drivers": drivers, "length": length,
             "cv_grid": [{"cap": k[0], "ridge": k[1], "cv": round(float(s.mean()), 4)} for k, s in scores.items()]}
    (RESULTS / "scorecard.json").write_text(json.dumps(stats, indent=1))

    # 7. the bundle for the dashboard: adds what a new post needs (Jev question definitions, post-type centres)
    arche = json.loads((RESULTS / "archetypes.json").read_text())
    at_table = pd.read_csv(RESULTS / "archetypes.csv").set_index("type")
    levers = pd.read_csv(RESULTS / "archetype_levers.csv")
    labels = {i[0]: i[1] for i in items}
    types = {"questions": arche["questions"], "centres": arche["centres"], "table": {}}
    for name, r in at_table.iterrows():
        lv = levers[(levers["type"] == name) & levers["significant"] & levers["id"].isin(labels)]
        lv = lv.assign(s=np.abs(np.log(lv["mult"]))).sort_values("s", ascending=False)
        types["table"][name] = {
            "description": r["description"], "share": float(r["share"]), "reactions": float(r["lift"]),
            "reactions_lo": float(r["lift_lo"]), "reactions_hi": float(r["lift_hi"]), "comments": float(r["comments"]),
            "reposts": float(r["reposts"]), "hit_rate": float(r["hit_rate"]), "flop_rate": float(r["flop_rate"]),
            "levers": [{"id": x["id"], "label": labels[x["id"]], "mult": float(x["mult"]), "lo": float(x["lo"]),
                        "hi": float(x["hi"]), "test_mult": float(x["test_mult"]), "share_in_type": float(x["share_in_type"])}
                       for _, x in lv.iterrows()]}
    bank = {f["id"]: f for f in json.loads((PROCESSED / "features" / "feature_bank_general.json").read_text())["features"]}
    for r in (1, 2):
        bank.update({f["id"]: f for f in json.loads((RESEARCH_DATA / f"bank_round{r}.json").read_text())["features"]})
    need = [q for q in dict.fromkeys(arche["questions"] + ids) if not q.startswith("text.")]
    features = [{k: bank[q][k] for k in ("id", "question", "definition", "yes_example", "no_example")} for q in need]
    bundle = {"calibration": metrics["pooled_calibration"], "items": rows, "drivers": [{k: d[k] for k in ("id", "name", "direction", "summary", "advice")} for d in drivers],
              "length": length, "bands": bands, "pair_gap": pair_gap, "types": types, "features": features, "metrics": metrics}
    (RESEARCH_DATA / "scorecard.json").write_text(json.dumps(bundle, indent=1))
    print(f"bundle: {len(features)} Jev questions per post (about ${len(features) * 6.3e-6:.4f})")


if __name__ == "__main__":
    main()
