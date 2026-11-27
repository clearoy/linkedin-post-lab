"""Fit, calibrate and test the scorecards that the dashboard uses to evaluate a post, at four sizes.

Usage:
    uv run python research/scorecard.py

Model. Every property that passed discovery on train authors (research/drivers.json gives each
one a label and a direction) gets its own weight on Jev's P(yes), fitted jointly within author,
with the direction fixed in advance: a helpful property can only add, a harmful one can only
subtract (bounded least squares, small ridge penalty). Length enters as log(min(words, cap)).

    score = sum over questions of weight * (answer - average post's answer)
            + length weight * (log(min(words, cap)) - average)

Sizes. The full fit leaves some properties at exactly 0 (they add nothing beyond the others);
the rest form the full scorecard. The 20 and 40-question scorecards keep the questions that move
the full score most (|weight| x spread of the answers on train posts) and are refitted on those
alone. The 5 and 10-question scorecards are the best combination found by search on train authors,
using only common questions (present in at least 10% of train posts, so the score applies to most
posts rather than resting on a rare one): greedy forward selection by out-of-fold pairwise accuracy,
then swaps (replace one question with one outside the set) while any swap raises it. The ranked 10
is also fitted, for comparison only.

Everything is chosen on train authors: the ridge strength and the length cap by 5-fold author
CV, the ranking of questions, and for each size a shrink factor so that out-of-fold predictions
are calibrated (slope 1). Test authors are used once per size, to report accuracy, the
calibration bands and the score-gap table.

Writes
  data/processed/research/scorecard.json   the bundle the dashboard loads (has Jev question definitions)
  research/results/scorecard.json          accuracy and calibration per size (statistics only)
  research/results/scorecard_items.csv     every question's weight at each size, with its single-property evidence
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

from linkedin_analysis.model import bootstrap_ci, cv_splits, pairwise_by_author  # noqa: E402

HERE = Path(__file__).resolve().parent
# the 150 generation authors were scored on the new questions afterwards (judgments_r*_generate), so the
# scorecard trains on all 667 train authors; test authors are unchanged
DIRS = BASE_DIRS + [RESEARCH_DATA / "judgments_r1_generate", RESEARCH_DATA / "judgments_r2_generate"]
CAPS = [None, 100, 120, 160, 200]
RIDGE = [0.0, 10.0, 30.0]
SEARCHED_SIZES = [5, 10]  # best combination found by search on train authors
COMMON_SHARE = 0.10  # the searched sizes use only questions present (P(yes) >= 0.5) in at least this share of train posts
SMALL_SIZES = [20, 40]  # top questions of the full fit, refitted; plus the full scorecard (every non-zero weight)
BAND_QUANTILES = [0.10, 0.30, 0.70, 0.90]
BAND_NAMES = ["Low", "Below average", "Average", "Above average", "High"]
GAP_EDGES = [0.1, 0.25, 0.5, 0.8, 1.2]


def length_col(words: pd.Series, cap: int | None) -> np.ndarray:
    return np.log(np.minimum(words, cap) if cap else words).to_numpy(float)


def bounded_fit(Xd: np.ndarray, yd: np.ndarray, sign: np.ndarray, lam: float) -> np.ndarray:
    """Least squares with each weight held to its sign (+1: >= 0, -1: <= 0) and a ridge penalty."""
    lo, hi = np.where(sign > 0, 0.0, -np.inf), np.where(sign > 0, np.inf, 0.0)
    if lam > 0:
        Xd = np.vstack([Xd, np.sqrt(lam) * np.eye(Xd.shape[1])])
        yd = np.r_[yd, np.zeros(Xd.shape[1])]
    return lsq_linear(Xd, yd, bounds=(lo, hi)).x


def oof_predictions(X, y, authors, folds, sign, lam) -> np.ndarray:
    """Out-of-fold scores, each centred on its own fit's average post (as the dashboard scores a draft)."""
    Xd, yd = demean(X, authors), demean(y, authors)
    out = np.zeros(len(y))
    for fit, val in cv_splits(folds):
        b = bounded_fit(Xd[fit], yd[fit], sign, lam)
        out[val] = (X[val] - X[fit].mean(axis=0)) @ b
    return out


def search(k, tr, ids, sign, cap, lam, start=None) -> list[int]:
    """Best k questions by out-of-fold pairwise accuracy on train authors: greedy forward, then best swaps."""
    A, L = tr[ids].to_numpy(float), length_col(tr["words"], cap)
    y, au, folds = tr["lift"].to_numpy(), tr["author"].to_numpy(), tr["fold"].to_numpy()
    pairs = Pairs(y, au)

    def cv(cols):
        s = np.r_[sign[cols], 1.0]
        return float(pairs.per_author(oof_predictions(np.column_stack([A[:, cols], L]), y, au, folds, s, lam)).mean())

    chosen = list(start or [])
    while len(chosen) < k:
        c = max((c for c in range(len(ids)) if c not in chosen), key=lambda c: cv(chosen + [c]))
        chosen.append(c)
    cur = cv(chosen)
    print(f"   forward {k}: cv {cur:.4f}", flush=True)
    for _ in range(10):  # swap passes: take the best improving swap, stop when none helps
        best = (cur, None)
        for pos in range(k):
            for c in range(len(ids)):
                if c in chosen:
                    continue
                v = cv(chosen[:pos] + [c] + chosen[pos + 1:])
                if v > best[0] + 1e-4:
                    best = (v, (pos, c))
        if best[1] is None:
            break
        pos, c = best[1]
        print(f"   swap {ids[chosen[pos]]} -> {ids[c]}: cv {best[0]:.4f}", flush=True)
        chosen[pos], cur = c, best[0]
    return chosen


def fit_size(cols, tr, te, ids, sign, cap, lam) -> dict:
    """Fit one scorecard on train authors (questions `cols` + length), then score test authors once."""
    names = [ids[k] for k in cols]
    X = np.column_stack([tr[names].to_numpy(float), length_col(tr["words"], cap)])
    Xt = np.column_stack([te[names].to_numpy(float), length_col(te["words"], cap)])
    s = np.r_[sign[cols], 1.0]
    y, au, folds = tr["lift"].to_numpy(), tr["author"].to_numpy(), tr["fold"].to_numpy()
    oof = oof_predictions(X, y, au, folds, s, lam)
    shrink = float(within_ols(y, oof, au)["beta"][0])
    beta = bounded_fit(demean(X, au), demean(y, au), s, lam) * shrink
    means = X.mean(axis=0)
    # one post by an unknown author: pooled fit of lift on the out-of-fold score. Within one author a score
    # difference of d means about exp(d) (slope 1 by construction); across authors part of the score is the
    # author's habitual style, which lift (already author-relative) removes.
    s_tr = oof * shrink
    cal_b, cal_a = np.polyfit(s_tr, y, 1)

    s_te = (Xt - means) @ beta
    yt, at = te["lift"].to_numpy(), te["author"].to_numpy()
    per = pairwise_by_author(yt, s_te, at)
    slope = within_ols(yt, s_te, at)
    metrics = {
        "questions": len(cols), "jev_questions": sum(not n.startswith("text.") for n in names),
        "cv_pairwise": round(float(Pairs(y, au).per_author(oof).mean()), 4), "shrink": round(shrink, 4),
        "test_pairwise": round(float(per.mean()), 4), "test_pairwise_ci95": bootstrap_ci(per)["ci95"],
        "test_calibration_slope": round(float(slope["beta"][0]), 3),
        "test_calibration_slope_ci95": [round(float(slope["beta"][0] + z * slope["se"][0]), 3) for z in (-1.96, 1.96)],
        "pooled_calibration": {"a": round(float(cal_a), 4), "b": round(float(cal_b), 4)},
    }

    # calibration bands: cut points from train out-of-fold scores, outcomes from test authors
    cuts = [float(q) for q in np.quantile(s_tr, BAND_QUANTILES)]
    tb = te.assign(score=s_te, band=np.digitize(s_te, cuts))
    bands = []
    for b, name in enumerate(BAND_NAMES):  # lift is already relative to the author's usual post
        g = tb[tb["band"] == b]
        lo, hi = author_boot(g["lift"], g["author"])
        bands.append({"name": name, "lo": cuts[b - 1] if b else None, "hi": cuts[b] if b < len(cuts) else None,
                      "posts": len(g), "share": len(g) / len(tb),
                      "predicted": float(np.exp(cal_a + cal_b * g["score"].mean())),
                      "actual": float(np.exp(g["lift"].mean())), "actual_lo": float(np.exp(lo)), "actual_hi": float(np.exp(hi)),
                      "above_usual": float((g["lift"] > 0).mean()), "hit_rate": float(g["hit"].mean()),
                      "flop_rate": float(g["flop"].mean())})

    # score gap between two posts by the same test author -> how often the higher score won
    tp = Pairs(yt, at)
    gap = np.abs(s_te[tp.i] - s_te[tp.j])
    right = (np.sign(s_te[tp.i] - s_te[tp.j]) == tp.sy).astype(float)
    pair_gap = []
    for k in range(len(GAP_EDGES) + 1):
        m = (gap >= (GAP_EDGES[k - 1] if k else 0)) & (gap < (GAP_EDGES[k] if k < len(GAP_EDGES) else np.inf))
        pair_gap.append({"lo": GAP_EDGES[k - 1] if k else 0.0, "hi": GAP_EDGES[k] if k < len(GAP_EDGES) else None,
                         "pairs": int(m.sum()), "share": float(m.mean()), "accuracy": float(right[m].mean()) if m.any() else None})
    metrics["max_gap_seen"] = round(float(gap.max()), 3)  # largest same-author score gap among held-out pairs
    return {"names": names, "beta": beta, "means": means, "metrics": metrics, "bands": bands, "pair_gap": pair_gap,
            "per_author": per}


def main() -> None:
    spec = json.loads((HERE / "drivers.json").read_text())
    df = add_outcomes(frame(DIRS))
    items = [(m[0], m[1]) for v in spec["drivers"].values() for m in v["members"]]
    ids, labels = [i[0] for i in items], dict(items)
    sign = np.array([1.0 if v["direction"] == "+" else -1.0 for v in spec["drivers"].values() for _ in v["members"]])

    full = df[df[ids].notna().all(axis=1)].reset_index(drop=True)
    tr = full[full["split"] == "train"].reset_index(drop=True)
    te = full[full["split"] == "test"].reset_index(drop=True)
    assert len(te) == (df["split"] == "test").sum()
    y, au, folds = tr["lift"].to_numpy(), tr["author"].to_numpy(), tr["fold"].to_numpy()
    print(f"train {tr['author'].nunique()} authors / {len(tr)} posts; test {te['author'].nunique()} / {len(te)}; {len(ids)} properties")

    # 1. choose length cap and ridge strength by author CV on train, with every property
    sign_l = np.r_[sign, 1.0]
    pairs, scores = Pairs(y, au), {}
    for cap in CAPS:
        X = np.column_stack([tr[ids].to_numpy(float), length_col(tr["words"], cap)])
        for lam in RIDGE:
            scores[(cap, lam)] = pairs.per_author(oof_predictions(X, y, au, folds, sign_l, lam))
    best = max(scores, key=lambda k: scores[k].mean())
    close = [k for k, s in scores.items()
             if (s - scores[best]).mean() >= -(s - scores[best]).std(ddof=1) / np.sqrt(len(s)) or k == best]
    cap, lam = min(close, key=lambda k: (k[0] or 10**9, -k[1]))  # tie-break: capped length, then more shrinkage
    for k, s in scores.items():
        print(f"   cap {str(k[0]):>4}  ridge {k[1]:>4}  cv {s.mean():.4f}{'  <- best' if k == best else ''}{'  <- chosen' if k == (cap, lam) else ''}")

    # 2. rank properties by how much they move the full score on train posts; zero weights drop out
    X = np.column_stack([tr[ids].to_numpy(float), length_col(tr["words"], cap)])
    b_all = bounded_fit(demean(X, au), demean(y, au), sign_l, lam)
    importance = np.abs(b_all[:-1]) * tr[ids].to_numpy(float).std(axis=0)
    order = np.argsort(-importance)
    n_full = int((np.abs(b_all[:-1]) > 1e-6).sum())
    sizes = [k for k in SMALL_SIZES if k < n_full] + [n_full]
    print(f"{n_full} of {len(ids)} properties have a non-zero weight; sizes {SEARCHED_SIZES + sizes}")

    # 3. fit and test each size. Searched sizes: the best combination on train authors (the 10 starts from the best 5).
    # Ranked sizes: if refitting a small set leaves a question at exactly 0 (the others cover it), swap it for the
    # next question in the ranking, so every size has that many questions that count.
    share = (tr[ids] >= 0.5).mean().to_numpy()
    common = [j for j in range(len(ids)) if share[j] >= COMMON_SHARE]
    print(f"{len(common)} of {len(ids)} properties are in at least {COMMON_SHARE:.0%} of train posts (searched sizes use only these)")
    fits, found = {}, []
    for k in SEARCHED_SIZES:
        local = search(k, tr, [ids[j] for j in common], sign[common], cap, lam,
                       start=[common.index(j) for j in found])
        found = [common[j] for j in local]
        fits[k] = fit_size(found, tr, te, ids, sign, cap, lam)
        fits[k]["metrics"]["selection"] = f"best combination of common questions (in at least {COMMON_SHARE:.0%} of train posts) searched on train authors"
        fits[k]["metrics"]["common_share"] = COMMON_SHARE
    for k in sizes + ["10 ranked"]:
        k_n = 10 if k == "10 ranked" else k
        cols, nxt = list(order[:k_n]), k_n
        while True:
            f = fit_size(cols, tr, te, ids, sign, cap, lam)
            zero = [c for c, w in zip(cols, f["beta"][:-1]) if abs(w) < 1e-6]
            if not zero or nxt >= n_full:
                break
            cols = [c for c in cols if c not in zero] + list(order[nxt:nxt + len(zero)])
            nxt += len(zero)
        f["metrics"]["selection"] = "all non-zero weights" if k == n_full else "top questions of the full fit, refitted"
        fits[k] = f
    ref = fits[n_full]["per_author"]
    for k, f in fits.items():
        f["metrics"]["vs_full"] = bootstrap_ci(f["per_author"], ref)
        m = f["metrics"]
        print(f"size {k!s:>9}: jev {m['jev_questions']:>3}  cv {m['cv_pairwise']:.4f}  test {m['test_pairwise']:.4f} "
              f"{m['test_pairwise_ci95']}  vs full {m['vs_full']['mean']:+.4f} {m['vs_full']['ci95']}  slope {m['test_calibration_slope']}")
        if k in SEARCHED_SIZES:
            print("           " + ", ".join(f["names"]))
    ranked10 = fits.pop("10 ranked")
    fits[10]["metrics"]["vs_ranked_10"] = bootstrap_ci(fits[10]["per_author"], ranked10["per_author"])
    fits[10]["metrics"]["ranked_10_test_pairwise"] = ranked10["metrics"]["test_pairwise"]
    fits = dict(sorted(fits.items()))

    # 4. outputs
    eff = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]).drop_duplicates("id").set_index("id")
    bank = {f["id"]: f for f in json.loads((PROCESSED / "features" / "feature_bank_general.json").read_text())["features"]}
    for r in (1, 2):
        bank.update({f["id"]: f for f in json.loads((RESEARCH_DATA / f"bank_round{r}.json").read_text())["features"]})
    question = {i: (bank[i]["question"] if i in bank else labels[i]) for i in ids}

    used = {n for f in fits.values() for n in f["names"]}
    rank = {ids[k]: n + 1 for n, k in enumerate(order) if np.abs(b_all[k]) > 1e-6 or ids[k] in used}
    rows = []
    for i in sorted(rank, key=rank.get):
        e = eff.loc[i]
        row = {"rank": rank[i], "id": i, "label": labels[i], "question": question[i], "share": float(e["share"]),
               "alone_mult": float(e["mult"]), "alone_test_mult": float(e["test_mult"]), "status": e["status"]}
        for k, f in fits.items():
            row[f"weight_{k}"] = float(f["beta"][f["names"].index(i)]) if i in f["names"] else None
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "scorecard_items.csv", index=False)

    def size_bundle(f: dict) -> dict:
        return {"metrics": f["metrics"], "calibration": f["metrics"]["pooled_calibration"], "bands": f["bands"],
                "pair_gap": f["pair_gap"],
                "items": [{"id": n, "label": labels[n], "question": question[n], "weight": float(f["beta"][k]),
                           "mean": float(f["means"][k])} for k, n in enumerate(f["names"])],
                "length": {"weight": float(f["beta"][-1]), "cap": cap, "mean": float(f["means"][-1])}}

    common = {"train_authors": int(tr["author"].nunique()), "train_posts": len(tr), "test_authors": int(te["author"].nunique()),
              "test_posts": len(te), "properties_tested": len(ids), "length_cap": cap, "ridge": lam}
    stats = {"common": common, "cv_grid": [{"cap": k[0], "ridge": k[1], "cv": round(float(s.mean()), 4)} for k, s in scores.items()],
             "sizes": {str(k): {"metrics": f["metrics"], "bands": f["bands"], "pair_gap": f["pair_gap"]} for k, f in fits.items()}}
    (RESULTS / "scorecard.json").write_text(json.dumps(stats, indent=1))

    need = [n for n in ids if n in used and not n.startswith("text.")]
    features = [{k: bank[q][k] for k in ("id", "question", "definition", "yes_example", "no_example")} for q in need]
    bundle = {"common": common, "default": str(n_full), "sizes": {str(k): size_bundle(f) for k, f in fits.items()},
              "features": features}
    (RESEARCH_DATA / "scorecard.json").write_text(json.dumps(bundle, indent=1))
    print(f"bundle: up to {len(features)} Jev questions per post (about ${len(features) * 6.3e-6:.4f})")


if __name__ == "__main__":
    main()
