"""Two questions about prediction, scored on the same held-out test authors as compare_dashboard.py.

Usage:
    uv run python research/subset.py

1. Does combining the old and the new help? Ridge on old + text + new questions,
   and equal-weight blends of the dashboard model with each research model
   (each prediction scaled by its train standard deviation; no weights tuned).
2. Can a small subset do as well? Forward selection on train authors only:
   at each step add the question that most raises out-of-fold pairwise
   accuracy (5 author folds) of a within-author linear model. The subset size
   is the smallest whose out-of-fold accuracy is within one standard error
   (over train authors, paired) of the best size. Test authors are used once,
   to score the chosen subset. Two candidate pools:
     * old + text + new questions (517 train authors, because the 150
       generation authors were not scored on the new questions);
     * old + text only (all 667 train authors, the same data as the dashboard).

Writes research/results/subset.json and research/results/subset_items.csv.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from compare_dashboard import DIRS  # noqa: E402
from joint import frame  # noqa: E402
from stats import PROCESSED, RESULTS, YES, demean, within_ols  # noqa: E402

from linkedin_analysis.model import bootstrap_ci, cv_splits, pairwise_by_author  # noqa: E402

MAX_K = 30
MIN_GAP = 0.1  # same as the dashboard metric


class Pairs:
    """Same-author post pairs whose lift differs by >= MIN_GAP, for fast per-author pairwise accuracy."""

    def __init__(self, y: np.ndarray, authors: np.ndarray):
        I, J, A = [], [], []
        codes, self.names = pd.factorize(authors)
        for a in range(len(self.names)):
            i = np.where(codes == a)[0]
            ii, jj = np.triu_indices(len(i), k=1)
            keep = np.abs(y[i[ii]] - y[i[jj]]) >= MIN_GAP
            I.append(i[ii][keep]), J.append(i[jj][keep]), A.append(np.full(keep.sum(), a))
        self.i, self.j, self.a = np.concatenate(I), np.concatenate(J), np.concatenate(A)
        self.sy = np.sign(y[self.i] - y[self.j])
        self.n = np.bincount(self.a, minlength=len(self.names))
        self.has = self.n > 0

    def per_author(self, pred: np.ndarray) -> np.ndarray:
        dp = pred[self.i] - pred[self.j]
        hit = (np.sign(dp) == self.sy) * 1.0 + (dp == 0) * 0.5
        return np.bincount(self.a, weights=hit, minlength=len(self.names))[self.has] / self.n[self.has]


def oof(Xd: np.ndarray, yd: np.ndarray, X: np.ndarray, splits) -> np.ndarray:
    """Out-of-fold predictions of a within-author linear model (Xd, yd demeaned by author)."""
    out = np.zeros(len(X))
    for fit, val in splits:
        beta = np.linalg.lstsq(Xd[fit], yd[fit], rcond=None)[0]
        out[val] = X[val] @ beta
    return out


def forward(tr: pd.DataFrame, pool: list[str]) -> tuple[list[str], list[np.ndarray]]:
    """Greedy forward selection by out-of-fold pairwise accuracy. Returns the order and per-author scores."""
    y, au = tr["lift"].to_numpy(), tr["author"].to_numpy()
    X = tr[pool].to_numpy(float)
    Xd, yd = demean(X, au), demean(y, au)
    splits, pairs = cv_splits(tr["fold"].to_numpy()), Pairs(y, au)
    chosen, scores = [], []
    for step in range(MAX_K):
        best, best_s = None, None
        for c in range(len(pool)):
            if c in chosen:
                continue
            s = pairs.per_author(oof(Xd[:, chosen + [c]], yd, X[:, chosen + [c]], splits))
            if best_s is None or s.mean() > best_s.mean():
                best, best_s = c, s
        chosen.append(best), scores.append(best_s)
        print(f"   {step + 1:2d}  cv {best_s.mean():.4f}  + {pool[best]}", flush=True)
    return [pool[c] for c in chosen], scores


def pick_k(scores: list[np.ndarray]) -> int:
    """Smallest size within one paired standard error (over train authors) of the best size."""
    best = int(np.argmax([s.mean() for s in scores]))
    for k, s in enumerate(scores):
        d = s - scores[best]
        if d.mean() >= -d.std(ddof=1) / np.sqrt(len(d)):
            return k + 1
    return best + 1


def main() -> None:
    df = frame(DIRS)
    df["length.log_words"] = np.log1p(df["words"])
    bundle = json.loads((PROCESSED / "models" / "model_bundle.json").read_text())
    df["dashboard"] = bundle["linear"]["intercept"] + sum(
        c * df[bundle["themes"][t]["members"]].mean(axis=1) for t, c in bundle["linear"]["coef"].items())

    old_q = [c for c in pd.read_parquet(DIRS[0] / "features.parquet").columns if c != "urn"]
    old_q = [c for c in old_q if c in df.columns and df[c].notna().all()]
    text = [c for c in df.columns if c.startswith("text.")] + ["length.log_words"]
    new_q = [c for c in df.columns if c.startswith(("r1_", "r2_"))]
    n_answered = df[new_q].notna().sum()
    new_q = [c for c in new_q if n_answered[c] == n_answered.max()]
    full = df[df[new_q].notna().all(axis=1)].reset_index(drop=True)  # 517 train + all test authors

    te = df[df["split"] == "test"].reset_index(drop=True)
    assert len(te) == (full["split"] == "test").sum()  # every test post has every answer
    te = full[full["split"] == "test"].reset_index(drop=True)
    yt, at = te["lift"].to_numpy(), te["author"].to_numpy()
    dash = pairwise_by_author(yt, te["dashboard"].to_numpy(), at)
    results, preds = {}, {"dashboard": te["dashboard"].to_numpy()}

    def score(name: str, pred: np.ndarray, extra: dict | None = None) -> None:
        per = pairwise_by_author(yt, pred, at)
        results[name] = {"pairwise": round(float(per.mean()), 4), "ci95": bootstrap_ci(per)["ci95"],
                         "vs_dashboard": bootstrap_ci(per, dash), **(extra or {})}
        d = results[name]["vs_dashboard"]
        print(f"{per.mean():.4f}  vs dashboard {d['mean']:+.4f} {d['ci95']}  {name}", flush=True)

    print(f"test: {te['author'].nunique()} authors / {len(te)} posts; dashboard {dash.mean():.4f}")

    # 1. combinations
    trn = full[full["split"] == "train"]
    sd = lambda p_tr, p_te: p_te / p_tr.std()  # noqa: E731

    def ridge(tr_df, qs):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 16), cv=cv_splits(tr_df["fold"].to_numpy())))
        m.fit(tr_df[qs].to_numpy(float), tr_df["lift"].to_numpy())
        return m.predict(tr_df[qs].to_numpy(float)), m.predict(te[qs].to_numpy(float))

    r_tr, r_te = ridge(trn, old_q + text + new_q)
    score(f"ridge, old + text + new ({len(old_q + text + new_q)})", r_te)
    jt = pd.read_csv(RESULTS / "joint_r2_train.csv")
    cols = [c for c in jt["id"] if c != "log_words"]
    Z = lambda d: np.column_stack([(d[cols] >= YES).to_numpy(float), np.log1p(d["words"].to_numpy())])  # noqa: E731
    beta = np.nan_to_num(within_ols(trn["lift"].to_numpy(), Z(trn), trn["author"].to_numpy())["beta"])
    j_tr, j_te = Z(trn) @ beta, Z(te) @ beta
    d_tr = trn["dashboard"].to_numpy()
    score("blend: dashboard + research joint model", sd(d_tr, preds["dashboard"]) + sd(j_tr, j_te))
    score("blend: dashboard + ridge old + text + new", sd(d_tr, preds["dashboard"]) + sd(r_tr, r_te))

    # 2. small subsets chosen on train
    q_text = pd.concat([pd.read_csv(RESULTS / f"effects_{n}.csv") for n in ["existing", "r1", "r2"]]) \
        .drop_duplicates("id").set_index("id")["question"].to_dict()
    q_text["length.log_words"] = "Post length (log of word count)"
    items = []
    for label, data, pool in [("old + text + new", full, old_q + text + new_q),
                              ("old + text only", df[df["split"] == "train"], old_q + text)]:
        tr = data[data["split"] == "train"].reset_index(drop=True)
        print(f"\nforward selection, pool {label}: {len(pool)} candidates, {tr['author'].nunique()} train authors")
        order, cv = forward(tr, pool)
        k = pick_k(cv)
        Xtr = tr[order].to_numpy(float)
        curve = {}
        for kk in sorted({3, 5, 10, 15, 20, 30, k}):
            b = within_ols(tr["lift"].to_numpy(), Xtr[:, :kk], tr["author"].to_numpy())["beta"]
            p = te[order[:kk]].to_numpy(float) @ b
            per = pairwise_by_author(yt, p, at)
            curve[kk] = (round(float(cv[kk - 1].mean()), 4), round(float(per.mean()), 4), bootstrap_ci(per, dash)["ci95"])
            if kk == k:
                chosen_pred, chosen_beta = p, b
        name = f"small subset from {label}: {k} questions (size chosen on train)"
        score(name, chosen_pred, {"k": k, "train_authors": int(tr["author"].nunique()),
                                  "cv_and_test_by_size": {str(a): {"cv": c, "test": t, "vs_dashboard_ci95": d}
                                                          for a, (c, t, d) in curve.items()}})
        b_tr = Xtr[:, :k] @ chosen_beta
        score(f"blend: dashboard + {label} subset",
              sd(tr["dashboard"].to_numpy(), preds["dashboard"]) + sd(b_tr, chosen_pred))
        for a, (c, t, d) in curve.items():
            print(f"   {a:2d} questions: cv {c:.4f}  test {t:.4f}  vs dashboard {t - dash.mean():+.4f} {d}")
        for rank, (c, b) in enumerate(zip(order[:k], chosen_beta), 1):
            items.append({"pool": label, "rank": rank, "id": c, "question": q_text.get(c, c),
                          "multiplier_yes_vs_no": round(float(np.exp(b)), 3),
                          "share_yes": round(float((tr[c] >= YES).mean()), 3) if c != "length.log_words" else None,
                          "cv_after_adding": round(float(cv[rank - 1].mean()), 4)})
    pd.DataFrame(items).to_csv(RESULTS / "subset_items.csv", index=False)
    (RESULTS / "subset.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
