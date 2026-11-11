"""Train the controls baseline, a sparse linear model and RuleFit on train authors; evaluate on test authors.

The linear model is a relaxed lasso by default: at most --linear-max-terms
content features (count chosen by CV) plus the controls, refit without
shrinkage. --linear-max-terms 0 gives the plain lasso instead.

Usage:
    uv run python scripts/train_models.py --judgments data/processed/judgments_train data/processed/judgments_test

Needs labels.parquet, splits.parquet, and one or more judgments folders
(each with features.parquet from scripts/judge_features.py) plus the feature
bank they were scored with.

Writes data/processed/models/:
    model_bundle.json   everything the dashboard needs to score a new post
    metrics.json        CV (train) and test metrics for every model
    linear_coefs.csv    lasso coefficients with fold sign-stability
    rulefit_terms.csv   RuleFit's non-zero rules and linear terms
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # see README "Imports on this Mac"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from linkedin_analysis.data import load_all  # noqa: E402
from linkedin_analysis.model import (  # noqa: E402
    RelaxedLasso,
    RuleFit,
    ScaledLasso,
    bootstrap_ci,
    controls_frame,
    cv_splits,
    evaluate,
    fit_ridge,
    lasso_cv,
    pairwise_by_author,
    relaxed_lasso_cv,
)
from linkedin_analysis.paths import PROCESSED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judgments", type=Path, nargs="+", required=True)
    p.add_argument("--bank", type=Path, default=PROCESSED / "features" / "feature_bank.json")
    p.add_argument("--min-prevalence", type=float, default=0.02,
                   help="drop features whose mean P(yes) in train is below this or above 1 minus this")
    p.add_argument("--out", type=Path, default=PROCESSED / "models")
    p.add_argument("--seed", type=int, default=0)
    # Chosen by train CV (see docs/models.md): linear 0.589 (min) vs 0.519 (1se); RuleFit 0.580 (1se) vs 0.573 (min).
    p.add_argument("--linear-rule", choices=["min", "1se"], default="min",
                   help="lasso penalty: lowest CV error or sparsest within one standard error")
    p.add_argument("--rulefit-rule", choices=["min", "1se"], default="1se")
    p.add_argument("--linear-max-terms", type=int, default=10,
                   help="relaxed lasso with at most this many content features (CV picks how many); "
                        "0 = plain lasso with no cap")
    p.add_argument("--linear-k", type=int, default=None,
                   help="fix the relaxed lasso's feature cap instead of letting CV choose (e.g. 8 for steadier picks)")
    return p.parse_args()


def oof(fit_predict, X, y, folds) -> np.ndarray:
    """Out-of-fold predictions over the train folds."""
    pred = np.zeros(len(y))
    for tr, va in cv_splits(folds):
        pred[va] = fit_predict(X[tr], y[tr], folds[tr], X[va])
    return pred


def main() -> None:
    args = parse_args()
    bank = {f["id"]: f for f in json.loads(args.bank.read_text())["features"]}
    wide = pd.concat([pd.read_parquet(d / "features.parquet") for d in args.judgments]).drop_duplicates("urn")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")[["urn", "author", "lift"]]
    splits = pd.read_parquet(PROCESSED / "splits.parquet")[["author", "split", "fold"]]
    posts = load_all()[["urn", "media", "words", "hour_pt", "weekday"]]
    df = wide.merge(labels, on="urn").merge(splits, on="author").merge(posts, on="urn").reset_index(drop=True)

    feat_cols = [c for c in wide.columns if c != "urn" and c in bank]
    train = df["split"] == "train"
    prevalence = df.loc[train, feat_cols].mean()
    keep = [c for c in feat_cols if args.min_prevalence <= prevalence[c] <= 1 - args.min_prevalence]
    dropped = sorted(set(feat_cols) - set(keep))
    df[keep] = df[keep].fillna(df.loc[train, keep].mean())  # rare refusals

    ctl = controls_frame(df)
    ctl_cols = list(ctl.columns)
    df = pd.concat([df, ctl], axis=1)
    cols = keep + ctl_cols

    tr, te = df[train].reset_index(drop=True), df[~train].reset_index(drop=True)
    y_tr, y_te, folds = tr["lift"].to_numpy(), te["lift"].to_numpy(), tr["fold"].to_numpy()
    print(f"train {tr['author'].nunique()} authors / {len(tr)} posts, test {te['author'].nunique()} / {len(te)}; "
          f"{len(keep)} features kept, {len(dropped)} dropped for prevalence")

    # 1. Controls-only baseline
    Xc_tr, Xc_te = tr[ctl_cols].to_numpy(), te[ctl_cols].to_numpy()
    ridge = fit_ridge(Xc_tr, y_tr)
    oof_ctl = oof(lambda X, y, f, Xv: fit_ridge(X, y).predict(Xv), Xc_tr, y_tr, folds)

    # 2. Linear model on features + controls. coef_full is indexed like `cols`.
    X_tr, X_te = tr[cols].to_numpy(), te[cols].to_numpy()
    if args.linear_max_terms:
        F_tr, C_tr = tr[keep].to_numpy(), tr[ctl_cols].to_numpy()
        rl, lasso_info = relaxed_lasso_cv(F_tr, C_tr, y_tr, folds, tr["author"].to_numpy(),
                                          max(args.linear_max_terms, args.linear_k or 0), args.linear_k)

        def full(m: RelaxedLasso) -> np.ndarray:
            c = np.zeros(len(cols))
            c[m.selected_] = m.feature_coef_
            c[len(keep):] = m.control_coef_
            return c

        coef_full, intercept = full(rl), rl.intercept_
        oof_lin = np.zeros(len(y_tr))
        fold_signs = []
        for a, v in cv_splits(folds):
            m = RelaxedLasso(lasso_info["k"]).fit(F_tr[a], C_tr[a], y_tr[a])
            oof_lin[v] = m.predict(F_tr[v], C_tr[v])
            fold_signs.append(np.sign(full(m)))
        fold_signs = np.array(fold_signs)
        lin_test = rl.predict(te[keep].to_numpy(), te[ctl_cols].to_numpy())
        lasso_info.update(model="relaxed_lasso", nonzero=int((coef_full != 0).sum()))
    else:
        lasso, lasso_info = lasso_cv(X_tr, y_tr, folds, rule=args.linear_rule)
        coef_full, intercept = lasso.coef_, lasso.intercept_
        oof_lin = oof(lambda X, y, f, Xv: ScaledLasso(lasso_info["alpha"]).fit(X, y).predict(Xv), X_tr, y_tr, folds)
        fold_signs = np.array([np.sign(ScaledLasso(lasso_info["alpha"]).fit(X_tr[a], y_tr[a]).coef_)
                               for a, _ in cv_splits(folds)])
        lin_test = lasso.predict(X_te)
        lasso_info["model"] = "lasso"

    # 3. RuleFit
    rf = RuleFit(rule=args.rulefit_rule, random_state=args.seed).fit(tr[cols], y_tr, folds)
    oof_rf = np.zeros(len(y_tr))
    for a, v in cv_splits(folds):
        m = RuleFit(rule=args.rulefit_rule, random_state=args.seed).fit(tr.loc[a, cols], y_tr[a], folds[a], alpha=rf.info_["alpha"])
        oof_rf[v] = m.predict(tr.loc[v, cols])

    test_preds = {"controls": ridge.predict(Xc_te), "linear": lin_test, "rulefit": rf.predict(te[cols])}
    metrics = {
        name: {"cv_train": evaluate(y_tr, o, tr["author"]), "test": evaluate(y_te, test_preds[name], te["author"])}
        for name, o in [("controls", oof_ctl), ("linear", oof_lin), ("rulefit", oof_rf)]
    }
    per_author = {k: pairwise_by_author(y_te, p, te["author"]) for k, p in test_preds.items()}
    for name in metrics:
        metrics[name]["test"]["pairwise_accuracy_ci95"] = bootstrap_ci(per_author[name])["ci95"]
        if name != "controls":
            metrics[name]["test"]["gain_over_controls"] = bootstrap_ci(per_author[name], per_author["controls"])
    metrics["data"] = {"train_authors": int(tr["author"].nunique()), "train_posts": len(tr),
                       "test_authors": int(te["author"].nunique()), "test_posts": len(te),
                       "features_scored": len(feat_cols), "features_used": len(keep),
                       "features_dropped_prevalence": dropped}
    metrics["linear_info"] = lasso_info
    metrics["rulefit_info"] = rf.info_

    # Outputs
    args.out.mkdir(parents=True, exist_ok=True)
    coefs = pd.DataFrame({
        "column": cols, "coef": coef_full,
        "question": [bank[c]["question"] if c in bank else c for c in cols],
        "mean_p_yes_train": [float(tr[c].mean()) for c in cols],
        "sign_stability": [float((fold_signs[:, j] == np.sign(coef_full[j])).mean()) if coef_full[j] else 0.0
                           for j in range(len(cols))],
    })
    coefs["effect_x"] = np.exp(coefs["coef"])  # multiplier on reactions when P(yes) goes 0 -> 1
    coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
    coefs.to_csv(args.out / "linear_coefs.csv", index=False)
    terms = rf.terms(tr[cols])
    terms.to_csv(args.out / "rulefit_terms.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    means = tr[cols].mean()
    bundle = {
        "features": [bank[c] | {"mean_p_yes_train": float(means[c])} for c in keep],
        "column_means": {c: float(means[c]) for c in cols},
        "linear": {"intercept": float(intercept),
                   "coef": {c: float(v) for c, v in zip(cols, coef_full) if v != 0}},
        "rulefit": {
            "intercept": float(rf.lasso_.intercept_),
            "rules": [{"conditions": r.conditions, "coef": float(c), "text": r.text()}
                      for r, c in zip(rf.rules_, rf.lasso_.coef_[: len(rf.rules_)]) if c != 0],
            "linear": {col: [float(c), *rf.scale_[col]]
                       for col, c in zip(rf.linear_cols_, rf.lasso_.coef_[len(rf.rules_):]) if c != 0},
        },
        "metrics": {k: v for k, v in metrics.items() if k in ("controls", "linear", "rulefit", "data")},
    }
    (args.out / "model_bundle.json").write_text(json.dumps(bundle, indent=2))

    print(json.dumps({k: v for k, v in metrics.items() if k in ("controls", "linear", "rulefit")}, indent=2))
    print("\nLinear model terms:")
    print(coefs[coefs["coef"] != 0].head(15)[["column", "coef", "effect_x", "sign_stability"]].to_string(index=False))
    print("\nTop RuleFit terms:")
    print(terms.head(10)[["term", "coef", "support"]].to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
