import numpy as np
import pandas as pd

from linkedin_analysis.model import RelaxedLasso, RuleFit, controls_frame, evaluate, lasso_cv, relaxed_lasso_cv


def synthetic(n_authors: int = 120, per_author: int = 12, seed: int = 0):
    """Lift depends on f0 (linear) and on f1 AND f2 together (a rule)."""
    rng = np.random.default_rng(seed)
    n = n_authors * per_author
    X = pd.DataFrame(rng.uniform(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = 0.8 * X["f0"] + 0.6 * ((X["f1"] > 0.5) & (X["f2"] > 0.5)) + rng.normal(0, 0.2, n)
    authors = np.repeat(np.arange(n_authors), per_author)
    folds = authors % 5
    return X, y.to_numpy(), authors, folds


def test_evaluate_perfect_reversed_and_random():
    y = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    authors = np.array([0, 0, 0, 1, 1, 1])
    assert evaluate(y, y, authors)["pairwise_accuracy"] == 1.0
    assert evaluate(y, -y, authors)["pairwise_accuracy"] == 0.0
    assert evaluate(y, np.zeros(6), authors)["pairwise_accuracy"] == 0.5  # ties count half


def test_evaluate_only_compares_within_author():
    # Author 1 has much higher lift, but within each author the order is right.
    y = np.array([0.0, 0.5, 5.0, 5.5])
    pred = np.array([10.0, 11.0, 0.0, 1.0])
    authors = np.array([0, 0, 1, 1])
    assert evaluate(y, pred, authors)["pairwise_accuracy"] == 1.0


def test_lasso_finds_true_features():
    X, y, authors, folds = synthetic()
    model, info = lasso_cv(X.to_numpy(), y, folds)
    coef = dict(zip(X.columns, model.coef_))
    assert 0.6 < coef["f0"] < 1.0  # raw units: true effect 0.8
    assert all(abs(coef[f]) < 0.1 for f in ["f3", "f4", "f5"])
    assert info["alpha_1se"] >= info["alpha_min"] == info["alpha"]
    np.testing.assert_allclose(model.predict(X.to_numpy()).mean(), y.mean(), atol=0.05)


def test_rulefit_beats_linear_on_interaction_and_names_the_rule():
    X, y, authors, folds = synthetic()
    train, test = folds != 0, folds == 0
    rf = RuleFit(n_trees=80, random_state=0).fit(X[train], y[train], folds[train])
    lin, _ = lasso_cv(X[train].to_numpy(), y[train], folds[train])
    rf_acc = evaluate(y[test], rf.predict(X[test]), authors[test])["pairwise_accuracy"]
    lin_acc = evaluate(y[test], lin.predict(X[test].to_numpy()), authors[test])["pairwise_accuracy"]
    assert rf_acc > lin_acc
    top = rf.terms(X[train]).head(5)["term"].str.cat(sep=" | ")
    assert "f1 >" in top and "f2 >" in top


def test_controls_frame():
    df = pd.DataFrame({"media": ["text", "video"], "words": [10, 100], "hour_pt": [9, 20], "weekday": [1, 6]})
    c = controls_frame(df)
    assert c.loc[1, "ctl.media_video"] == 1 and c.loc[0, "ctl.media_video"] == 0
    assert c.loc[1, "ctl.weekend"] == 1 and c.loc[1, "ctl.hour_evening"] == 1
    assert c.loc[0, ["ctl.hour_night", "ctl.hour_afternoon", "ctl.hour_evening"]].sum() == 0


def test_lasso_max_terms_caps_the_model():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(1200, 15))
    y = X[:, :8] @ np.linspace(1.0, 0.3, 8) + rng.normal(0, 0.3, 1200)
    folds = np.arange(1200) % 5
    free, _ = lasso_cv(X, y, folds)
    capped, info = lasso_cv(X, y, folds, max_terms=3)
    assert (free.coef_ != 0).sum() > 3
    assert 0 < (capped.coef_ != 0).sum() <= 3
    assert set(np.flatnonzero(capped.coef_)) <= {0, 1, 2, 3}  # keeps the strongest true features


def test_relaxed_lasso_keeps_k_features_unshrunk_plus_controls():
    rng = np.random.default_rng(2)
    n = 1500
    F = rng.uniform(size=(n, 12))
    C = rng.normal(size=(n, 2))
    y = 0.9 * F[:, 0] - 0.6 * F[:, 1] + 0.3 * C[:, 0] + rng.normal(0, 0.2, n)
    m = RelaxedLasso(2).fit(F, C, y)
    assert sorted(m.selected_) == [0, 1]
    np.testing.assert_allclose(m.feature_coef_, [0.9, -0.6], atol=0.05)  # no shrinkage
    np.testing.assert_allclose(m.control_coef_[0], 0.3, atol=0.05)
    authors = np.arange(n) // 10
    best, info = relaxed_lasso_cv(F, C, y, authors % 5, authors, max_k=5)
    assert info["k"] >= 2 and len(best.selected_) <= 5
