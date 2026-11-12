"""Does the research version predict better than the dashboard model? Same held-out authors.

Usage:
    uv run python research/compare_dashboard.py

Every model is scored on the same test posts with the dashboard metric
(within-author pairwise accuracy) and compared with the dashboard's theme model
by a paired bootstrap over authors. Research models are fit on train authors
only, with properties chosen on train only (joint_r2_train), never on test.
New questions were not scored for the 150 generation authors, so the models
that use them train on the other 517 train authors; the ridge models are fit
on those same authors so the comparison of question sets is like for like.

Writes research/results/compare_dashboard.json.
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

from joint import frame  # noqa: E402
from stats import PROCESSED, RESEARCH_DATA, RESULTS, YES, within_ols  # noqa: E402

from linkedin_analysis.model import bootstrap_ci, cv_splits, evaluate, pairwise_by_author  # noqa: E402

R = RESEARCH_DATA
DIRS = [PROCESSED / "judgments_full_train", PROCESSED / "judgments_full_test",
        R / "judgments_r1_screen", R / "judgments_r1_confirm", R / "judgments_r2_screen", R / "judgments_r2_confirm"]


def main() -> None:
    df = frame(DIRS)
    bundle = json.loads((PROCESSED / "models" / "model_bundle.json").read_text())
    df["dashboard"] = bundle["linear"]["intercept"] + sum(
        c * df[bundle["themes"][t]["members"]].mean(axis=1) for t, c in bundle["linear"]["coef"].items())

    old_q = [c for c in pd.read_parquet(DIRS[0] / "features.parquet").columns if c != "urn"]
    old_q = [c for c in old_q if c in df.columns and df[c].notna().all()]
    text = [c for c in df.columns if c.startswith("text.")]
    new_q = [c for c in df.columns if c.startswith(("r1_", "r2_"))]
    # only questions answered for every train post outside the generation authors and every test post
    n_answered = df[new_q].notna().sum()
    new_q = [c for c in new_q if n_answered[c] == n_answered.max()]  # screened-out questions lack later authors
    sub = df[df[new_q].notna().all(axis=1)].reset_index(drop=True)
    tr, te = sub[sub["split"] == "train"], sub[sub["split"] == "test"]
    yt, at = te["lift"].to_numpy(), te["author"].to_numpy()
    print(f"train {tr['author'].nunique()} authors / {len(tr)} posts; test {te['author'].nunique()} / {len(te)}; "
          f"{len(old_q)} old + {len(text)} text + {len(new_q)} new questions", flush=True)

    preds = {"dashboard theme model (86 questions, 7 weights)": te["dashboard"].to_numpy()}

    # research joint model and checklist: properties chosen on train only
    jt = pd.read_csv(RESULTS / "joint_r2_train.csv")
    cols = [c for c in jt["id"] if c != "log_words"]
    Z = lambda d: np.column_stack([(d[cols] >= YES).to_numpy(float), np.log1p(d["words"].to_numpy())])  # noqa: E731
    beta = np.nan_to_num(within_ols(tr["lift"].to_numpy(), Z(tr), tr["author"].to_numpy())["beta"])
    preds[f"research joint model ({len(cols)} yes/no properties + length)"] = Z(te) @ beta
    items = pd.read_csv(RESULTS / "checklist_items.csv")["id"].tolist()
    preds[f"research checklist (count of {len(items)} items)"] = (te[items] >= YES).sum(axis=1).to_numpy(float)

    def ridge(qs):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 16), cv=cv_splits(tr["fold"].to_numpy())))
        return m.fit(tr[qs].to_numpy(float), tr["lift"].to_numpy()).predict(te[qs].to_numpy(float))

    preds[f"ridge, old {len(old_q)} questions"] = ridge(old_q)
    preds[f"ridge, old + text + new ({len(old_q) + len(text) + len(new_q)})"] = ridge(old_q + text + new_q)

    per = {k: pairwise_by_author(yt, p, at) for k, p in preds.items()}
    base = next(iter(per))
    out = {}
    for k, p in preds.items():
        r = evaluate(yt, p, at)
        out[k] = {"pairwise": r["pairwise_accuracy"], "ci95": bootstrap_ci(per[k])["ci95"],
                  "spearman": r["author_spearman"],
                  "vs_dashboard": None if k == base else bootstrap_ci(per[k], per[base])}
        d = out[k]["vs_dashboard"]
        print(f"{r['pairwise_accuracy']:.4f} {out[k]['ci95']}  "
              f"{'' if d is None else 'vs dashboard %+.4f %s' % (d['mean'], d['ci95'])}  {k}")
    old, new = list(per)[-2], list(per)[-1]
    gain = bootstrap_ci(per[new], per[old])
    print(f"new questions added to ridge: {gain['mean']:+.4f} {gain['ci95']}")
    out["new_questions_gain_in_ridge"] = gain
    (RESULTS / "compare_dashboard.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
