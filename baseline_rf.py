"""
Baseline 1: Random Forest on aggregate features in file_summary.csv,
GroupKFold(n_splits=5) by user.

Reproduces the OOF macro F1 ~ 0.6562 we measured.

Usage:
    python baseline_rf.py --summary path/to/file_summary.csv --out_dir .

Outputs (into --out_dir):
    - oof_predictions.csv : (file_id, user, true_label, pred_label) for each train file
    - fold_assignments.csv: (user, fold)  — locked-in CV split for future experiments
    - submission_rf.csv   : Kaggle submission (Id, Label), trained on the full train set
    - baseline_rf_log.txt : per-fold scores, OOF F1, classification report, confusion matrix
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)
from sklearn.model_selection import GroupKFold


# Feature prefixes from the EDA summary. Keep in one place so we can extend later.
FEATURE_PREFIXES = (
    "mean_x__", "mean_y__", "mean_z__",
    "std_x__",  "std_y__",  "std_z__",
    "mean_mag__", "std_mag__",
)


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="file_summary.csv")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    def emit(msg=""):
        print(msg, flush=True)
        log.append(msg)

    fs = pd.read_csv(args.summary)
    train = fs[fs.split == "train"].reset_index(drop=True)
    test = fs[fs.split == "test"].reset_index(drop=True)
    feat_cols = select_feature_cols(train)

    emit(f"Train shape: {train.shape}, Test shape: {test.shape}")
    emit(f"Number of features used: {len(feat_cols)}")
    emit(f"GroupKFold n_splits={args.n_splits}, seed={args.seed}")

    X_tr = train[feat_cols].values
    y_tr = train["label"].values
    groups = train["user"].values

    # ---------------- Cross-validation ----------------
    gkf = GroupKFold(n_splits=args.n_splits)
    oof = np.full_like(y_tr, fill_value=-1)
    fold_of_user = {}
    fold_scores = []

    emit("\n=== Per-fold OOF (RF, class_weight=balanced) ===")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_tr, groups)):
        for u in np.unique(groups[va_idx]):
            fold_of_user[u] = fold

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed,
            min_samples_leaf=2,
        )
        clf.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[va_idx] = clf.predict(X_tr[va_idx])
        f1 = f1_score(y_tr[va_idx], oof[va_idx], average="macro")
        fold_scores.append(f1)
        emit(f"  fold {fold}: val_users={len(np.unique(groups[va_idx]))}  "
             f"val macro F1 = {f1:.4f}")

    overall_f1 = f1_score(y_tr, oof, average="macro")
    emit(f"\nOOF macro F1: {overall_f1:.4f}  "
         f"(fold mean {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f})")

    emit("\n=== OOF classification report ===")
    emit(classification_report(y_tr, oof, digits=3))

    cm = confusion_matrix(y_tr, oof, normalize="true")
    cm_df = pd.DataFrame(
        cm.round(3),
        index=[f"true={i}" for i in range(cm.shape[0])],
        columns=[f"pred={i}" for i in range(cm.shape[1])],
    )
    emit("=== OOF confusion matrix (row-normalized) ===")
    emit(cm_df.to_string())

    # ---------------- Save OOF + fold assignments ----------------
    pd.DataFrame({
        "file_id": train["file_id"].values,
        "user": train["user"].values,
        "true_label": y_tr,
        "pred_label": oof,
    }).to_csv(out_dir / "oof_predictions.csv", index=False)
    emit(f"\nWrote {out_dir/'oof_predictions.csv'}")

    pd.DataFrame(
        sorted(fold_of_user.items()), columns=["user", "fold"]
    ).to_csv(out_dir / "fold_assignments.csv", index=False)
    emit(f"Wrote {out_dir/'fold_assignments.csv'}  "
         "(use the same splits in every future experiment)")

    # ---------------- Train on full train, predict test, write submission ----------------
    final_clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
        min_samples_leaf=2,
    )
    final_clf.fit(X_tr, y_tr)
    X_te = test[feat_cols].values
    test_pred = final_clf.predict(X_te)

    sub = pd.DataFrame({"Id": test["file_id"].values, "Label": test_pred})
    sub = sub.sort_values("Id").reset_index(drop=True)
    sub.to_csv(out_dir / "submission_rf.csv", index=False)
    emit(f"Wrote {out_dir/'submission_rf.csv'}  ({len(sub)} rows)")

    (out_dir / "baseline_rf_log.txt").write_text("\n".join(log))
    emit(f"Wrote {out_dir/'baseline_rf_log.txt'}")


if __name__ == "__main__":
    main()
