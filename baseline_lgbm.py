"""
Baseline 2: LightGBM on aggregate + temporal features (file_summary_v2.csv),
GroupKFold(n_splits=5) by user.

Drop-in replacement for baseline_rf.py. Same input file, same output schema.
Reuses fold_assignments.csv if present so the v2-RF and v2-LGBM CV numbers are
directly comparable (each fold trains and validates on the exact same users).

Usage:
    python baseline_lgbm.py \
        --summary    file_summary_v2.csv \
        --folds      fold_assignments.csv  (optional but recommended)
        --out_dir    .

Outputs:
    oof_predictions_lgbm.csv : (file_id, user, true_label, pred_label, p0..p5)
    submission_lgbm.csv      : Kaggle submission (Id, Label)
    baseline_lgbm_log.txt    : per-fold scores, classification report, CM
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score)
from sklearn.model_selection import GroupKFold


FEATURE_PREFIXES = (
    "mean_x__", "mean_y__", "mean_z__",
    "std_x__",  "std_y__",  "std_z__",
    "mean_mag__", "std_mag__",
    "corr_mean_",   # cross-axis correlations added in v2
)


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]
    # Drop boolean / non-numeric sanity columns that snuck in
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def get_fold_indices(train: pd.DataFrame, folds_csv: Path | None,
                      n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    If folds_csv exists, use the saved user→fold mapping (so CV scores across
    different models are directly comparable). Otherwise, regenerate via
    GroupKFold so this script can run standalone.
    """
    if folds_csv is not None and folds_csv.exists():
        fold_map = pd.read_csv(folds_csv).set_index("user")["fold"].to_dict()
        missing = [u for u in train["user"].unique() if u not in fold_map]
        if missing:
            raise SystemExit(f"users in train not in {folds_csv}: {missing[:5]}...")
        train_fold = train["user"].map(fold_map).values
        n = int(train_fold.max()) + 1
        print(f"[folds] reusing {folds_csv}  ({n} folds)")
        return [(np.where(train_fold != f)[0], np.where(train_fold == f)[0])
                for f in range(n)]

    print(f"[folds] no fold_assignments.csv found — regenerating GroupKFold(n_splits={n_splits})")
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(train, train["label"].values, train["user"].values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default="file_summary_v2.csv")
    ap.add_argument("--folds", type=str, default="fold_assignments.csv")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--n_splits", type=int, default=5)
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

    X_tr = train[feat_cols].values.astype(np.float32)
    y_tr = train["label"].values.astype(int)
    X_te = test[feat_cols].values.astype(np.float32)
    n_classes = int(y_tr.max()) + 1
    emit(f"Number of classes: {n_classes}")

    # Class weights — same idea as RF's class_weight='balanced'
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)
    sample_weight = cw[y_tr]

    folds_csv = Path(args.folds) if args.folds else None
    fold_iter = get_fold_indices(train, folds_csv, args.n_splits)

    lgbm_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": 0,  # auto
    }
    emit(f"\nLightGBM params: {lgbm_params}")

    oof_proba = np.zeros((len(y_tr), n_classes), dtype=np.float32)
    test_proba = np.zeros((len(X_te), n_classes), dtype=np.float32)
    fold_scores = []
    best_iters = []

    emit("\n=== Per-fold OOF (LightGBM, sample_weight=balanced) ===")
    for fold, (tr_idx, va_idx) in enumerate(fold_iter):
        dtrain = lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx],
                             weight=sample_weight[tr_idx],
                             feature_name=feat_cols)
        dvalid = lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx],
                             weight=sample_weight[va_idx],
                             reference=dtrain)
        model = lgb.train(
            lgbm_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iters.append(model.best_iteration)
        oof_proba[va_idx] = model.predict(X_tr[va_idx], num_iteration=model.best_iteration)
        test_proba += model.predict(X_te, num_iteration=model.best_iteration) / len(fold_iter)
        oof_pred_fold = oof_proba[va_idx].argmax(axis=1)
        f1 = f1_score(y_tr[va_idx], oof_pred_fold, average="macro")
        fold_scores.append(f1)
        emit(f"  fold {fold}: best_iter={model.best_iteration:>4d}  "
             f"val macro F1 = {f1:.4f}")

    oof_pred = oof_proba.argmax(axis=1)
    overall_f1 = f1_score(y_tr, oof_pred, average="macro")
    emit(f"\nOOF macro F1: {overall_f1:.4f}  "
         f"(fold mean {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f})")
    emit(f"Best iterations across folds: {best_iters}  (median {int(np.median(best_iters))})")

    emit("\n=== OOF classification report ===")
    emit(classification_report(y_tr, oof_pred, digits=3))

    cm = confusion_matrix(y_tr, oof_pred, normalize="true")
    cm_df = pd.DataFrame(
        cm.round(3),
        index=[f"true={i}" for i in range(cm.shape[0])],
        columns=[f"pred={i}" for i in range(cm.shape[1])],
    )
    emit("=== OOF confusion matrix (row-normalized) ===")
    emit(cm_df.to_string())

    # ---------------- Save outputs ----------------
    oof_df = pd.DataFrame({
        "file_id": train["file_id"].values,
        "user": train["user"].values,
        "true_label": y_tr,
        "pred_label": oof_pred,
    })
    for c in range(n_classes):
        oof_df[f"p{c}"] = oof_proba[:, c]
    oof_df.to_csv(out_dir / "oof_predictions_lgbm.csv", index=False)
    emit(f"\nWrote {out_dir/'oof_predictions_lgbm.csv'}")

    # Test predictions are the average of the 5 fold-models' probabilities
    test_pred = test_proba.argmax(axis=1)
    sub = pd.DataFrame({"Id": test["file_id"].values, "Label": test_pred})
    sub = sub.sort_values("Id").reset_index(drop=True)
    sub.to_csv(out_dir / "submission_lgbm.csv", index=False)
    emit(f"Wrote {out_dir/'submission_lgbm.csv'}  ({len(sub)} rows)")

    # Also save test probabilities — needed later for ensembling with the CNN
    test_proba_df = pd.DataFrame(test_proba, columns=[f"p{c}" for c in range(n_classes)])
    test_proba_df.insert(0, "file_id", test["file_id"].values)
    test_proba_df = test_proba_df.sort_values("file_id").reset_index(drop=True)
    test_proba_df.to_csv(out_dir / "test_proba_lgbm.csv", index=False)
    emit(f"Wrote {out_dir/'test_proba_lgbm.csv'}")

    (out_dir / "baseline_lgbm_log.txt").write_text("\n".join(log))
    emit(f"Wrote {out_dir/'baseline_lgbm_log.txt'}")


if __name__ == "__main__":
    main()
