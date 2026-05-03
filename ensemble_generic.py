"""
Generic probability ensemble — 2-way or 3-way.

Doesn't care about filenames; you pass each model's OOF file and test_proba file
explicitly. This avoids the file-name lock-in of ensemble_lgbm_cnn.py and
ensemble_three.py.

Usage (3-way example):
    python ensemble_generic.py \
        --oof    /path/A/oof_lgbm.csv,/path/B/oof_cnn.csv,/path/C/oof_cnn_wide.csv \
        --test   /path/A/test_proba_lgbm.csv,/path/B/test_proba_cnn.csv,/path/C/test_proba_cnn_wide.csv \
        --names  lgbm,cnn_small,cnn_wide \
        --out_dir ./ensemble_out

Number of OOF, test, and names lists must match (2 or 3 entries each).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_proba(path: Path, with_label: bool):
    df = pd.read_csv(path).sort_values("file_id").reset_index(drop=True)
    cols = sorted([c for c in df.columns if c.startswith("p") and c[1:].isdigit()],
                  key=lambda c: int(c[1:]))
    if with_label and "true_label" not in df.columns:
        raise SystemExit(f"{path} has no true_label column (is this an OOF file?)")
    y = df["true_label"].values if with_label else None
    return df, df[cols].values, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof",   required=True, help="comma-separated OOF csv paths")
    ap.add_argument("--test",  required=True, help="comma-separated test_proba csv paths")
    ap.add_argument("--names", required=True, help="comma-separated model names")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--step", type=float, default=0.05,
                    help="grid step for weight search (smaller = finer + slower)")
    args = ap.parse_args()

    oof_paths  = [Path(p.strip()) for p in args.oof.split(",")]
    test_paths = [Path(p.strip()) for p in args.test.split(",")]
    names      = [n.strip() for n in args.names.split(",")]
    if not (len(oof_paths) == len(test_paths) == len(names)):
        raise SystemExit("--oof, --test, --names must all have the same number of entries")
    n_models = len(oof_paths)
    if n_models not in (2, 3, 4):
        raise SystemExit("currently supports 2-way, 3-way, or 4-way only")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    def emit(s=""): print(s, flush=True); log.append(s)

    # ---- Load OOF ----
    dfs_oof, probas_oof = [], []
    y_ref = None
    for p, n in zip(oof_paths, names):
        df, proba, y = load_proba(p, with_label=True)
        dfs_oof.append(df); probas_oof.append(proba)
        if y_ref is None:
            y_ref = y; ref_ids = df["file_id"].values
        else:
            if not np.array_equal(df["file_id"].values, ref_ids):
                raise SystemExit(f"OOF file_ids don't match between {names[0]} and {n}")
            if not np.array_equal(y, y_ref):
                raise SystemExit(f"OOF true_labels don't match between {names[0]} and {n}")
    y = y_ref
    emit(f"OOF aligned: {len(y)} files, {probas_oof[0].shape[1]} classes\n")

    # ---- Component scores ----
    emit("Component OOF macro F1:")
    comp_f1 = {}
    for n, p in zip(names, probas_oof):
        f1 = f1_score(y, p.argmax(1), average="macro")
        comp_f1[n] = f1
        emit(f"  {n:15s} : {f1:.4f}")

    # ---- Weight sweep (constrained to sum to 1) ----
    grid = np.arange(0, 1 + 1e-9, args.step)
    best = {"f1": -1.0, "w": None}
    if n_models == 2:
        emit(f"\nTwo-way sweep ({names[0]} vs {names[1]}):")
        for w0 in grid:
            w = (w0, 1 - w0)
            blend = w[0]*probas_oof[0] + w[1]*probas_oof[1]
            f1 = f1_score(y, blend.argmax(1), average="macro")
            if f1 > best["f1"]:
                best = {"f1": f1, "w": w}
        emit(f"  best weights: {names[0]}={best['w'][0]:.2f}  {names[1]}={best['w'][1]:.2f}")
    elif n_models == 3:
        emit(f"\nThree-way sweep ({', '.join(names)}):")
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < -1e-9 or w2 > 1 + 1e-9:
                    continue
                w2 = max(0.0, min(1.0, w2))
                w = (w0, w1, w2)
                blend = w[0]*probas_oof[0] + w[1]*probas_oof[1] + w[2]*probas_oof[2]
                f1 = f1_score(y, blend.argmax(1), average="macro")
                if f1 > best["f1"]:
                    best = {"f1": f1, "w": w}
        emit(f"  best weights: {names[0]}={best['w'][0]:.2f}  "
             f"{names[1]}={best['w'][1]:.2f}  {names[2]}={best['w'][2]:.2f}")
    else:  # n_models == 4
        emit(f"\nFour-way sweep ({', '.join(names)}):")
        for w0 in grid:
            for w1 in grid:
                if w0 + w1 > 1 + 1e-9:
                    continue
                for w2 in grid:
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -1e-9 or w3 > 1 + 1e-9:
                        continue
                    w3 = max(0.0, min(1.0, w3))
                    w = (w0, w1, w2, w3)
                    blend = sum(w[i] * probas_oof[i] for i in range(4))
                    f1 = f1_score(y, blend.argmax(1), average="macro")
                    if f1 > best["f1"]:
                        best = {"f1": f1, "w": w}
        emit(f"  best weights: " + "  ".join(
            f"{n}={w_:.2f}" for n, w_ in zip(names, best["w"])
        ))
    emit(f"  best  OOF F1: {best['f1']:.4f}")
    for n in names:
        emit(f"  vs {n} alone: {best['f1'] - comp_f1[n]:+.4f}")

    # ---- Detailed report at best blend ----
    blend = sum(best["w"][i] * probas_oof[i] for i in range(n_models))
    pred = blend.argmax(axis=1)
    emit("\n=== OOF classification report at best blend ===")
    emit(classification_report(y, pred, digits=3))
    cm = confusion_matrix(y, pred, normalize="true")
    cm_df = pd.DataFrame(cm.round(3),
                         index=[f"true={i}" for i in range(cm.shape[0])],
                         columns=[f"pred={i}" for i in range(cm.shape[1])])
    emit("=== OOF confusion matrix at best blend (row-normalized) ===")
    emit(cm_df.to_string())

    # ---- Apply to test ----
    dfs_te, probas_te = [], []
    test_ids_ref = None
    for p, n in zip(test_paths, names):
        df, proba, _ = load_proba(p, with_label=False)
        dfs_te.append(df); probas_te.append(proba)
        if test_ids_ref is None:
            test_ids_ref = df["file_id"].values
        elif not np.array_equal(df["file_id"].values, test_ids_ref):
            raise SystemExit(f"test file_ids don't match between {names[0]} and {n}")

    test_blend = sum(best["w"][i] * probas_te[i] for i in range(n_models))
    test_pred = test_blend.argmax(axis=1)

    sub = pd.DataFrame({"Id": test_ids_ref, "Label": test_pred}).sort_values("Id").reset_index(drop=True)
    sub.to_csv(out_dir / "submission_ensemble.csv", index=False)
    emit(f"\nWrote {out_dir/'submission_ensemble.csv'}  ({len(sub)} rows)")

    proba_df = pd.DataFrame(test_blend, columns=[f"p{c}" for c in range(test_blend.shape[1])])
    proba_df.insert(0, "file_id", test_ids_ref)
    proba_df.to_csv(out_dir / "test_proba_ensemble.csv", index=False)
    emit(f"Wrote {out_dir/'test_proba_ensemble.csv'}")

    weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(names, best["w"]))
    emit(f"\nMetadata: weights = ({weight_str})")
    (out_dir / "ensemble_log.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()
