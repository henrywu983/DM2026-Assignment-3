"""
Baseline 4: Wider 1D-CNN.

Identical to baseline_cnn.py (CE+balanced, no augmentation, same input pipeline,
same fold splits) EXCEPT the CNN itself is wider:

  block channels:  v1     32 -> 64  -> 96   (~37k params)
                   wide   64 -> 128 -> 192  (~155k params)
  FC head:         v1     96 -> 64 -> 6
                   wide   192 -> 128 -> 6
  dropout:         v1     0.1 conv / 0.2 fc
                   wide   0.2 conv / 0.3 fc   (more capacity -> more dropout)

Same kernel sizes (7/5/3), same pool factors, same global avg pool. The point
is to test whether the CNN's ceiling was being set by capacity, not by training.

Usage:
    python baseline_cnn_wide.py \
        --train_dir /path/to/train \
        --test_dir  /path/to/test \
        --folds     fold_assignments.csv \
        --out_dir   ./baseline_out_cnn_wide \
        --epochs    40 --patience 8

Outputs:
    oof_predictions_cnn_wide.csv
    test_proba_cnn_wide.csv
    submission_cnn_wide.csv
    baseline_cnn_wide_log.txt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


# ----------------------------- data layer -----------------------------

RAW_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def find_user_root(root: Path, max_depth: int = 3) -> Path:
    """Same wrapper-folder descend as eda_summary.py / v2."""
    cur = root
    for _ in range(max_depth + 1):
        subdirs = [p for p in cur.iterdir() if p.is_dir()]
        if any(p.name.startswith("User_") for p in subdirs):
            return cur
        if len(subdirs) == 1:
            cur = subdirs[0]
            continue
        break
    raise SystemExit(
        f"Could not find User_* folders under {root}. Last inspected: {cur}."
    )


def index_split(root: Path, split: str) -> pd.DataFrame:
    """Walk a split folder; return one row per file with file_id, user, label, path."""
    user_root = find_user_root(root)
    rows = []
    for ud in sorted(p for p in user_root.iterdir() if p.is_dir() and p.name.startswith("User_")):
        for f in sorted(ud.glob("*.csv")):
            # Read just enough to get file_id and label (cheap)
            head = pd.read_csv(f, nrows=1)
            file_id = int(head["file_id"].iloc[0]) if "file_id" in head.columns else int(f.stem)
            label = int(head["label"].iloc[0]) if ("label" in head.columns and split == "train") else -1
            rows.append({"split": split, "user": ud.name, "file_id": file_id,
                         "label": label, "path": str(f)})
    return pd.DataFrame(rows)


def load_sequence(path: str) -> np.ndarray:
    """Load one CSV and produce an (8, 300) float32 array.

    Channels 0-5: raw, per-file z-normalized
    Channels 6-7: ||mean|| and ||std||, kept on original scale
    """
    df = pd.read_csv(path)
    raw = df[RAW_COLS].values.astype(np.float32)        # (300, 6)
    if raw.shape[0] != 300:
        # pad / truncate defensively (shouldn't happen with this dataset)
        if raw.shape[0] < 300:
            pad = np.zeros((300 - raw.shape[0], 6), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)
        else:
            raw = raw[:300]
    mean_mag = np.sqrt(raw[:, 0]**2 + raw[:, 1]**2 + raw[:, 2]**2)
    std_mag = np.sqrt(raw[:, 3]**2 + raw[:, 4]**2 + raw[:, 5]**2)

    # Per-file z-normalize the 6 raw channels only
    mu = raw.mean(axis=0, keepdims=True)
    sigma = raw.std(axis=0, keepdims=True) + 1e-6
    raw_n = (raw - mu) / sigma

    seq = np.concatenate([raw_n, mean_mag[:, None], std_mag[:, None]], axis=1)  # (300, 8)
    return seq.T.astype(np.float32)  # (8, 300) for Conv1d


class HARDataset(Dataset):
    def __init__(self, df: pd.DataFrame, has_label: bool = True):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist() if has_label else [-1] * len(df)
        self.has_label = has_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = load_sequence(self.paths[i])  # (8, 300)
        y = self.labels[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ----------------------------- model -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7, pool=2, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2)
        self.bn = nn.BatchNorm1d(c_out)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        x = self.drop(x)
        return x


class WideCNN(nn.Module):
    """~155k param 1D-CNN: wider channels and beefier FC head than SmallCNN.

    Channel widths doubled (64/128/192 vs 32/64/96), FC hidden doubled
    (128 vs 64), and dropout raised (0.2 conv / 0.3 fc) to compensate
    for the increased capacity.
    """
    def __init__(self, n_classes: int = 6, in_ch: int = 8):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 64,  k=7, pool=2, dropout=0.2)   # 300 -> 150
        self.block2 = ConvBlock(64,    128, k=5, pool=2, dropout=0.2)   # 150 -> 75
        self.block3 = ConvBlock(128,   192, k=3, pool=2, dropout=0.2)   # 75  -> 37
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


# ----------------------------- training loop -----------------------------

@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    all_p = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = F.softmax(logits, dim=1).cpu().numpy()
        all_p.append(p)
    return np.concatenate(all_p, axis=0)


def train_one_fold(train_df, valid_df, n_classes, class_weights, args, device, log):
    train_ds = HARDataset(train_df, has_label=True)
    valid_ds = HARDataset(valid_df, has_label=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = WideCNN(n_classes=n_classes, in_ch=8).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  model params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=cw)

    best_f1, best_epoch, best_state = -1.0, -1, None
    patience = args.patience
    epochs_since_best = 0
    y_valid_np = np.array(valid_df["label"].tolist())

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss, n_seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        sched.step()
        train_loss = running_loss / max(n_seen, 1)

        proba = predict_proba(model, valid_loader, device)
        pred = proba.argmax(axis=1)
        val_f1 = f1_score(y_valid_np, pred, average="macro")
        dt = time.time() - t0

        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_proba = proba
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        log(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_f1={val_f1:.4f}  "
            f"best={best_f1:.4f}@{best_epoch}  ({dt:.1f}s)")

        if epochs_since_best >= patience:
            log(f"  early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights for test prediction
    model.load_state_dict(best_state)
    return model, best_proba, best_f1, best_epoch


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, required=True)
    ap.add_argument("--test_dir",  type=str, required=True)
    ap.add_argument("--folds",     type=str, default="fold_assignments.csv")
    ap.add_argument("--out_dir",   type=str, default="./baseline_out_cnn")
    ap.add_argument("--epochs",    type=int, default=40)
    ap.add_argument("--patience",  type=int, default=8)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(msg)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    # ---- index files ----
    log("\nIndexing train and test directories...")
    train_idx = index_split(Path(args.train_dir), "train")
    test_idx  = index_split(Path(args.test_dir),  "test")
    log(f"  train files: {len(train_idx)}  ({train_idx['user'].nunique()} users)")
    log(f"  test files:  {len(test_idx)}   ({test_idx['user'].nunique()} users)")

    n_classes = int(train_idx["label"].max()) + 1
    log(f"  n_classes:   {n_classes}")

    # ---- class weights (balanced, same as LGBM) ----
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.arange(n_classes),
                               y=train_idx["label"].values)
    log(f"  class weights: {[round(float(w),3) for w in cw]}")

    # ---- fold assignments ----
    folds_path = Path(args.folds)
    if not folds_path.exists():
        raise SystemExit(f"Need fold_assignments.csv at {folds_path}")
    fold_map = pd.read_csv(folds_path).set_index("user")["fold"].to_dict()
    train_idx["fold"] = train_idx["user"].map(fold_map)
    if train_idx["fold"].isna().any():
        missing = train_idx[train_idx["fold"].isna()]["user"].unique()
        raise SystemExit(f"users missing from {folds_path}: {missing[:5]}")
    n_folds = int(train_idx["fold"].max()) + 1
    log(f"  reused {folds_path}  ({n_folds} folds)")

    # ---- per-fold training ----
    oof_proba = np.zeros((len(train_idx), n_classes), dtype=np.float32)
    test_proba = np.zeros((len(test_idx), n_classes), dtype=np.float32)
    fold_scores = []
    test_ds = HARDataset(test_idx, has_label=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    for fold in range(n_folds):
        log(f"\n=== Fold {fold} ===")
        valid_mask = (train_idx["fold"] == fold).values
        tr_df = train_idx[~valid_mask].reset_index(drop=True)
        va_df = train_idx[ valid_mask].reset_index(drop=True)
        log(f"  train n={len(tr_df)}  val n={len(va_df)}  "
            f"val users={va_df['user'].nunique()}")

        model, val_proba, best_f1, best_epoch = train_one_fold(
            tr_df, va_df, n_classes, cw, args, device, log
        )
        # store OOF probas at the right indices in train_idx
        va_indices = np.where(valid_mask)[0]
        oof_proba[va_indices] = val_proba
        fold_scores.append(best_f1)
        log(f"  fold {fold} best val macro F1 = {best_f1:.4f} (epoch {best_epoch})")

        # predict test with this fold's model, accumulate average
        fold_test_proba = predict_proba(model, test_loader, device)
        test_proba += fold_test_proba / n_folds
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    y_tr = train_idx["label"].values
    oof_pred = oof_proba.argmax(axis=1)
    overall_f1 = f1_score(y_tr, oof_pred, average="macro")
    log(f"\nOOF macro F1: {overall_f1:.4f}  "
        f"(fold mean {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f})")
    log("\n=== OOF classification report ===")
    log(classification_report(y_tr, oof_pred, digits=3))

    cm = confusion_matrix(y_tr, oof_pred, normalize="true")
    cm_df = pd.DataFrame(cm.round(3),
                          index=[f"true={i}" for i in range(cm.shape[0])],
                          columns=[f"pred={i}" for i in range(cm.shape[1])])
    log("=== OOF confusion matrix (row-normalized) ===")
    log(cm_df.to_string())

    # ---- save OOF ----
    oof_df = pd.DataFrame({
        "file_id": train_idx["file_id"].values,
        "user": train_idx["user"].values,
        "true_label": y_tr,
        "pred_label": oof_pred,
    })
    for c in range(n_classes):
        oof_df[f"p{c}"] = oof_proba[:, c]
    oof_df.to_csv(out_dir / "oof_predictions_cnn_wide.csv", index=False)
    log(f"\nWrote {out_dir/'oof_predictions_cnn_wide.csv'}")

    # ---- save test predictions and submission ----
    test_pred = test_proba.argmax(axis=1)
    sub = pd.DataFrame({"Id": test_idx["file_id"].values, "Label": test_pred})
    sub = sub.sort_values("Id").reset_index(drop=True)
    sub.to_csv(out_dir / "submission_cnn_wide.csv", index=False)
    log(f"Wrote {out_dir/'submission_cnn_wide.csv'}  ({len(sub)} rows)")

    test_proba_df = pd.DataFrame(test_proba, columns=[f"p{c}" for c in range(n_classes)])
    test_proba_df.insert(0, "file_id", test_idx["file_id"].values)
    test_proba_df = test_proba_df.sort_values("file_id").reset_index(drop=True)
    test_proba_df.to_csv(out_dir / "test_proba_cnn_wide.csv", index=False)
    log(f"Wrote {out_dir/'test_proba_cnn_wide.csv'}")

    (out_dir / "baseline_cnn_wide_log.txt").write_text("\n".join(log_lines))
    log(f"Wrote {out_dir/'baseline_cnn_wide_log.txt'}")


if __name__ == "__main__":
    main()
