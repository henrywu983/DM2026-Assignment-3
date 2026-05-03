"""
EDA summary script for HAR Assignment 3.

Run this locally pointing at your train/ folder (and optionally test/).
It produces two small files you can upload back:

  - file_summary.csv   : one row per CSV file (file_id, user, label, aggregate stats)
  - dataset_meta.txt   : dataset-level counts and a quick overview

Usage:
    python eda_summary.py --train_dir path/to/train --test_dir path/to/test --out_dir .

Both --train_dir and --test_dir are optional; pass at least one. Output files
are small (a few MB at most) and safe to upload.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


def summarize_file(csv_path: Path, split: str):
    """Read one CSV and return a dict of summary stats for that file."""
    df = pd.read_csv(csv_path)

    # Basic identifiers
    file_id = int(df["file_id"].iloc[0]) if "file_id" in df.columns else int(csv_path.stem)
    user = csv_path.parent.name  # e.g. "User_017"

    # Label may not exist in test/
    if "label" in df.columns and df["label"].notna().any():
        label_vals = df["label"].dropna().unique()
        label = int(label_vals[0]) if len(label_vals) >= 1 else -1
        n_unique_labels = int(df["label"].nunique())
    else:
        label = -1
        n_unique_labels = 0

    n_rows = len(df)

    # Sanity: index is contiguous 0..n_rows-1?
    if "index" in df.columns:
        idx_ok = bool((df["index"].values == np.arange(n_rows)).all())
    else:
        idx_ok = False

    # Per-feature aggregates over the 300 seconds
    out = {
        "split": split,
        "user": user,
        "file_id": file_id,
        "label": label,
        "n_rows": n_rows,
        "idx_contiguous": idx_ok,
        "n_unique_labels": n_unique_labels,
        "any_null": bool(df[FEATURE_COLS].isnull().any().any()),
    }
    for c in FEATURE_COLS:
        v = df[c].values
        out[f"{c}__mean"] = float(np.mean(v))
        out[f"{c}__std"] = float(np.std(v))
        out[f"{c}__min"] = float(np.min(v))
        out[f"{c}__max"] = float(np.max(v))

    # Magnitude features (orientation-invariant)
    mean_mag = np.sqrt(df["mean_x"] ** 2 + df["mean_y"] ** 2 + df["mean_z"] ** 2)
    std_mag = np.sqrt(df["std_x"] ** 2 + df["std_y"] ** 2 + df["std_z"] ** 2)
    out["mean_mag__mean"] = float(mean_mag.mean())
    out["mean_mag__std"] = float(mean_mag.std())
    out["mean_mag__min"] = float(mean_mag.min())
    out["mean_mag__max"] = float(mean_mag.max())
    out["std_mag__mean"] = float(std_mag.mean())
    out["std_mag__std"] = float(std_mag.std())
    out["std_mag__min"] = float(std_mag.min())
    out["std_mag__max"] = float(std_mag.max())

    # Fraction of seconds with std numerically zero (raw signal was constant in that 1s)
    for c in ["std_x", "std_y", "std_z"]:
        out[f"{c}__frac_zero"] = float((df[c].abs() < 1e-10).mean())

    # Temporal change: how much do means drift across the 5-minute window?
    # First half vs second half (proxy for whether activity is steady or evolves)
    half = n_rows // 2
    for c in ["mean_x", "mean_y", "mean_z"]:
        out[f"{c}__drift"] = float(df[c].iloc[half:].mean() - df[c].iloc[:half].mean())

    return out


def find_user_root(root: Path, max_depth: int = 3) -> Path:
    """
    The Kaggle zip extracts to a nested layout like train/train/User_001/...,
    so the path the user passes might be one or two levels above the actual
    User_xxx folders. Descend up to max_depth levels until we find a directory
    whose immediate children start with 'User_'.
    """
    cur = root
    for _ in range(max_depth + 1):
        subdirs = [p for p in cur.iterdir() if p.is_dir()]
        if any(p.name.startswith("User_") for p in subdirs):
            return cur
        # Not at the right level — if there's exactly one subdir, descend
        if len(subdirs) == 1:
            cur = subdirs[0]
            continue
        break
    raise SystemExit(
        f"Could not find User_* folders under {root}. "
        f"Last inspected: {cur}. Subdirs there: {[p.name for p in cur.iterdir() if p.is_dir()]}"
    )


def walk_split(root: Path, split: str):
    """Walk a split folder (train/ or test/) and summarize every CSV inside it."""
    user_root = find_user_root(root)
    if user_root != root:
        print(f"[{split}] descended from {root} to {user_root}")
    rows = []
    user_dirs = sorted([p for p in user_root.iterdir() if p.is_dir() and p.name.startswith("User_")])
    print(f"[{split}] found {len(user_dirs)} user folders under {user_root}")
    for ud in user_dirs:
        files = sorted(ud.glob("*.csv"))
        for f in files:
            try:
                rows.append(summarize_file(f, split))
            except Exception as e:
                print(f"  ! failed to read {f}: {e}")
    return rows


def write_meta(df: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("=== HAR Assignment 3 dataset summary ===\n")
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        lines.append(f"\n--- split: {split} ---")
        lines.append(f"files: {len(sub)}")
        lines.append(f"users: {sub['user'].nunique()}")
        lines.append(f"sequence length unique values: {sorted(sub['n_rows'].unique())}")
        lines.append(f"any file with null features: {bool(sub['any_null'].any())}")
        lines.append(f"any file with non-contiguous index: {bool((~sub['idx_contiguous']).any())}")
        lines.append(f"any file with multiple distinct labels: {bool((sub['n_unique_labels'] > 1).any())}")
        if (sub["label"] >= 0).any():
            lines.append("\nfiles per label:")
            lines.append(sub["label"].value_counts().sort_index().to_string())
            lines.append("\nfiles per user (head/tail):")
            counts = sub["user"].value_counts().sort_values(ascending=False)
            lines.append(counts.head(10).to_string())
            lines.append("...")
            lines.append(counts.tail(5).to_string())
            lines.append(f"\nfiles per user — min={counts.min()}, max={counts.max()}, "
                         f"mean={counts.mean():.1f}, median={counts.median():.0f}")
            # Per-user label coverage: how many distinct labels each user has
            user_label_cov = sub.groupby("user")["label"].nunique()
            lines.append("\nnumber of distinct labels per user (distribution):")
            lines.append(user_label_cov.value_counts().sort_index().to_string())
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default=None)
    ap.add_argument("--test_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    if not args.train_dir and not args.test_dir:
        raise SystemExit("Pass at least one of --train_dir or --test_dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    if args.train_dir:
        all_rows += walk_split(Path(args.train_dir), "train")
    if args.test_dir:
        all_rows += walk_split(Path(args.test_dir), "test")

    df = pd.DataFrame(all_rows)
    out_csv = out_dir / "file_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  shape={df.shape}")

    write_meta(df, out_dir / "dataset_meta.txt")


if __name__ == "__main__":
    main()
