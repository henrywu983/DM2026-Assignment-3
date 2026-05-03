"""
EDA / feature-extraction script v2.

Superset of v1 (eda_summary.py): every column from file_summary.csv is also
present in file_summary_v2.csv, plus a richer set of features designed to
capture *temporal pattern* rather than just per-file aggregates.

What's new vs v1 (per-file):
  - Percentiles (p10, p25, p50, p75, p90) for each raw column and each magnitude
  - Spectral features on mean_x/y/z and std_mag (dominant freq, spectral entropy,
    spectral centroid, low-band energy ratio)
  - Autocorrelation at lags 1, 2, 5, 10 on mean_x/y/z and std_mag
  - Peak features on std_mag (motion-burst count and amplitudes)
  - Cross-axis correlations between mean_x/y/z
  - Stationarity ratios (early-vs-late variance on std_mag)

Note on sampling: data is already aggregated to 1 Hz, so spectral/autocorrelation
features capture *slow* (multi-second) modulations of the per-second statistics,
not the original sub-second sensor cadence (which has been lost in aggregation).

Usage:
    python eda_summary_v2.py --train_dir path/to/train --test_dir path/to/test --out_dir .

Outputs:
    file_summary_v2.csv     (1 row per file, ~150 columns)
    dataset_meta_v2.txt     (counts and overview, same format as v1)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

FEATURE_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]


# -------------------------- per-series feature helpers --------------------------

def basic_stats(v: np.ndarray) -> dict:
    """Mean / std / min / max / percentiles."""
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    return {
        "mean": float(v.mean()), "std": float(v.std()),
        "min": float(v.min()), "max": float(v.max()),
        "p10": float(p10), "p25": float(p25), "p50": float(p50),
        "p75": float(p75), "p90": float(p90),
    }


def spectral_stats(v: np.ndarray, fs: float = 1.0) -> dict:
    """
    FFT-based spectral features. Operates on the de-meaned signal so DC doesn't
    dominate. Returns NaN-safe values (zeros) for near-constant inputs.
    """
    n = len(v)
    v0 = v - v.mean()
    if v0.std() < 1e-9:
        return {"dom_freq": 0.0, "dom_pow_ratio": 0.0,
                "spec_entropy": 0.0, "spec_centroid": 0.0,
                "lowband_ratio": 0.0}

    # rfft: positive frequencies only, length n//2+1
    spec = np.abs(np.fft.rfft(v0))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # drop DC bin (already 0 because we de-meaned, but kept defensively)
    spec, freqs = spec[1:], freqs[1:]
    power = spec ** 2
    total = power.sum()
    if total < 1e-12:
        return {"dom_freq": 0.0, "dom_pow_ratio": 0.0,
                "spec_entropy": 0.0, "spec_centroid": 0.0,
                "lowband_ratio": 0.0}

    p_norm = power / total
    dom_idx = int(np.argmax(power))
    dom_freq = float(freqs[dom_idx])
    dom_pow_ratio = float(p_norm[dom_idx])
    spec_entropy = float(-(p_norm * np.log(p_norm + 1e-12)).sum())
    spec_centroid = float((freqs * p_norm).sum())
    # "rhythmic" mid band 0.05-0.25 Hz (period 4-20 s), captures slow modulations
    band_mask = (freqs >= 0.05) & (freqs <= 0.25)
    lowband_ratio = float(p_norm[band_mask].sum())

    return {"dom_freq": dom_freq, "dom_pow_ratio": dom_pow_ratio,
            "spec_entropy": spec_entropy, "spec_centroid": spec_centroid,
            "lowband_ratio": lowband_ratio}


def autocorr_lags(v: np.ndarray, lags=(1, 2, 5, 10)) -> dict:
    """Pearson autocorrelation at small integer lags."""
    out = {}
    if v.std() < 1e-9:
        return {f"acf_lag{k}": 0.0 for k in lags}
    for k in lags:
        a, b = v[:-k], v[k:]
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            out[f"acf_lag{k}"] = 0.0
        else:
            out[f"acf_lag{k}"] = float(np.corrcoef(a, b)[0, 1])
    return out


def peak_stats(v: np.ndarray) -> dict:
    """Count and amplitude of peaks in a signal (used on std_mag = motion bursts)."""
    if v.std() < 1e-9:
        return {"n_peaks": 0, "peak_mean": 0.0, "peak_max": 0.0,
                "peak_interval_mean": 0.0}
    # height threshold: any peak above the median
    height = float(np.median(v))
    peaks, props = find_peaks(v, height=height, distance=2)
    if len(peaks) == 0:
        return {"n_peaks": 0, "peak_mean": 0.0, "peak_max": 0.0,
                "peak_interval_mean": 0.0}
    peak_heights = props["peak_heights"]
    if len(peaks) >= 2:
        intervals = np.diff(peaks)
        interval_mean = float(intervals.mean())
    else:
        interval_mean = 0.0
    return {"n_peaks": int(len(peaks)),
            "peak_mean": float(peak_heights.mean()),
            "peak_max": float(peak_heights.max()),
            "peak_interval_mean": interval_mean}


def add_prefixed(out: dict, prefix: str, d: dict):
    for k, v in d.items():
        out[f"{prefix}__{k}"] = v


# -------------------------- per-file summary --------------------------

def summarize_file(csv_path: Path, split: str) -> dict:
    df = pd.read_csv(csv_path)
    file_id = int(df["file_id"].iloc[0]) if "file_id" in df.columns else int(csv_path.stem)
    user = csv_path.parent.name

    if "label" in df.columns and df["label"].notna().any():
        label_vals = df["label"].dropna().unique()
        label = int(label_vals[0]) if len(label_vals) >= 1 else -1
        n_unique_labels = int(df["label"].nunique())
    else:
        label = -1
        n_unique_labels = 0

    n_rows = len(df)
    idx_ok = ("index" in df.columns
              and bool((df["index"].values == np.arange(n_rows)).all()))

    out = {
        "split": split, "user": user, "file_id": file_id, "label": label,
        "n_rows": n_rows, "idx_contiguous": idx_ok,
        "n_unique_labels": n_unique_labels,
        "any_null": bool(df[FEATURE_COLS].isnull().any().any()),
    }

    # ---- Per raw-column basic stats (extends v1: adds percentiles) ----
    for c in FEATURE_COLS:
        v = df[c].values.astype(float)
        add_prefixed(out, c, basic_stats(v))

    # ---- Magnitude series ----
    mean_mag = np.sqrt(df["mean_x"]**2 + df["mean_y"]**2 + df["mean_z"]**2).values
    std_mag = np.sqrt(df["std_x"]**2 + df["std_y"]**2 + df["std_z"]**2).values
    add_prefixed(out, "mean_mag", basic_stats(mean_mag))
    add_prefixed(out, "std_mag", basic_stats(std_mag))

    # ---- v1 stillness features ----
    for c in ["std_x", "std_y", "std_z"]:
        out[f"{c}__frac_zero"] = float((df[c].abs() < 1e-10).mean())

    # ---- v1 first-vs-second-half drift on means ----
    half = n_rows // 2
    for c in ["mean_x", "mean_y", "mean_z"]:
        out[f"{c}__drift"] = float(df[c].iloc[half:].mean() - df[c].iloc[:half].mean())

    # ---- NEW: spectral features on the most informative series ----
    for name, v in [("mean_x", df["mean_x"].values),
                    ("mean_y", df["mean_y"].values),
                    ("mean_z", df["mean_z"].values),
                    ("std_mag", std_mag)]:
        add_prefixed(out, name, spectral_stats(v.astype(float)))

    # ---- NEW: autocorrelation features on same series ----
    for name, v in [("mean_x", df["mean_x"].values),
                    ("mean_y", df["mean_y"].values),
                    ("mean_z", df["mean_z"].values),
                    ("std_mag", std_mag)]:
        add_prefixed(out, name, autocorr_lags(v.astype(float)))

    # ---- NEW: peak features on std_mag (motion-burst count) ----
    add_prefixed(out, "std_mag", peak_stats(std_mag.astype(float)))

    # ---- NEW: cross-axis correlations on the per-second mean signal ----
    mx, my, mz = df["mean_x"].values, df["mean_y"].values, df["mean_z"].values
    def corr_safe(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    out["corr_mean_xy"] = corr_safe(mx, my)
    out["corr_mean_xz"] = corr_safe(mx, mz)
    out["corr_mean_yz"] = corr_safe(my, mz)

    # ---- NEW: stationarity (early-vs-late energy) ----
    out["std_mag__half_ratio"] = (
        float(std_mag[half:].mean() / std_mag[:half].mean())
        if std_mag[:half].mean() > 1e-9 else 1.0
    )
    # variance of energy across thirds — proxy for changing activity
    thirds = [std_mag[:n_rows//3], std_mag[n_rows//3:2*n_rows//3], std_mag[2*n_rows//3:]]
    out["std_mag__thirds_var"] = float(np.var([t.mean() for t in thirds]))

    return out


# -------------------------- folder traversal (same logic as v1) --------------------------

def find_user_root(root: Path, max_depth: int = 3) -> Path:
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
        f"Could not find User_* folders under {root}. Last inspected: {cur}. "
        f"Subdirs: {[p.name for p in cur.iterdir() if p.is_dir()]}"
    )


def walk_split(root: Path, split: str):
    user_root = find_user_root(root)
    if user_root != root:
        print(f"[{split}] descended from {root} to {user_root}", flush=True)
    rows = []
    user_dirs = sorted([p for p in user_root.iterdir()
                        if p.is_dir() and p.name.startswith("User_")])
    print(f"[{split}] found {len(user_dirs)} user folders under {user_root}",
          flush=True)
    n_files = 0
    for ud in user_dirs:
        files = sorted(ud.glob("*.csv"))
        for f in files:
            try:
                rows.append(summarize_file(f, split))
            except Exception as e:
                print(f"  ! failed to read {f}: {e}", flush=True)
            n_files += 1
            if n_files % 1000 == 0:
                print(f"  [{split}] {n_files} files processed", flush=True)
    return rows


def write_meta(df: pd.DataFrame, out_path: Path):
    lines = ["=== HAR Assignment 3 dataset summary (v2) ===\n"]
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        lines.append(f"\n--- split: {split} ---")
        lines.append(f"files: {len(sub)}")
        lines.append(f"users: {sub['user'].nunique()}")
        lines.append(f"feature columns: {len(sub.columns)}")
        lines.append(f"sequence lengths: {sorted(sub['n_rows'].unique())}")
        lines.append(f"any null: {bool(sub['any_null'].any())}")
        if (sub["label"] >= 0).any():
            lines.append("\nfiles per label:")
            lines.append(sub["label"].value_counts().sort_index().to_string())
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}", flush=True)


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

    rows = []
    if args.train_dir:
        rows += walk_split(Path(args.train_dir), "train")
    if args.test_dir:
        rows += walk_split(Path(args.test_dir), "test")

    df = pd.DataFrame(rows)
    out_csv = out_dir / "file_summary_v2.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  shape={df.shape}", flush=True)

    write_meta(df, out_dir / "dataset_meta_v2.txt")


if __name__ == "__main__":
    main()
