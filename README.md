# NYCU Data Mining Spring 2026 — Assignment 3

Human Activity Recognition from accelerometer statistics. 6-class classification over 5-minute sequences of 1-Hz aggregated triaxial accelerometer data.

**Public leaderboard macro F1: 0.7784** (Baseline 3: 0.7088).

## Final pipeline

The final submission is a probability-average ensemble of two component models, weighted 0.55 / 0.45:

- **LightGBM** trained on 131 hand-engineered per-file features (aggregate + temporal-pattern statistics)
- **Small 1D-CNN** (~37k parameters) trained on the raw 300×8 sequence input

Both components use the same 5-fold user-disjoint cross-validation, defined by `fold_assignments.csv`.

## Repository layout

```
.
├── README.md
├── eda_summary.py            # v1 feature extractor (38 features per file)
├── eda_summary_v2.py         # v2 feature extractor (131 features, used by final pipeline)
├── baseline_rf.py            # Random Forest baseline; produces fold_assignments.csv
├── baseline_lgbm.py          # LightGBM baseline (final tabular component)
├── baseline_cnn.py           # 1D-CNN small (final sequence component)
├── baseline_cnn_v2.py        # CNN with focal loss + augmentation (ablation)
├── baseline_cnn_wide.py      # Wider CNN (ablation)
├── baseline_cnn_bilstm.py    # CNN-BiLSTM hybrid (ablation)
└── ensemble_generic.py       # Probability-average ensemble (2/3/4-way)
```

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, scipy
- lightgbm (for `baseline_lgbm.py`)
- torch (for any CNN script)

```bash
pip install pandas numpy scikit-learn scipy lightgbm torch
```

A GPU is recommended but not required for the CNN scripts. Training without a GPU will take roughly 5× longer.

## Data layout expected

The scripts expect the dataset structured as in the Kaggle release:

```
data/
├── train/
│   └── train/
│       ├── User_001/
│       │   ├── 00001.csv
│       │   └── ...
│       └── User_060/
└── test/
    └── test/
        ├── User_061/
        └── ...
```

The feature-extraction and CNN scripts auto-descend through the doubly-nested `train/train/` wrapper, so passing either `data/train` or `data/train/train` as `--train_dir` will work.

## Reproduction — full pipeline (final submission)

The following six commands reproduce the public leaderboard score of 0.7784. Total wall-clock time on a Colab T4 GPU is approximately 90 minutes.

### Step 1 — Extract v1 features and create the fold assignment

```bash
python eda_summary.py \
    --train_dir data/train \
    --test_dir  data/test \
    --out_dir   eda_out
```

Produces `eda_out/file_summary.csv`.

### Step 2 — Run RF on v1 features, then on v2 features

```bash
# RF on v1 features (produces fold_assignments.csv)
python baseline_rf.py \
    --summary eda_out/file_summary.csv \
    --out_dir baseline_rf_out
```

This step's primary purpose is to produce `baseline_rf_out/fold_assignments.csv`. **This is the locked-in cross-validation split used by every subsequent script.** Do not regenerate it from this point on — every later experiment reuses this file.

### Step 3 — Extract v2 features (used by the final LightGBM)

```bash
python eda_summary_v2.py \
    --train_dir data/train \
    --test_dir  data/test \
    --out_dir   eda_out
```

Produces `eda_out/file_summary_v2.csv`.

A second RF run was performed on the v2 feature set for the ablation in Section 5.1 of the report. It uses the same fold assignments (since `GroupKFold` keyed on `user` produces identical splits regardless of feature columns):

```bash
# RF on v2 features (used only for the v1 vs v2 ablation row)
python baseline_rf.py \
    --summary eda_out/file_summary_v2.csv \
    --out_dir baseline_rf_out_v2
```

**Note on script behavior:** `baseline_rf.py` writes its own `fold_assignments.csv` into `--out_dir` based on internal `GroupKFold` (not by reading an input fold file). Both runs therefore produce identical `fold_assignments.csv` files. We treat the one from `baseline_rf_out/` as canonical.

### Step 4 — Train the LightGBM component

```bash
python baseline_lgbm.py \
    --summary eda_out/file_summary_v2.csv \
    --folds   baseline_rf_out_v2/fold_assignments.csv \
    --out_dir baseline_out_lgbm_v2
```

Produces:
- `baseline_out_lgbm_v2/oof_predictions_lgbm.csv` (OOF predictions for the train set)
- `baseline_out_lgbm_v2/test_proba_lgbm.csv` (averaged test probabilities across 5 fold-models)
- `baseline_out_lgbm_v2/submission_lgbm.csv` (LGBM-only submission)

### Step 5 — Train the small 1D-CNN component

```bash
python baseline_cnn.py \
    --train_dir data/train \
    --test_dir  data/test \
    --folds     baseline_rf_out_v2/fold_assignments.csv \
    --out_dir   baseline_out_cnn \
    --epochs    40 \
    --patience  8 \
    --batch_size 64 \
    --num_workers 2
```

Produces `baseline_out_cnn/oof_predictions_cnn.csv`, `test_proba_cnn.csv`, and `submission_cnn.csv`.

### Step 6 — Generate the final ensemble submission

```bash
python ensemble_generic.py \
    --oof    baseline_out_lgbm_v2/oof_predictions_lgbm.csv,baseline_out_cnn/oof_predictions_cnn.csv \
    --test   baseline_out_lgbm_v2/test_proba_lgbm.csv,baseline_out_cnn/test_proba_cnn.csv \
    --names  lgbm,cnn \
    --out_dir ensemble_out
```

The final submission file is `ensemble_out/submission_ensemble.csv`. The script also reports the OOF macro F1 of every blend weight from 0.0 to 1.0 in steps of 0.05; the optimal blend (0.55 / 0.45) corresponds to OOF macro F1 = 0.7223.

## Reproduction — ablation models

Each ablation row in the report corresponds to one additional script invocation, all reusing the same `fold_assignments.csv` and `file_summary_v2.csv`:

```bash
# CNN with focal loss + augmentation
python baseline_cnn_v2.py \
    --train_dir data/train --test_dir data/test \
    --folds baseline_rf_out_v2/fold_assignments.csv \
    --out_dir baseline_out_cnn_v2 \
    --epochs 50 --patience 10 --focal_gamma 2.0

# Wider CNN
python baseline_cnn_wide.py \
    --train_dir data/train --test_dir data/test \
    --folds baseline_rf_out_v2/fold_assignments.csv \
    --out_dir baseline_out_cnn_wide

# CNN-BiLSTM hybrid
python baseline_cnn_bilstm.py \
    --train_dir data/train --test_dir data/test \
    --folds baseline_rf_out_v2/fold_assignments.csv \
    --out_dir baseline_out_cnn_bilstm
```

For 3-way and 4-way ensembles, pass the corresponding OOF and test_proba files as comma-separated lists to `ensemble_generic.py` (see the report's Section 5.2 Table 7 for the configurations evaluated).

## Hyperparameters used in the final pipeline

**LightGBM (`baseline_lgbm.py`):**
- objective: `multiclass`, num_class: 6, metric: `multi_logloss`
- learning_rate: 0.05, num_leaves: 63, min_data_in_leaf: 20
- feature_fraction: 0.8, bagging_fraction: 0.8, bagging_freq: 5
- lambda_l2: 1.0, sample_weight: balanced (sklearn `compute_class_weight`)
- early stopping: 100 rounds on validation logloss, max 2000 rounds

**1D-CNN small (`baseline_cnn.py`):**
- 3 conv blocks: channels 32 → 64 → 96, kernels 7/5/3, max-pool 2 each
- FC head: 96 → 64 → 6 with ReLU and Dropout 0.2
- Loss: weighted CE with sklearn balanced class weights
- Optimizer: AdamW, lr 1e-3, weight decay 1e-4
- Schedule: CosineAnnealingLR, T_max = 40
- Batch size: 64, max 40 epochs, early stop patience 8 on validation macro F1

**Ensemble (`ensemble_generic.py`):**
- Weight grid: step 0.05 from 0.0 to 1.0, constrained to sum to 1
- Selection criterion: OOF macro F1 maximization
- Final 2-way weights for LGBM + CNN-small: 0.55 / 0.45

## Notes on reproducibility

- All scripts set `random_state=42` (RF, LGBM) or `seed=42` (CNN). Identical hardware should produce identical numbers; minor variation is possible across different GPU drivers due to non-determinism in CUDA convolution algorithms.
- The fold split is deterministic given the same `fold_assignments.csv`. Do not regenerate the fold assignments unless you want to invalidate comparison with the report's tables.
- Test predictions in every model script are the average of five fold-trained models. This costs nothing extra (each fold trains anyway during cross-validation) and gives slightly more robust predictions than a single full-train model.
