# GPU Server Execution Guide — NC-MedAI Reviewer Experiments

**Paper:** *NC-MedAI: Geometry-Aware Training via Fixed Equiangular Tight Frame Heads for Imbalanced Dermoscopic Skin-Lesion Classification*

**Purpose:** Camera-ready revision — executing reviewer-requested experiments on a GPU server after pulling from Git.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setup on the GPU Server](#2-setup-on-the-gpu-server)
3. [Verify the Dataset](#3-verify-the-dataset)
4. [Experiment Overview](#4-experiment-overview)
5. [Execution — All Experiments (Recommended)](#5-execution--all-experiments-recommended)
6. [Execution — Group by Group](#6-execution--group-by-group)
7. [Execution — Individual Experiments](#7-execution--individual-experiments)
8. [Output Directory Structure](#8-output-directory-structure)
9. [Monitoring Progress](#9-monitoring-progress)
10. [Expected Runtime Estimates](#10-expected-runtime-estimates)
11. [Resume After Interruption](#11-resume-after-interruption)
12. [Collecting Results After All Runs Finish](#12-collecting-results-after-all-runs-finish)
13. [Troubleshooting](#13-troubleshooting)
14. [Experiment Specifications (Reference)](#14-experiment-specifications-reference)

---

## 1. Prerequisites

### Required Hardware
- NVIDIA GPU with ≥ 16 GB VRAM (tested on RTX 6000 Ada; A100/V100/A6000 also suitable)
- ≥ 32 GB RAM
- ≥ 60 GB free disk (for checkpoints across 9 runs)

### Required Software
- Python 3.10+
- CUDA 11.8+ (or CUDA 13.0 as used in the original runs)
- Git

---

## 2. Setup on the GPU Server

### Step 1: Clone the repository
```bash
git clone <your-repository-url> NC-MedAI
cd NC-MedAI
```

### Step 2: Create Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` includes PyTorch. If your GPU server has a specific CUDA version, you may need to install a matching PyTorch wheel manually:
> ```bash
> # Example for CUDA 12.1:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> # Then install the rest:
> pip install -r requirements.txt --ignore-requires-python
> ```

### Step 4: Verify GPU is detected
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output (example):
```
CUDA: True | Device: NVIDIA A100 80GB PCIe
```

### Step 5: Verify dataset is present
```bash
ls datasets/HAM10000/
# Expected: HAM10000_metadata.csv  images/
ls datasets/HAM10000/images/ | head -5
# Expected: ISIC_0024306.jpg  ISIC_0024307.jpg  ...
```

> **If the dataset is missing:** Download from Kaggle (`kmader/skin-cancer-mnist-ham10000`) and place images in `datasets/HAM10000/images/` and the CSV at `datasets/HAM10000/HAM10000_metadata.csv`.

### Step 6: Run the configuration validation (no GPU needed, ~2 min)
```bash
python -m experiments.validate_reviewer_configs
```

This checks all 10 configuration correctness groups without running real training. All checks should show `[PASS]` and exit with code 0.

---

## 3. Verify the Dataset

The experiments use **HAM10000 with 10:1 long-tail subsampling**, identical to the original Phase-2 study. The split is determined by `seed` — do **not** change the seed in the dataset loader.

Quick sanity check:
```bash
python -c "
from config import load_config
from data import get_medical_dataloaders
cfg = load_config(overrides=['dataset.name=ham10000','dataset.imbalance_ratio=10','training.num_workers=0','training.batch_size=32'])
cfg['dataset']['num_classes'] = 7
train_loader, val_loader, cw = get_medical_dataloaders(cfg, seed=42)
print('Train batches:', len(train_loader))
print('Val batches:', len(val_loader))
print('Class weights:', cw.tolist())
"
```

Expected output (approximately):
```
Train batches: 63
Val batches: 47
Class weights: [0.1434..., 0.0197..., ...]
```

---

## 4. Experiment Overview

All 9 new experiments share the **identical protocol** as the original Phase-2 study:

| Setting | Value |
|---------|-------|
| Dataset | HAM10000, 10:1 imbalance ratio |
| Backbone | ResNet-18 (ImageNet pretrained) |
| Epochs | 50 |
| Batch size | 64 |
| Optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=1e-4 |
| LR schedule | Cosine annealing, 3-epoch warmup |
| ETF scale | 16.0 |

### Group A — Multi-seed reproducibility (4 runs)

| Key | Method | Seed | Sampling | λ_collapse | λ_align |
|-----|--------|------|----------|-----------|---------|
| A1  | ETF+NC-Reg | 7 | WeightedRandom | 0.01 | 0.01 |
| A2  | ETF+NC-Reg | 123 | WeightedRandom | 0.01 | 0.01 |
| A3  | ETF+NC-Reg+Bal | 7 | ClassBalanced | 0.01 | 0.01 |
| A4  | ETF+NC-Reg+Bal | 123 | ClassBalanced | 0.01 | 0.01 |

> **Reference (seed 42 already exists):** `results/phase2/etf_nc_reg_ham10000_r10_s42/` and `results/phase2/etf_nc_reg_balanced_ham10000_r10_s42/`

### Group B — Component-wise ablation (3 runs, seed 42)

| Key | Method | L_collapse | L_align | Note |
|-----|--------|-----------|---------|------|
| B1  | ETF only | ✗ disabled | ✗ disabled | No NC regularization |
| B2  | ETF + L_collapse | ✓ λ=0.01 | ✗ 0.0 | Collapse term only |
| B3  | ETF + L_align | ✗ 0.0 | ✓ λ=0.01 | Alignment term only |

> **Reference (B4 — full model, seed 42):** `results/phase2/etf_nc_reg_ham10000_r10_s42/`

### Group C — Lambda sensitivity (2 runs, seed 42)

| Key | λ_collapse | λ_align | Note |
|-----|-----------|---------|------|
| C1  | 0.001 | 0.001 | 10× smaller than paper |
| C2  | 0.05  | 0.05  | 5× larger than paper |

> **Reference (λ=0.01, seed 42):** `results/phase2/etf_nc_reg_ham10000_r10_s42/`

---

## 5. Execution — All Experiments (Recommended)

Run all 9 experiments sequentially (safest, uses 1 GPU):

```bash
source venv/bin/activate

nohup python -m experiments.reviewer_experiments \
    --groups A B C \
    --num-workers 8 \
    > study_logs/reviewer_experiments/full_run.log 2>&1 &

echo "PID: $!"
```

To monitor:
```bash
tail -f study_logs/reviewer_experiments/full_run.log
```

The script automatically:
- Skips experiments whose `best_results.json` already exists (safe resume)
- Saves checkpoints every epoch at `checkpoints/reviewer_experiments/`
- Saves all metrics/plots at `results/reviewer_experiments/`
- Appends a combined summary CSV at `results/reviewer_experiments/reviewer_summary.csv`

---

## 6. Execution — Group by Group

Run groups independently (e.g., to parallelize across multiple GPUs):

### Group A only (multi-seed):
```bash
nohup python -m experiments.reviewer_experiments \
    --groups A \
    --num-workers 8 \
    > study_logs/reviewer_experiments/groupA.log 2>&1 &
```

### Group B only (ablations):
```bash
nohup python -m experiments.reviewer_experiments \
    --groups B \
    --num-workers 8 \
    > study_logs/reviewer_experiments/groupB.log 2>&1 &
```

### Group C only (lambda sensitivity):
```bash
nohup python -m experiments.reviewer_experiments \
    --groups C \
    --num-workers 8 \
    > study_logs/reviewer_experiments/groupC.log 2>&1 &
```

---

## 7. Execution — Individual Experiments

To run a specific experiment by key:

```bash
# Single experiment:
python -m experiments.reviewer_experiments --only A1

# Multiple specific experiments:
python -m experiments.reviewer_experiments --only A1 B2 C1

# Force re-run even if results exist:
python -m experiments.reviewer_experiments --only A1 --force
```

### Equivalent manual `train.py` commands

Each experiment can also be run directly. The exact hyperparameters are listed below.

#### A1: ETF+NC-Reg, seed=7
```bash
python train.py --method etf --override \
  dataset.name=ham10000 \
  dataset.imbalance_ratio=10 \
  seed=7 \
  model.head=etf \
  model.backbone=resnet18 \
  training.epochs=50 \
  training.batch_size=64 \
  training.lr=0.01 \
  training.momentum=0.9 \
  training.weight_decay=0.0001 \
  training.warmup_epochs=3 \
  training.lr_schedule=cosine \
  training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 \
  nc_regularization.etf_align_weight=0.01 \
  sampling.strategy=weighted \
  etf.scale=16.0 \
  nc_tracking.enabled=true \
  nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false \
  debug.fast_dev_batches=0 \
  logging.run_tag=etf_nc_reg_ham10000_r10_s7 \
  logging.results_dir=results/reviewer_experiments/multiseed \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/multiseed
```

#### A2: ETF+NC-Reg, seed=123
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=123 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 nc_regularization.etf_align_weight=0.01 \
  sampling.strategy=weighted etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_nc_reg_ham10000_r10_s123 \
  logging.results_dir=results/reviewer_experiments/multiseed \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/multiseed
```

#### A3: ETF+NC-Reg+Bal, seed=7
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=7 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 nc_regularization.etf_align_weight=0.01 \
  sampling.strategy=balanced etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_nc_reg_balanced_ham10000_r10_s7 \
  logging.results_dir=results/reviewer_experiments/multiseed \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/multiseed
```

#### A4: ETF+NC-Reg+Bal, seed=123
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=123 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 nc_regularization.etf_align_weight=0.01 \
  sampling.strategy=balanced etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_nc_reg_balanced_ham10000_r10_s123 \
  logging.results_dir=results/reviewer_experiments/multiseed \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/multiseed
```

#### B1: ETF-only (no NC regularization), seed=42
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=42 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=false \
  nc_regularization.collapse_weight=0.0 nc_regularization.etf_align_weight=0.0 \
  sampling.strategy=weighted etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_only_ham10000_r10_s42 \
  logging.results_dir=results/reviewer_experiments/ablations \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/ablations
```

#### B2: ETF + L_collapse only, seed=42
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=42 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.01 nc_regularization.etf_align_weight=0.0 \
  sampling.strategy=weighted etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_lcollapse_ham10000_r10_s42 \
  logging.results_dir=results/reviewer_experiments/ablations \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/ablations
```

#### B3: ETF + L_align only, seed=42
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=42 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.0 nc_regularization.etf_align_weight=0.01 \
  sampling.strategy=weighted etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_lalign_ham10000_r10_s42 \
  logging.results_dir=results/reviewer_experiments/ablations \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/ablations
```

#### C1: ETF+NC-Reg, λ=0.001, seed=42
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=42 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.001 nc_regularization.etf_align_weight=0.001 \
  sampling.strategy=weighted etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_nc_reg_lambda0001_ham10000_r10_s42 \
  logging.results_dir=results/reviewer_experiments/lambda_sensitivity \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/lambda_sensitivity
```

#### C2: ETF+NC-Reg, λ=0.05, seed=42
```bash
python train.py --method etf --override \
  dataset.name=ham10000 dataset.imbalance_ratio=10 seed=42 \
  model.head=etf model.backbone=resnet18 training.epochs=50 \
  training.batch_size=64 training.lr=0.01 training.momentum=0.9 \
  training.weight_decay=0.0001 training.warmup_epochs=3 \
  training.lr_schedule=cosine training.num_workers=8 \
  nc_regularization.enabled=true \
  nc_regularization.collapse_weight=0.05 nc_regularization.etf_align_weight=0.05 \
  sampling.strategy=weighted etf.scale=16.0 \
  nc_tracking.enabled=true nc_tracking.every_n_epochs=1 \
  tracking.tensorboard=false debug.fast_dev_batches=0 \
  logging.run_tag=etf_nc_reg_lambda005_ham10000_r10_s42 \
  logging.results_dir=results/reviewer_experiments/lambda_sensitivity \
  logging.checkpoint_dir=checkpoints/reviewer_experiments/lambda_sensitivity
```

---

## 8. Output Directory Structure

After all experiments complete, the output tree will look like:

```
results/
└── reviewer_experiments/
    ├── reviewer_summary.csv            ← Combined metrics across all 9 runs
    ├── multiseed/
    │   ├── etf_nc_reg_ham10000_r10_s7/
    │   │   ├── best_results.json       ← Best epoch metrics
    │   │   ├── class_metrics.csv       ← Per-class P/R/F1 (incl. melanoma recall)
    │   │   ├── nc_metrics.csv          ← NC1–NC4 per epoch (50 rows)
    │   │   ├── metrics.csv             ← Final summary row
    │   │   ├── confusion_matrix.png
    │   │   ├── per_class_recall.png
    │   │   ├── config_snapshot.yaml    ← Full resolved config
    │   │   ├── run_info.json           ← Git hash, device, env
    │   │   └── training_summary.json   ← Epoch timings
    │   ├── etf_nc_reg_ham10000_r10_s123/
    │   ├── etf_nc_reg_balanced_ham10000_r10_s7/
    │   └── etf_nc_reg_balanced_ham10000_r10_s123/
    ├── ablations/
    │   ├── etf_only_ham10000_r10_s42/
    │   ├── etf_lcollapse_ham10000_r10_s42/
    │   └── etf_lalign_ham10000_r10_s42/
    └── lambda_sensitivity/
        ├── etf_nc_reg_lambda0001_ham10000_r10_s42/
        └── etf_nc_reg_lambda005_ham10000_r10_s42/

checkpoints/
└── reviewer_experiments/
    ├── multiseed/
    │   ├── etf_nc_reg_ham10000_r10_s7/
    │   │   ├── best_model.pth          ← Best checkpoint
    │   │   ├── latest.pth              ← Most recent epoch
    │   │   └── epoch_N.pth             ← Periodic checkpoints
    │   └── ...
    ├── ablations/
    └── lambda_sensitivity/

study_logs/
└── reviewer_experiments/
    ├── multiseed/
    │   ├── etf_nc_reg_ham10000_r10_s7.log
    │   └── ...
    ├── ablations/
    └── lambda_sensitivity/
```

> **Important:** These directories are entirely separate from `results/phase2/` — the original paper results are **never touched**.

---

## 9. Monitoring Progress

### Live log tail (when using reviewer_experiments.py):
```bash
tail -f study_logs/reviewer_experiments/multiseed/etf_nc_reg_ham10000_r10_s7.log
```

### Check which experiments are done:
```bash
python -c "
from experiments.reviewer_experiments import REVIEWER_EXPERIMENTS, is_complete
for k, exp in REVIEWER_EXPERIMENTS.items():
    status = '✓ DONE' if is_complete(exp) else '  PENDING'
    print(f'  [{k}] {status}  {exp[\"run_tag\"]}')
"
```

### GPU utilization:
```bash
watch -n 2 nvidia-smi
```

### Training loss / epoch progress:
```bash
# Look for "ep X/50" in the log:
grep "ep " study_logs/reviewer_experiments/multiseed/etf_nc_reg_ham10000_r10_s7.log | tail -20
```

---

## 10. Expected Runtime Estimates

Runtime depends on GPU model. Based on original runs on an RTX 6000 Ada (50.86 GB):

| Experiment | Estimated Time |
|-----------|---------------|
| A1–A4 (ETF+NC-Reg variants) | ~27–30 min each |
| B1–B3 (ablations, seed 42) | ~27–30 min each |
| C1–C2 (lambda variants) | ~27–30 min each |
| **Total (9 runs, sequential)** | **~4–4.5 hours** |

> **Tip:** A100/H100 GPUs will be roughly 2× faster. Adjust `--num-workers` based on available CPU cores (8 is a good default).

---

## 11. Resume After Interruption

The script is **idempotent**: if interrupted mid-run, restart from where you left off:

```bash
# Just re-run the same command — completed runs are automatically skipped
python -m experiments.reviewer_experiments --groups A B C --num-workers 8
```

The resume logic checks for `best_results.json` in the output directory. If it exists, the experiment is considered complete and skipped.

For individual `train.py` runs that were interrupted but have a `latest.pth` checkpoint:
```bash
python train.py --method etf --resume checkpoints/reviewer_experiments/multiseed/etf_nc_reg_ham10000_r10_s7/latest.pth \
  --override ... [same overrides as original run]
```

---

## 12. Collecting Results After All Runs Finish

### Quick summary table:
```bash
python -c "
import pandas as pd
from pathlib import Path

# Combined reviewer summary
rev_csv = Path('results/reviewer_experiments/reviewer_summary.csv')
if rev_csv.exists():
    df = pd.read_csv(rev_csv)
    cols = ['key','group','seed','best_val_acc','macro_f1','roc_auc','nc1','nc4','melanoma_recall']
    print(df[cols].sort_values(['group','key']).to_string(index=False))
else:
    print('reviewer_summary.csv not found — run experiments first')
"
```

### Combine with original seed-42 results for multi-seed analysis:
```bash
python -c "
import pandas as pd, json
from pathlib import Path

orig_a = Path('results/phase2/etf_nc_reg_ham10000_r10_s42')
orig_b = Path('results/phase2/etf_nc_reg_balanced_ham10000_r10_s42')

def load_br(p):
    f = p / 'best_results.json'
    cm = p / 'class_metrics.csv'
    if not f.exists(): return None
    d = json.loads(f.read_text())
    if cm.exists():
        df = pd.read_csv(cm)
        row = df[df['class'].str.lower()=='melanoma']
        if not row.empty:
            d['melanoma_recall'] = float(row['recall'].iloc[0])
    return d

seed42_etf     = load_br(orig_a)
seed42_etfbal  = load_br(orig_b)

if seed42_etf:
    print('ETF+NC-Reg seed=42: F1={macro_f1:.4f} Melanoma={melanoma_recall:.4f}'.format(**seed42_etf))
if seed42_etfbal:
    print('ETF+NC-Reg+Bal seed=42: F1={macro_f1:.4f} Melanoma={melanoma_recall:.4f}'.format(**seed42_etfbal))
"
```

### Reconstruct mean ± std table for paper:
```bash
python -c "
import pandas as pd, numpy as np
from pathlib import Path, json

results_dir = Path('results/reviewer_experiments')
phase2_dir  = Path('results/phase2')

multiseed_runs = {
    'ETF+NC-Reg': [
        phase2_dir / 'etf_nc_reg_ham10000_r10_s42',
        results_dir / 'multiseed/etf_nc_reg_ham10000_r10_s7',
        results_dir / 'multiseed/etf_nc_reg_ham10000_r10_s123',
    ],
    'ETF+NC-Reg+Bal': [
        phase2_dir / 'etf_nc_reg_balanced_ham10000_r10_s42',
        results_dir / 'multiseed/etf_nc_reg_balanced_ham10000_r10_s7',
        results_dir / 'multiseed/etf_nc_reg_balanced_ham10000_r10_s123',
    ],
}
# Inspect reviewer_summary.csv for the remaining rows
"
```

---

## 13. Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```bash
python -m experiments.reviewer_experiments --groups A --num-workers 4
# If still OOM, edit reviewer_experiments.py: _BATCH_SIZE = 32
```

### Dataset path not found
Check your paths:
```bash
python -c "
from config import load_config
cfg = load_config()
print(cfg['medical']['ham10000'])
"
```
Edit `config/config.yaml` → `medical.ham10000.csv_path` and `img_dir` to match your server's dataset location.

### Import errors / missing packages
```bash
pip install -r requirements.txt
```

### Validation script reports FAIL
```bash
python -m experiments.validate_reviewer_configs
# Read the [FAIL] lines — they identify exactly which check and why
```

### Run completed but best_results.json is missing
The training may have crashed after the model saved but before post-hoc evaluation. Check the log:
```bash
tail -100 study_logs/reviewer_experiments/multiseed/etf_nc_reg_ham10000_r10_s7.log
```
Then re-run with `--force` to redo that experiment.

---

## 14. Experiment Specifications (Reference)

### Loss function structure

For all ETF experiments, the optimizer sees:

$$L_{\text{total}} = L_{\text{CE}} + \lambda_{\text{collapse}} \cdot L_{\text{collapse}} + \lambda_{\text{align}} \cdot L_{\text{align}}$$

where:
- $L_{\text{CE}}$ = standard cross-entropy (no class weighting for ETF methods)
- $L_{\text{collapse}}$ = within-class feature variance penalty (`NCCollapseRegularizer`)
- $L_{\text{align}}$ = pairwise cosine deviation from ETF ideal penalty (`ETFAlignmentLoss`)

Setting a weight to `0.0` **completely removes that term** (verified by early-exit guards in `training/nc_regularization.py`).

### Seed controls

The seed passed via `--override seed=N` controls:
1. `random.seed(N)` — Python RNG
2. `np.random.seed(N)` — NumPy RNG
3. `torch.manual_seed(N)` — PyTorch CPU RNG
4. `torch.cuda.manual_seed_all(N)` — PyTorch GPU RNG
5. Dataset shuffle (`df.sample(frac=1, random_state=N)`)
6. Long-tail subsampling (`np.random.default_rng(N)`)
7. DataLoader sampler generator (`torch.Generator().manual_seed(N)`)
8. **ETF frame construction** (`torch.randn(D, C)` in `ETFClassifier._init_etf`)
9. `cudnn.deterministic=True`, `cudnn.benchmark=False`

Different seeds produce genuinely different ETF frames (verified by the validation script CHECK 4).

### Exact hyperparameter table

| Parameter | Value | Config key |
|-----------|-------|-----------|
| LR | 0.01 | `training.lr` |
| Batch size | 64 | `training.batch_size` |
| Momentum | 0.9 | `training.momentum` |
| Weight decay | 1e-4 | `training.weight_decay` |
| Epochs | 50 | `training.epochs` |
| Warmup epochs | 3 | `training.warmup_epochs` |
| LR schedule | cosine | `training.lr_schedule` |
| ETF scale | 16.0 | `etf.scale` |
| λ (paper default) | 0.01 | `nc_regularization.collapse_weight` = `nc_regularization.etf_align_weight` |
| Imbalance ratio | 10 | `dataset.imbalance_ratio` |
| Val fraction | 0.15 | hardcoded in `HAM10000Dataset` |
| Test fraction | 0.10 | hardcoded in `HAM10000Dataset` |

---

*Guide prepared for NC-MedAI camera-ready revision — October 2026*
