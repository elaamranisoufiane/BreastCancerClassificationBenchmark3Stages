# Breast Cancer Histopathology Classification — 3-Stage Pipeline

## Architecture Overview

```
stage0_hpo.py        stage1_benchmark.py      stage2_ensemble.py
 ┌──────────┐          ┌──────────────┐          ┌──────────────┐
 │ NSGA-III │          │  5-Fold CV   │          │  Hold-out    │
 │  64 HPO  │──────►   │  320 folds   │──────►   │  Ensemble    │
 │  combos  │          │  Borda rank  │          │  eval        │
 └──────────┘          └──────────────┘          └──────────────┘
   hpo_results.json   benchmark_results.csv    stage2_results.csv
                       top3_models.json
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Layout Expected

```
data/
├── BreaKHis/
│   ├── train/
│   │   ├── benign/      ← filenames contain 40X / 100X / 200X / 400X
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
└── IDC/
    ├── 0/               ← non-IDC patches
    └── 1/               ← IDC-positive patches
```

## Running

```bash
# Stage 0 — HPO (≈ 4-12h depending on GPU)
python stage0_hpo.py

# Preview console output without data
python stage0_hpo.py --preview

# Stage 1 — 5-Fold Benchmark (≈ 2-8h)
python stage1_benchmark.py

python stage1_benchmark.py --preview

# Stage 2 — Final Evaluation
python stage2_ensemble.py

python stage2_ensemble.py --preview
```

## Key Design Decisions

| Concern | Solution |
|---------|----------|
| Dynamic resizing | `INPUT_SIZES` dict keyed by extractor name; injected into every `T.Resize()` call |
| GPU memory | `torch.cuda.empty_cache()` + `gc.collect()` after every combination |
| ETC accuracy | Simple Moving Average over last 10 timed units |
| Crash safety | Stage 0 writes `hpo_results.json` after **every** combination |
| Scientific rigour | BreaKHis `/test` folder never touched until Stage 2 |
| Borda ranking | Per-metric rank sums avoid single-metric bias |

## Feature Extractors & Input Sizes

| Model | Input Size | ImageNet Weights |
|-------|-----------|-----------------|
| ResNet50 | 224×224 | IMAGENET1K_V1 |
| DenseNet121 | 224×224 | IMAGENET1K_V1 |
| EfficientNetB5 | 456×456 | IMAGENET1K_V1 |
| InceptionV3 | 299×299 | IMAGENET1K_V1 |

## Classifiers

| Classifier | Config |
|-----------|--------|
| SVM | RBF kernel, probability=True |
| RandomForest | 200 trees, random_state=42 |
| XGBoost | 200 estimators, hist tree method |
| Logistic Regression | L2 penalty, max_iter=1000 |

## Expected Console Output (Stage 2)

```
====================================================================================================
  Model                                    BreaKHis Acc  BreaKHis AUC   IDC Acc   IDC AUC   IDC F1
  --------------------------------------------------------------------------------------------------
  Top 1: 400X-EfficientNetB5-SVM                 96.80%       0.9912    91.20%    0.9541   0.9140
  Top 2: 400X-EfficientNetB5-LR                  95.40%       0.9874    89.70%    0.9412   0.8970
  Top 3: 200X-DenseNet121-XGBoost                94.10%       0.9801    88.30%    0.9278   0.8830
  --------------------------------------------------------------------------------------------------
  Ensemble (Top 3)                               98.40%       0.9967    93.90%    0.9731   0.9390
====================================================================================================
```
