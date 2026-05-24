# Interpretable Chest X-Ray Pneumonia Detection

**Grad-CAM & Eigen-CAM Explainability · Transfer Learning · Medical AI**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> End-to-end computer vision pipeline that classifies chest X-rays as **Normal** or **Pneumonia**, with visual heatmaps showing exactly which lung regions drive each prediction.

---

## Overview

Pneumonia is a leading cause of death worldwide. Deep learning models can assist radiologists, but **black-box predictions are not trustworthy in medicine**. This project prioritizes **interpretability** using **Grad-CAM** and **Eigen-CAM** heatmaps overlaid on original X-rays.

| Feature | Description |
|---------|-------------|
| **Backbone** | ResNet50 (primary) + EfficientNet-B0 (optional) |
| **Imbalance** | Weighted sampler + weighted cross-entropy |
| **Metrics** | F1, Precision, Recall, PR-AUC, ROC-AUC |
| **Explainability** | Grad-CAM + Eigen-CAM side-by-side |
| **Splits** | Official Kaggle train / val / test |

> ⚠️ **Disclaimer:** For research and education only. **Not for clinical diagnosis.**

---

## Sample Results

*Run training and evaluation to generate figures in `reports/figures/`.*

Each CAM comparison shows three panels side-by-side:

```
[ Original X-Ray ]  →  [ Grad-CAM Overlay ]  →  [ Eigen-CAM Overlay ]
```

**After training**, view heatmaps in:
- `reports/figures/cam_comparisons/` — correct predictions
- `reports/figures/failure_cases/` — false positives & false negatives
- `reports/figures/confusion_matrix.png`
- `reports/figures/precision_recall_curve.png`

---

## Quick Start

### Option A — GitHub + Colab GPU (recommended)

No Google Drive needed for code. See **[GITHUB_SETUP.md](GITHUB_SETUP.md)** for push instructions, then open **`notebook/chest_xray_github_colab.ipynb`** on [Colab](https://colab.research.google.com) with a GPU runtime.

### Option B — Local

### 1. Clone & install

```bash
cd chest_xray_pneumonia_detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download dataset

[Kaggle — Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

```bash
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json
pip install kaggle
python scripts/download_dataset.py
```

Expected layout:

```
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 3. Train

```bash
python -m src.train --data-root data/raw/chest_xray --backbone resnet50 --epochs 15
```

### 4. Evaluate + generate heatmaps

```bash
python -m src.evaluate --checkpoint models/best_resnet50.pth --data-root data/raw/chest_xray
```

### 5. Interactive demo (optional)

```bash
python demo_app.py
```

### 6. Full narrative notebook

```bash
jupyter notebook notebook/chest_xray_report.ipynb
```

### 7. Train on Google Colab (free cloud GPU)

1. Open [Google Colab](https://colab.research.google.com/)
2. **File → Upload notebook** → select `notebook/chest_xray_colab.ipynb`
3. **Runtime → Change runtime type → GPU** (T4)
4. Run all cells (mount Drive, upload project zip or use Drive copy, Kaggle token, train)

The Colab notebook saves the dataset and models to **Google Drive** so you don't re-download each session. See `notebook/chest_xray_colab.ipynb` for step-by-step cells.

---

## Methodology

### Why ResNet50?

ResNet50 offers an excellent balance of **accuracy**, **training speed**, and **well-understood CAM target layers** (`layer4`). ImageNet pretraining transfers well to chest X-ray texture patterns.

### Class imbalance (~25% Normal, ~75% Pneumonia)

| Strategy | Rationale |
|----------|-----------|
| Weighted Random Sampler | Ensures Normal cases appear equally often during training |
| Weighted Cross-Entropy | Higher penalty for misclassifying the minority class |
| F1 / PR-AUC metrics | Accuracy is misleading on imbalanced medical data |

### Preprocessing

- Grayscale X-rays converted to **3-channel RGB**
- Resize to **224×224**
- **ImageNet normalization** (mean/std) — standard for transfer learning

### Fine-tuning schedule

1. **Epochs 1–4:** Frozen backbone, train classifier head only
2. **Epochs 5–15:** Unfreeze `layer4`, fine-tune with 10× lower learning rate
3. **Checkpoint:** Best model saved by **validation F1** (Pneumonia class)

### Explainability

| Method | How it works | Strength |
|--------|--------------|----------|
| **Grad-CAM** | Gradients × activations for target class | Class-discriminative, widely used |
| **Eigen-CAM** | 1st principal component of activations | Often **cleaner**, less noisy on lung tissue |

---

## Project Structure

```
chest_xray_pneumonia_detection/
├── data/
│   ├── raw/                  # Kaggle dataset (not in git)
│   └── processed/
├── src/
│   ├── dataset.py            # Dataset, transforms, samplers
│   ├── model.py              # ResNet50, EfficientNet-B0
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Metrics + visualizations
│   └── cam.py                # Grad-CAM + Eigen-CAM
├── notebook/
│   └── chest_xray_report.ipynb
├── reports/
│   └── figures/              # Heatmaps, curves, confusion matrix
├── models/                   # Saved checkpoints
├── scripts/
│   └── download_dataset.py
├── demo_app.py               # Gradio demo
├── requirements.txt
└── README.md
```

---

## Results (Template)

*Fill in after running evaluation on your machine:*

| Metric | Value |
|--------|-------|
| Test Accuracy | — |
| Pneumonia F1 | — |
| Pneumonia Precision | — |
| Pneumonia Recall | — |
| PR-AUC | — |
| ROC-AUC | — |

```bash
# View saved metrics
cat reports/figures/test_metrics.json
```

---

## Failure Case Analysis

After evaluation, inspect:

- **`reports/figures/failure_cases/false_negative_*`** — Pneumonia missed (critical clinically)
- **`reports/figures/failure_cases/false_positive_*`** — Normal flagged as Pneumonia

Common failure modes:
- Subtle or early-stage infiltrates
- Other opacities (atelectasis, pleural effusion) confused with consolidation
- Image quality / positioning artifacts

---

## Limitations

1. **Dataset bias** — Pediatric patients, single hospital (Guangzhou)
2. **Binary classification** — Bacterial and viral pneumonia combined
3. **No calibration** — Raw softmax scores are not clinical probabilities
4. **No prospective validation** — Benchmark performance ≠ real-world deployment
5. **Regulatory** — Not a medical device; not for diagnostic use

---

## Compare Backbones (Optional)

```bash
python -m src.train --backbone efficientnet_b0 --epochs 15
python -m src.evaluate --checkpoint models/best_efficientnet_b0.pth
```

---

## Tech Stack

- Python 3.10+, PyTorch, torchvision, timm
- pytorch-grad-cam, scikit-learn, matplotlib, seaborn
- Jupyter, Gradio (demo)

---

## License

MIT License — see [LICENSE](LICENSE).

Dataset: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — review Kaggle terms before use.

---

## Citation

If you use this project academically, cite the original dataset:

> Kermany, Daniel S., et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 2018.
