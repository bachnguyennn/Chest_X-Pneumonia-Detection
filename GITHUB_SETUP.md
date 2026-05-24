# GitHub + Colab Setup

Train on a free GPU by cloning this repo in Colab — **no Google Drive mount needed for code**.

---

## Step 1: Create GitHub repo

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `chest_xray_pneumonia_detection`
3. **Public** (required for free Colab clone without tokens)
4. Do **not** add README / .gitignore (this project already has them)
5. Click **Create repository**

---

## Step 2: Push code from your Mac

Your repo: **https://github.com/bachnguyennn/Chest_X-Pneumonia-Detection**

```bash
cd "/Users/Bach/Documents/OTU WINTER 2026/Project 3/Chext_X Pneumonia detection/chest_xray_pneumonia_detection"

git add .
git commit -m "Initial commit: pneumonia detection with Grad-CAM and Eigen-CAM"
git branch -M main
git remote add origin https://github.com/bachnguyennn/Chest_X-Pneumonia-Detection.git
git push -u origin main
```

If `git init` says "already initialized", skip that line.

---

## Step 3: Train on Colab

**Browser (recommended):**

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub**
3. Paste: `bachnguyennn/Chest_X-Pneumonia-Detection`
4. Open `notebook/chest_xray_github_colab.ipynb`
5. **Runtime → Change runtime type → GPU**
6. Run all cells (repo URL is pre-configured)

**Cursor + Colab extension:**

1. Open `notebook/chest_xray_github_colab.ipynb`
2. Kernel → **Colab** → **GPU**
3. Set `GITHUB_USER` in cell 2 → run all cells

---

## What gets cloned vs downloaded

| Item | Source |
|------|--------|
| `src/`, notebooks, scripts | **GitHub** (clone) |
| Chest X-ray dataset (~1.2 GB) | **Kaggle** (notebook cell) |
| Trained model + figures | Saved under `/content/chest_xray_pneumonia_detection/` |

Download `best_resnet50.pth` and `reports/figures/` before the session ends, or mount Drive only at the end to copy results.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Repository not found` | Repo must be **public**, or use a personal access token |
| Kaggle download fails | Accept dataset rules + upload `kaggle.json` |
| CUDA OOM | Set `BATCH_SIZE = 8` in the notebook |
