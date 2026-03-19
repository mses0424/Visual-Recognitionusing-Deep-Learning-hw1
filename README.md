# NYCU Computer Vision 2026 HW1

- **Student ID:** 111550136
- **Name:** 連家堯

---

## Introduction

This project implements an image classification pipeline for a 100-class dataset using a fine-tuned **ResNet-101** backbone pretrained on ImageNet.

Key design choices:
- **Data augmentation**: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomErasing
- **Training tricks**: Mixup + CutMix (50/50 random per batch), label smoothing (0.1)
- **Optimizer**: AdamW with differential learning rates (backbone 0.1×, head 1×)
- **Scheduler**: Linear warmup (5 epochs) → cosine decay

---

## Environment Setup

Python **3.9+** is required. Install all dependencies with:

```bash
pip install torch torchvision torchinfo numpy matplotlib seaborn scikit-learn tqdm Pillow
```

---

## Usage

This project is run on **Google Colab**.

### Mount Google Drive

Mount your Google Drive to save weights and results:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = '/content/drive/MyDrive/hw1'
os.makedirs(DRIVE_DIR, exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/weights', exist_ok=True)
os.makedirs(f'{DRIVE_DIR}/results', exist_ok=True)

print(f"Drive mounted. Working dir: {DRIVE_DIR}")
```

### Download Dataset

Run the following in a Colab cell to download and extract the dataset:

```python
!pip install -q gdown
!gdown --id 1vxiXJHUo6ZPGxBGXwrsSut0pqfJ6HN9D -O cv_hw1_data.tar
!tar -xf cv_hw1_data.tar
print("Data ready!")
```

### Training

Run all cells in the notebook sequentially. The training will:
1. Load the dataset from `data/train`, `data/val`, `data/test`
2. Train ResNet-101 for 40 epochs
3. Save the best model to `weights/best_resnet101.pth`
4. Output training curves to `results/loss_accuracy.png`
5. Output confusion matrix to `results/confusion_matrix.png`
6. Save test predictions to `prediction.csv`

---

## Performance Snapshot

![Leaderboard Screenshot](leaderboard.png)

### Confusion Matrix

![Confusion Matrix](confusion martix.png)
