# Brain Tumor MRI Classification

CS-171 Final Project

**Team Members:**
- Sean Elopre (016139292)
- Ayman Rabia (017889495)
- Harshit Jaglan (016723967)

---

## Overview

MRI scans are the primary diagnostic tool for detecting brain tumors, but manual review is time-consuming for radiologists. This project applies deep learning to automate the multiclass classification of brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

## Dataset

**Brain Tumor MRI Dataset** by Masoud Nickparvar — a combined dataset drawn from Figshare, SARTAJ, and Br35H.

| Class | Train | Test |
|---|---|---|
| Glioma | 1,400 | 400 |
| Meningioma | 1,400 | 400 |
| No Tumor | 1,400 | 400 |
| Pituitary | 1,400 | 400 |
| **Total** | **5,600** | **1,600** |

Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## Notebook

All training, evaluation, and plot generation is contained in a single Colab notebook:

**`BrainTumorMRI_Notebook.ipynb`**

Run on Google Colab with a T4 GPU (`Runtime → Change runtime type → T4`). The notebook trains all three models sequentially, generates all training curves, confusion matrices, and class distribution plots, and saves everything to Google Drive.

---

## Models

Three models are implemented and compared using TensorFlow/Keras:

1. **Custom CNN** — Baseline convolutional network built from scratch to establish a performance floor.
2. **EfficientNetB0** — ImageNet-pretrained model fine-tuned using a two-phase transfer learning strategy.
3. **ResNet50** — Deeper residual network pretrained on ImageNet; uses skip connections to mitigate vanishing gradients.

---

## Preprocessing

All preprocessing is handled inside the notebook via a shared pipeline used by all three models, ensuring a fair apples-to-apples comparison.

| Step | Detail |
|---|---|
| Resize | All images resized to 224×224 (matches ImageNet pretraining size) |
| Normalize | Pixel values scaled from [0, 255] to [0.0, 1.0] |
| Augmentation | Horizontal flip, ±10° rotation, ±10% zoom, ±10% brightness — training set only |
| Train/Val split | 80/20 split from the Training folder (4,480 train / 1,120 validation) |
| Test set | Kept completely separate — only used for final evaluation |
| Batch size | 32 images per batch |
| Seed | 42 (fixed for reproducibility) |

---

## Model Architectures

### Custom CNN
- 4 convolutional blocks: 32 → 64 → 128 → 256 filters
- Each block: Conv2D(relu) → BatchNorm → MaxPooling(2×2) → Dropout(0.25)
- Head: Flatten → Dense(512, relu) → BatchNorm → Dropout(0.5) → Dense(4, softmax)
- Single-phase training, max 30 epochs, EarlyStopping (patience=5)

### EfficientNetB0
- ImageNet-pretrained base with two-phase fine-tuning
- Rescaling(255.0) layer inserted before the base — EfficientNetB0 has built-in normalization expecting [0,255] input; without this the model is stuck at random-chance accuracy (~25%)
- Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(4, softmax)
- Phase 1: base frozen, 10 epochs, lr=0.001
- Phase 2: base unfrozen, 20 epochs, lr=1e-5

### ResNet50
- ImageNet-pretrained 50-layer residual network with two-phase fine-tuning
- Rescaling(255.0) + per-channel ImageNet mean subtraction (Normalization layer)
- Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(256, relu) → BatchNorm → Dropout(0.2) → Dense(4, softmax)
- Phase 1: base frozen, 10 epochs, lr=0.001
- Phase 2: base unfrozen, 20 epochs, lr=1e-5

---

## Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Custom CNN | 84.31% | 0.8380 |
| EfficientNetB0 | 90.56% | 0.9035 |
| **ResNet50** | **93.50%** | **0.9336** |

### Per-Class Results

| Class | CNN P/R/F1 | EfficientNetB0 P/R/F1 | ResNet50 P/R/F1 |
|---|---|---|---|
| Glioma | 0.96 / 0.62 / 0.75 | 0.98 / 0.77 / 0.86 | 0.99 / 0.80 / 0.89 |
| Meningioma | 0.76 / 0.79 / 0.77 | 0.88 / 0.86 / 0.87 | 0.89 / 0.94 / 0.91 |
| No Tumor | 0.78 / 0.99 / 0.87 | 0.88 / 0.99 / 0.94 | 0.91 / 1.00 / 0.95 |
| Pituitary | 0.93 / 0.97 / 0.95 | 0.90 / 0.99 / 0.94 | 0.96 / 1.00 / 0.98 |

### Key Observations
- Glioma had the lowest recall across all three models (62%, 77%, 80%) due to visual similarity with Meningioma — not a class imbalance issue, as all classes are balanced at 25% each
- No Tumor and Pituitary were near-perfect for both pretrained models
- Transfer learning from ImageNet provided significant gains over the from-scratch baseline (~6–9% test accuracy improvement)
- ResNet50 outperformed EfficientNetB0 across all metrics

---

## Evaluation Metrics

All three models are evaluated using:

- Accuracy
- Per-class Precision and Recall
- Macro F1-score (primary metric — treats all classes equally)
- Confusion matrices

---

## Timeline

| Milestone | Date |
|---|---|
| Project Proposal | April 15, 2026 |
| Dataset preprocessing & model development | May 2, 2026 |
| Source code & final report submission | May 11, 2026 |
| Final presentation | May 4, 2026 |
