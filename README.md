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

## Models

Three models are implemented and compared using TensorFlow/Keras:

1. **Custom CNN** — Baseline convolutional network built from scratch to establish a performance floor.
2. **EfficientNetB0** — ImageNet-pretrained model fine-tuned for this dataset; chosen for its strong accuracy-to-parameter ratio.
3. **ResNet50** — Deeper residual network pretrained on ImageNet; uses skip connections to mitigate vanishing gradients.

---

## Preprocessing

All preprocessing is handled in `preprocessing.py` via a shared `get_data_loaders()` function imported by all three models, ensuring a fair apples-to-apples comparison.

| Step | Detail |
|---|---|
| Resize | All images resized to 224×224 (matches EfficientNetB0 and ResNet50 ImageNet pretraining size) |
| Normalize | Pixel values scaled from [0, 255] to [0.0, 1.0] |
| Augmentation | Horizontal flip, ±10° rotation, ±10% zoom, ±10% brightness — training set only |
| Train/Val split | 80/20 split from the Training folder (4,480 train / 1,120 validation) |
| Test set | Kept completely separate — only used for final evaluation |
| Batch size | 32 images per batch |

**Verified output:**
```
Classes: ['glioma', 'meningioma', 'notumor', 'pituitary']
Training batches : 140  (4480 images)
Validation batches: 35  (1120 images)
Test batches     : 50  (1600 images)
Batch shape: (32, 224, 224, 3) — Pixel range: [0.0, 1.0]
```

---

## Results

| Model | Val Accuracy | Val Loss | Epochs | Status |
|---|---|---|---|---|
| Custom CNN | 93.48% | 0.2193 | 21 | Done |
| EfficientNetB0 | — | — | — | Pending |
| ResNet50 | — | — | — | Pending |

### Custom CNN
- Built from scratch with 4 conv blocks (32 → 64 → 128 → 256 filters)
- Training accuracy: 97.52% / Validation accuracy: 93.48%
- Ran 21 epochs (~50 min on CPU), EarlyStopping triggered at epoch 21
- ~4% gap between train and val accuracy indicates minor overfitting, expected for a model with no pretrained weights
- Saved to `custom_cnn_model.keras`

---

## Evaluation Metrics

All three models are evaluated using:

- Accuracy
- Per-class Precision and Recall
- Macro F1-score (primary metric — accounts for class imbalance and differing clinical costs of misclassification)
- Confusion matrices

---

## Objectives

1. Preprocess the dataset for model training
2. Implement and train all three deep learning models
3. Compare model performance across all evaluation metrics
4. Identify the best-performing architecture as a potential clinical decision support tool
5. Analyze misclassification patterns across tumor types

---

## Timeline

| Milestone | Date |
|---|---|
| Project Proposal | April 15, 2026 |
| Dataset preprocessing & model development | Ongoing |
| Source code & final report submission | May 11, 2026 |
| Final presentation | TBD |
