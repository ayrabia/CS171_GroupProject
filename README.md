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
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |
| **Total** | **5,712** | **1,311** |

Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## Models

Three models are implemented and compared using TensorFlow/Keras:

1. **Custom CNN** — Baseline convolutional network built from scratch to establish a performance floor.
2. **EfficientNetB0** — ImageNet-pretrained model fine-tuned for this dataset; chosen for its strong accuracy-to-parameter ratio.
3. **ResNet50** — Deeper residual network pretrained on ImageNet; uses skip connections to mitigate vanishing gradients.

---

## Preprocessing

- Resize all images to a uniform input size
- Normalize pixel values
- Apply data augmentation to improve generalization

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
