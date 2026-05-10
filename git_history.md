# Git Commit History
## Brain Tumor MRI Classification — CS171 Final Project
**Sean Elopre · Ayman Rabia · Harshit Jaglan**
**Repository:** github.com/ayrabia/CS171_GroupProject

---

## Phase 1 — Project Setup (April 27, 2026)

**`a99b969` — Add README for brain tumor MRI classification project**
Date: April 27, 2026

Initial project scaffold. Defined the problem statement, selected the Brain Tumor MRI
Dataset (Masoud Nickparvar, Kaggle), and outlined the four classification classes:
Glioma, Meningioma, No Tumor, Pituitary.

Files added: `README.md`

---

**`0ee5cf0` — Add dataset download script and gitignore**
Date: April 27, 2026

Added automated Kaggle download script and configured `.gitignore` to exclude the
dataset folder (data/) and model weights from version control.

Files added: `download_dataset.py`, `.gitignore`

---

## Phase 2 — Preprocessing & Baseline Model (April 30, 2026)

**`6d9d64c` — Add preprocessing pipeline and update README**
Date: April 30, 2026

Implemented the shared preprocessing pipeline used by all three models:
- Resize all images to 224×224 (matches ImageNet pretraining dimensions)
- Normalize pixel values from [0, 255] to [0.0, 1.0]
- Apply augmentation (horizontal flip, ±10° rotation, ±10% zoom, ±10% brightness)
  to training set only
- 80/20 train/validation split from Training folder; test set kept completely separate
- Fixed random seed (42) for reproducibility

Files added: `preprocessing.py`

---

**`b16013e` — Add Custom CNN model and update README with results**
Date: April 30, 2026

Built the from-scratch CNN baseline:
- 4 convolutional blocks (32 → 64 → 128 → 256 filters)
- Each block: Conv2D(ReLU) → BatchNorm → MaxPool(2×2) → Dropout(0.25)
- Head: Flatten → Dense(512) → BatchNorm → Dropout(0.5) → Dense(4, softmax)
- Trained with Adam (lr=0.001), EarlyStopping (patience=5), ReduceLROnPlateau
- Result: 84.31% test accuracy, Macro F1 = 0.8380

Files added: `custom_cnn.py`

---

## Phase 3 — Transfer Learning Models (May 1–3, 2026)

**`9d174a2` — Add EfficientNetB0 transfer learning model with two-phase training**
Date: May 1, 2026

Implemented EfficientNetB0 with two-phase fine-tuning strategy:
- Phase 1: Freeze pretrained base, train only new classification head (10 epochs, lr=0.001)
- Phase 2: Unfreeze full network, fine-tune end-to-end (20 epochs, lr=1e-5)
- Fixed critical input normalization bug: EfficientNetB0 expects [0, 255] input but the
  shared pipeline outputs [0, 1]. Inserted a Rescaling(255.0) layer before the base —
  without this fix the model was stuck at ~25% accuracy (random chance).
- Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(128, relu) → Dropout(0.2) → Dense(4, softmax)
- Result: 90.56% test accuracy, Macro F1 = 0.9035

Files added: `efficientnet_model.py`

---

**`22242c9` — Merge pull request #1 from ayrabia/ayman-branch**
Date: May 1, 2026

Merged preprocessing pipeline, Custom CNN, and EfficientNetB0 into main branch.

---

**`c536c04` — Add ResNet50 model, evaluation script, and confusion matrices**
Date: May 3, 2026

Implemented ResNet50 with the same two-phase transfer learning strategy:
- Preprocessing chain: Rescaling(255.0) → Normalization(mean=[123.68, 116.779, 103.939])
  to match ResNet50's exact ImageNet input distribution
- Head: GlobalAveragePooling2D → Dropout(0.3) → Dense(256, relu) → BatchNorm →
  Dropout(0.2) → Dense(4, softmax)
- Phase 2 EarlyStopping triggered at epoch 17 (confirmed genuine convergence)
- Result: 93.50% test accuracy, Macro F1 = 0.9336 (best across all three models)

Added evaluation script computing accuracy, macro F1, per-class precision/recall,
and confusion matrices for all three models side-by-side.

Files added: `resnet50_model.py`, `evaluate.py`, `confusion_matrices.png`

---

**`9f7d890` — Update README with final test results for all three models**
Date: May 3, 2026

Documented final test results for all three models in the README.

---

**`958f349` — Update timeline with completed dates**
Date: May 3, 2026

---

**`8cf7838` — Fix dataset table counts to match actual downloaded data**
Date: May 3, 2026

Corrected README dataset table: the Kaggle page listed uneven class sizes, but the
actual downloaded dataset has exactly 1,400 train and 400 test images per class
(5,600 training / 1,600 test total — perfectly balanced).

---

**`f294228` — Merge pull request #2 from ayrabia/ayman-branch**
Date: May 3, 2026

Merged ResNet50, evaluation script, and final results into main branch.

---

## Phase 4 — Consolidation into Single Notebook (May 10, 2026)

At this stage, the project consisted of five separate Python scripts
(`download_dataset.py`, `preprocessing.py`, `custom_cnn.py`, `efficientnet_model.py`,
`resnet50_model.py`, `evaluate.py`) developed and tested locally. To ensure consistency
in results and environment, all code was consolidated into a single Google Colab notebook
(`BrainTumorMRI_Notebook.ipynb`) and retrained end-to-end on a T4 GPU.

**Motivation for the switch:**
- Local training used TensorFlow 2.13 (CPU, Apple Silicon); Colab uses TF 2.20 (GPU)
- Version differences caused non-deterministic behavior in the Custom CNN, making
  local results unreliable for final reporting
- A single notebook guarantees all three models are trained on identical data splits,
  the same environment, and the same random seed — ensuring a fair comparison
- All plots (training curves, confusion matrices, class distribution) are generated
  inline in the notebook for full reproducibility

---

**`e875333` — Finalized Procedure**
Date: May 10, 2026

Uploaded the final trained Colab notebook (`BrainTumorMRI_Notebook.ipynb`) containing
all training outputs, evaluation results, and plots inline. This notebook was run on
Google Colab with a T4 GPU (TensorFlow 2.20) and represents the definitive results
reported in the final submission.

Files added: `BrainTumorMRI_Notebook.ipynb`

---

**`8abcc01` — Add instructor comments to notebook; update README and repo structure**
Date: May 10, 2026

- Added explanatory comments to all 17 code cells covering: seeds and reproducibility,
  preprocessing design decisions, transfer learning strategy, EfficientNetB0/ResNet50
  input normalization contracts, callback rationale, and evaluation methodology
- Updated README with final test results, per-class precision/recall/F1, and
  architecture details for all three models
- Added class_distribution.png and sample_images.png for the written report
- Removed standalone .py model files — all code now lives in the notebook

Files modified/removed: `BrainTumorMRI_Notebook.ipynb`, `README.md`,
`custom_cnn.py` (deleted), `efficientnet_model.py` (deleted),
`resnet50_model.py` (deleted), `preprocessing.py` (deleted),
`download_dataset.py` (deleted), `evaluate.py` (deleted)

---

**`71d7a01` — Remove plot_samples.py; update notebook**
Date: May 10, 2026

Removed `plot_samples.py` (local-only utility script, not part of the Colab workflow).
Final notebook updated with any remaining edits before submission.

---

## Summary of Project Evolution

| Phase | Date | Milestone |
|---|---|---|
| Setup | Apr 27 | Dataset identified, repo initialized |
| Preprocessing | Apr 30 | Shared pipeline built and verified |
| Custom CNN | Apr 30 | Baseline model: 84.31% / F1 0.8380 |
| EfficientNetB0 | May 1 | Transfer learning: 90.56% / F1 0.9035 |
| ResNet50 | May 3 | Best model: 93.50% / F1 0.9336 |
| Consolidation | May 10 | All code moved to single Colab notebook for consistency |
| Finalization | May 10 | Comments added, README updated, repo cleaned up |
