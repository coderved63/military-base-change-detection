![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Colab%20Pro-F9AB00?logo=googlecolab&logoColor=white)

# Military Base Construction Monitoring — Change Detection

A deep learning system for detecting new structures and infrastructure changes between satellite image pairs. Built for defense applications: **military base expansion detection**, **runway construction monitoring**, and **infrastructure development tracking**.

We implement and compare **three architectures** on the LEVIR-CD benchmark, ranging from a lightweight CNN baseline to a state-of-the-art transformer.

---

## Architecture Overview

| Model | Backbone | Role | Params | Notes |
|---|---|---|---|---|
| **Siamese CNN** | ResNet18 (shared) | Baseline | ~11M | Feature diff + transposed-conv decoder |
| **UNet++** | ResNet34 (shared) | Mid-tier | ~25M | Nested skip connections via SMP |
| **ChangeFormer** | MiT-B1 (shared) | SOTA | ~16M | Hierarchical transformer + MLP decoder |

All models share a common interface:
```python
def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
    # x1: before image [B, 3, 256, 256]
    # x2: after image  [B, 3, 256, 256]
    # returns: raw logits [B, 1, 256, 256]
```

---

## Datasets

### Primary: LEVIR-CD
- **637 image pairs** at 1024x1024, cropped to 256x256 non-overlapping patches
- Folders: `A/` (before), `B/` (after), `label/` (binary change mask)
- Split: train / val / test
- Focus: building construction and demolition

### Secondary: WHU-CD
- Used for cross-dataset generalisation validation
- Single large image pair, tiled into patches

---

## Quick Start

### Local Setup

```bash
# Clone and install
git clone <repo_url>
cd military-base-change-detection
pip install -r requirements.txt

# Download and preprocess LEVIR-CD
python data/download.py --dataset levir-cd --raw_dir ./raw_data --out_dir ./processed_data
```

### Google Colab Setup

```python
# In the first cell of your Colab notebook:
!git clone <repo_url>
%cd military-base-change-detection
from setup_colab import setup
paths = setup()    # mounts Drive, checks GPU, installs deps, creates dirs
```

---

## Training

All hyperparameters are in `configs/config.yaml`. GPU type is auto-detected and batch size / LR / epochs are set from per-model lookup tables.

```bash
# Train Siamese CNN baseline (~3 hrs on T4)
python train.py --config configs/config.yaml --model siamese_cnn

# Train UNet++ (~6 hrs on T4)
python train.py --config configs/config.yaml --model unet_pp

# Train ChangeFormer (~15 hrs on T4 — use resume for multi-session)
python train.py --config configs/config.yaml --model changeformer

# Resume training after Colab disconnect
python train.py --config configs/config.yaml --model changeformer \
    --resume /content/drive/MyDrive/change-detection/checkpoints/changeformer_last.pth
```

### GPU-Specific Settings (Auto-Detected)

| Model | T4 Batch Size | V100 Batch Size | Learning Rate | Epochs |
|---|---|---|---|---|
| Siamese CNN | 16 | 16 | 1e-3 | 100 |
| UNet++ | 8 | 12 | 1e-4 | 100 |
| ChangeFormer | 4 | 6 | 6e-5 | 200 |

### Training Features
- Mixed precision (AMP) with `GradScaler`
- Gradient accumulation (configurable, default 2 for ChangeFormer on T4)
- Gradient clipping (max_norm=1.0)
- CosineAnnealingLR with linear warmup
- Early stopping on validation F1 (patience=15)
- Saves `best.pth` (best val F1) + `last.pth` (every epoch) to Google Drive
- TensorBoard logging: losses, all metrics, sample prediction grids

---

## Evaluation

```bash
# Evaluate on test set
python evaluate.py --config configs/config.yaml \
    --checkpoint checkpoints/unet_pp_best.pth

# With model override and custom output directory
python evaluate.py --config configs/config.yaml \
    --checkpoint checkpoints/changeformer_best.pth \
    --model changeformer --output_dir ./results
```

Outputs: `results.json`, prediction grid (5x4), 20 individual plots, top-10 overlay images ranked by change area.

---

## Inference

Run on arbitrary before/after image pairs of any resolution:

```bash
python inference.py \
    --before path/to/before.png \
    --after path/to/after.png \
    --model changeformer \
    --checkpoint checkpoints/changeformer_best.pth \
    --output outputs/my_analysis
```

Outputs: `change_mask.png` (binary), `overlay.png` (red-tinted changes), plus console printout of percentage area changed.

---

## Gradio Demo

```bash
python app.py
```

Opens a web interface at `localhost:7860` with:
- Before/after image upload
- Model architecture dropdown
- Checkpoint file selector
- Detection threshold slider
- Outputs: change mask, red overlay, change statistics

---

## Expected Results (LEVIR-CD Test Set)

| Model | F1 | IoU | Precision | Recall | OA |
|---|---|---|---|---|---|
| Siamese CNN | TBD | TBD | TBD | TBD | TBD |
| UNet++ | TBD | TBD | TBD | TBD | TBD |
| ChangeFormer | TBD | TBD | TBD | TBD | TBD |

*Results will be populated after training on LEVIR-CD.*

---

## Project Structure

```
military-base-change-detection/
├── configs/
│   └── config.yaml              # All hyperparameters, paths, model selection
├── data/
│   ├── download.py              # Download & preprocess LEVIR-CD / WHU-CD
│   └── dataset.py               # ChangeDetectionDataset with synced augmentations
├── models/
│   ├── __init__.py              # get_model() factory
│   ├── siamese_cnn.py           # Siamese CNN (ResNet18 + transposed-conv decoder)
│   ├── unet_pp.py               # UNet++ (ResNet34 encoder via SMP)
│   └── changeformer.py          # ChangeFormer (MiT-B1 + MLP decoder)
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # ConfusionMatrix, MetricTracker, F1/IoU/Prec/Rec/OA
│   ├── losses.py                # BCEDiceLoss, FocalLoss, get_loss() factory
│   └── visualization.py         # Plotting, overlays, TensorBoard image logging
├── train.py                     # Training (AMP, grad accum, early stopping, resume)
├── evaluate.py                  # Test-set evaluation with full metrics + visualisations
├── inference.py                 # Tiled inference on arbitrary image pairs
├── app.py                       # Gradio web demo
├── setup_colab.py               # Colab setup (Drive, GPU, deps, directories)
├── requirements.txt             # Pinned dependencies
└── README.md
```

---

## Evaluation Metrics

All metrics computed at threshold=0.5 on binary change masks:

| Metric | Description |
|---|---|
| **F1-Score** | Primary metric for model selection and early stopping |
| **IoU (Jaccard)** | Intersection over union of predicted and true change pixels |
| **Precision** | Fraction of predicted changes that are correct |
| **Recall** | Fraction of true changes that are detected |
| **Overall Accuracy** | Fraction of all pixels correctly classified |

---

## References

- **LEVIR-CD**: Chen & Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection", *Remote Sensing*, 2020. [Paper](https://arxiv.org/abs/2107.09244)
- **ChangeFormer**: Bandara & Patel, "A Transformer-Based Siamese Network for Change Detection", *IGARSS*, 2022. [Paper](https://arxiv.org/abs/2201.01293)
- **UNet++**: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation", *DLMIA*, 2018. [Paper](https://arxiv.org/abs/1807.10165)
