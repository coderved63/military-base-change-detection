# Military Base Construction Monitoring — Change Detection

Deep learning system for detecting new structures and infrastructure changes between satellite image pairs. Targets defense applications: military base expansion, runway construction, and infrastructure development monitoring.

## Models

| Model | Backbone | Role | Paper |
|---|---|---|---|
| Siamese CNN | ResNet18 (shared) | Baseline | — |
| UNet++ | ResNet34 (shared) | Mid-tier | [arXiv:1807.10165](https://arxiv.org/abs/1807.10165) |
| ChangeFormer | MiT-B1 (shared) | SOTA | [arXiv:2201.01293](https://arxiv.org/abs/2201.01293) |

## Dataset

**LEVIR-CD** — 637 image pairs at 1024×1024, cropped to 256×256 non-overlapping patches. Contains building change annotations across urban areas.

## Quick Start (Google Colab)

```python
# 1. Setup
from setup_colab import setup
dirs = setup()

# 2. Train
!python train.py --config configs/config.yaml --model siamese_cnn

# 3. Evaluate
!python evaluate.py --config configs/config.yaml --checkpoint checkpoints/siamese_cnn_best.pth

# 4. Resume after disconnect
!python train.py --config configs/config.yaml --model changeformer \
    --resume /content/drive/MyDrive/change-detection/checkpoints/changeformer_last.pth
```

## Local Usage

```bash
# Preprocess dataset
python data/download.py --dataset levir-cd --raw_dir ./raw_data --out_dir ./processed_data

# Train
python train.py --config configs/config.yaml --model unet_pp

# Evaluate
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/unet_pp_best.pth

# Inference on new image pair
python inference.py --before path/to/before.png --after path/to/after.png \
    --model changeformer --checkpoint checkpoints/changeformer_best.pth

# Gradio demo
python app.py
```

## GPU Batch Sizes (Auto-Detected)

| Model | T4 (16GB) | V100 (16GB) | LR |
|---|---|---|---|
| Siamese CNN | 16 | 16 | 1e-3 |
| UNet++ | 8 | 12 | 1e-4 |
| ChangeFormer | 4 | 6 | 6e-5 |

## Evaluation Metrics

- **F1-Score** (primary, used for model selection and early stopping)
- IoU / Jaccard
- Precision, Recall
- Overall Accuracy

## Project Structure

```
military-base-change-detection/
├── configs/config.yaml         # All hyperparameters and paths
├── data/
│   ├── download.py             # Dataset download & patch cropping
│   └── dataset.py              # PyTorch Dataset with synced augmentations
├── models/
│   ├── __init__.py             # get_model() factory
│   ├── siamese_cnn.py          # Siamese CNN baseline
│   ├── unet_pp.py              # UNet++ change detection
│   └── changeformer.py         # ChangeFormer transformer
├── utils/
│   ├── metrics.py              # F1, IoU, Precision, Recall, OA
│   ├── losses.py               # BCEDiceLoss, FocalLoss
│   └── visualization.py        # Plotting utilities
├── train.py                    # Training with AMP, early stopping, resume
├── evaluate.py                 # Test set evaluation
├── inference.py                # Inference on new image pairs
├── app.py                      # Gradio demo
├── setup_colab.py              # Colab environment setup
└── requirements.txt            # Pinned dependencies
```
