# SBSNet Froth Segmentation Pipeline

A modular PyTorch implementation of **SBSNet** for industrial froth image segmentation. The repository contains training, evaluation, prediction, and watershed post-processing scripts. Its internal polygon dataset loader is not included in the public source.

## Research Status and Data Availability

This repository forms part of broader froth-image analysis research. Experimental work completed; a data-paper manuscript is currently in preparation.

The research dataset is private/proprietary and is not distributed with this repository. Code is provided for research and reproducibility with compatible, independently supplied data. Exact reproduction of the private experiments also requires their data, splits, configuration, and checkpoints.

> **Public workflow requirement:** `sbsnet_froth.data.PolygonDataset` is not shipped. Training, evaluation, and prediction require a compatible implementation at that import path before the commands below can run.

---

## Table of contents
- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Prepare your dataset](#prepare-your-dataset)
- [Configure the pipeline](#configure-the-pipeline)
- [Run the training & evaluation workflow](#run-the-training--evaluation-workflow)
  - [1. Train](#1-train)
  - [2. Evaluate](#2-evaluate)
  - [3. Predict soft masks](#3-predict-soft-masks)
  - [4. Watershed post-processing](#4-watershed-post-processing)
- [Outputs](#outputs)
- [Reproducibility tips](#reproducibility-tips)
- [Extending the project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [Authors & citation](#authors--citation)

---

## Key features
- **SBSNet model implementation** built with PyTorch and TorchVision.
- **Flexible pipeline scripts** for training, evaluation, inference, and post-processing.
- **Watershed refinement** that splits merged froth blobs, filters small artifacts, and reports pre/post counts.
- **Modular configuration** via `config.py` for datasets, hyperparameters, and device selection.
- **Utility functions** for IoU & Dice metrics, mask analysis, and ROI debugging.

## Repository structure
```
SBS-Net-froth-segmentation-pipeline/
├── README.md
├── config.py                # Global configuration used by all scripts
├── sbsnet_froth/
│   ├── models/              # SBSNet architecture and building blocks
│   └── utils/               # Metrics, watershed post-processing helpers
└── scripts/
    ├── train.py             # Train SBSNet on polygon annotations
    ├── eval.py              # Compute IoU / Dice on the evaluation split
    ├── predict.py           # Export soft segmentation maps as PNGs
    └── postprocess.py       # Apply watershed-based instance refinement
```

> **Polygon dataset loader**
> The scripts expect a `PolygonDataset` implementation exposed as `sbsnet_froth.data.PolygonDataset`. This class is part of the internal tooling used by the authors and is not shipped publicly. Implement a compatible dataset (e.g., wrapping LabelMe-style JSON polygons) that returns image / binary mask pairs resized to `Config.image_size`.

## Getting started

### Prerequisites
- Python 3.9 or newer
- (Optional) CUDA-capable GPU + PyTorch CUDA build
- System packages for OpenCV image codecs (e.g., `libjpeg`, `libpng`)

Install Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision  # select a compatible CUDA build if needed
pip install numpy opencv-python scikit-image matplotlib tqdm
```

Select a compatible PyTorch/TorchVision build using the [official installation instructions](https://pytorch.org/get-started/locally/). Record the installed versions and dataset-loader implementation for reproducibility.

## Prepare your dataset
Organize your dataset under `data/` (by default, inside the repository root). Each split should contain paired image and annotation files:

```
data/
├── train/
│   ├── img_0001.tiff
│   ├── img_0001.json
│   └── ...
├── val/
│   ├── img_0101.tiff
│   ├── img_0101.json
│   └── ...
└── eval/
    ├── img_0201.tiff
    ├── img_0201.json
    └── ...
```

Assumptions:
- Images use a lossless format (TIFF/PNG) and RGB channels.
- JSON annotations follow LabelMe polygon semantics, convertible to binary masks.
- The dataset class converts polygons to binary masks with the same spatial dimensions as the images and returns `(image_tensor, mask_tensor)`.

## Configure the pipeline
All hyperparameters and paths live in [`config.py`](config.py). Key fields:

- `data_root`, `train_dir`, `val_dir`, `eval_dir`: point to your dataset splits.
- `checkpoint_dir`: directory for saving model checkpoints.
- `outputs_root`, `pred_masks_dir`, `postproc_masks_dir`: output folders used during inference.
- `image_size`: `(width, height)` for resizing inputs and masks.
- `batch_size`, `epochs`, `lr`, `weight_decay`: training knobs.
- `device`: `"auto"`, `"cuda"`, or `"cpu"`.
- `pixel_threshold`: probability threshold for turning logits into binary predictions during training validation.

> **Tip:** Ensure your `PolygonDataset` applies the same resizing and augmentations used during training to avoid misalignment.

## Quick instructions: run the model
Want to sanity-check the repository with a pretrained checkpoint or after you finish training? Follow these steps:

1. Place the `.pth` checkpoint you want to use inside the directory referenced by `Config.checkpoint_dir` (defaults to `./checkpoints/`).
2. Update `Config.resume_checkpoint` to match the filename (e.g., `"model_best.pth"`).
3. Point `Config.eval_dir` to the folder containing the images you want to segment.
4. Activate your environment and run:
   ```bash
   python scripts/predict.py
   python scripts/postprocess.py
   ```
   The first command produces raw probability masks, while the second converts them into final, binarized segmentation maps.
5. Collect the post-processed masks from `outputs/postprocessed_masks/` for downstream analysis.

## Run the training & evaluation workflow
All commands below are executed from the repository root with the virtual environment activated.

### 1. Train
```bash
python scripts/train.py
```
- Seeds Python, NumPy, and PyTorch for reproducibility.
- Logs epoch-level losses and pixel accuracy for train/validation splits.
- Saves `model_epoch_{N}.pth` every `Config.save_every` epochs and maintains `model_best.pth` based on validation pixel accuracy.

### 2. Evaluate
```bash
python scripts/eval.py
```
- Loads the checkpoint specified by `Config.resume_checkpoint` (or `model_best.pth` by default).
- Computes mean Intersection over Union (IoU) and Dice scores on `eval_dir`.

### 3. Predict soft masks
```bash
python scripts/predict.py
```
- Generates per-image logits, normalizes them to `[0, 1]`, and exports channel-0 activations as 8-bit grayscale PNGs to `outputs/raw_masks/`.
- Acts on the dataset referenced by `Config.eval_dir`.

### 4. Watershed post-processing
```bash
python scripts/postprocess.py
```
- Loads each PNG from `outputs/raw_masks/`.
- Applies dynamic thresholding, ROI-wise watershed splitting, and small-component filtering.
- Saves refined binary masks to `outputs/postprocessed_masks/` and prints before/after froth counts.

## Outputs
After running the full pipeline you should expect the following directories:

```
checkpoints/
├── model_best.pth
└── model_epoch_XX.pth

outputs/
├── raw_masks/
│   └── mask_0000.png
└── postprocessed_masks/
    └── mask_0000.png
```

Optional debug artifacts from the watershed pipeline can be written to `./debug_roi_watershed/` by enabling the `save_debug` or `visualize` flags when calling `process_image_roi_watershed` directly.

## Reproducibility tips
- Record `Config.seed`, package versions, and device. Seeding reduces randomness but does not guarantee identical runs across platforms.
- Keep `image_size`, `batch_size`, learning rate, and augmentation strategy consistent across runs.
- Track the repository commit hash and dataset snapshot used for each experiment.

## Extending the project
- **Model variants:** add new architectures under `sbsnet_froth/models/` and expose them via `__init__.py`.
- **Augmentations:** wrap `PolygonDataset` or the training script with Albumentations / TorchVision transforms.
- **Logging:** integrate TensorBoard, Weights & Biases, or MLFlow hooks into `scripts/train.py`.
- **Hyperparameter sweeps:** orchestrate with Optuna or Ray Tune by invoking the training script programmatically.
- **Multi-class froth segmentation:** increase `Config.num_classes` and adjust the dataset to produce multi-channel masks.

## Troubleshooting
- **Missing dataset module:** implement and install `sbsnet_froth.data.PolygonDataset` so the scripts can import it.
- **CUDA not available:** set `Config.device = "cpu"` or install the correct CUDA-enabled PyTorch wheel.
- **Checkpoint not found:** ensure `Config.resume_checkpoint` points to an existing `.pth` file or run training first.
- **Blank masks after post-processing:** inspect the intermediate masks with `process_image_roi_watershed(..., visualize=True)` to tune thresholds or `MIN_AREA`.

## Authors & citation
Primary authors: **Sina Lotfi** and **Reza Dadbin**.

If you build upon this work in academic or industrial settings, please acknowledge:

> *Sina Lotfi & Reza Dadbin – SBSNet-based Froth Segmentation Pipeline*

This is a code acknowledgment, not a published-paper citation. The data-paper manuscript is currently in preparation.