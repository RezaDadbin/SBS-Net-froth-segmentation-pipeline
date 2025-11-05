# SBSNet Froth Segmentation

A modular, production-ready PyTorch implementation of **SBSNet** for **froth image segmentation**, including:

- Training and validation pipeline for SBSNet
- Evaluation with IoU and Dice metrics
- Inference to generate soft segmentation maps from a trained model
- A **watershed-based post-processing pipeline** to split and count individual froths

> ⚠️ **Note:** The dataset used to develop this repository is **private** and **not included**.  
> The code is structured so you can plug in your own data in the same format.

---

## 1. Project Overview

Froth flotation is widely used in mineral processing. Analyzing the froth structure (bubble count, size distribution, morphology) can help monitor and optimize process performance.

This repository provides:

1. A PyTorch implementation of **SBSNet** tailored for **froth segmentation**.
2. A pipeline to generate **soft segmentation maps** (normalized logits).
3. A **watershed-based post-processing** stage that:
   - Refines the segmentation mask,
   - Splits merged froth regions,
   - Filters small artifacts,
   - Produces per-froth instance-like objects and counts.

The project is organized as a clean, reproducible Python package and is suitable both for research and for integration into larger systems.

---

## 2. Authors

**Primary Authors**

- **Sina Lotfi**
- **Reza Dadbin**

If you use this repository or results derived from it in academic or industrial work, please credit:

> *Sina Lotfi & Reza Dadbin – SBSNet-based Froth Segmentation Pipeline*

(Feel free to add formal citation info here if you later publish a paper or thesis.)

---

## 3. Repository Structure

```text
SBS-Net/
├─ config.py                  # Central configuration (paths, hyperparams, device, outputs)
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ sbsnet_froth/
│   ├─ __init__.py
│   │
│   ├─ models/
│   │   ├─ __init__.py
│   │   └─ sbsnet/
│   │       ├─ __init__.py
│   │       ├─ SBSNet.py               # Main SBSNet network definition
│   │       ├─ resnet.py               # Backbone / feature extractor
│   │       ├─ sknet.py                # Selective Kernel blocks
│   │       ├─ enhanced_feature_net.py # Enhanced feature fusion
│   │       └─ downsample.py           # Custom downsampling layers
│   │
│   ├─ data/
│   │   ├─ __init__.py
│   │   └─ polygon_dataset.py          # PolygonDataset for .tiff + .json pairs
│   │
│   └─ utils/
│       ├─ __init__.py
│       ├─ metrics.py                  # IoU + Dice
│       └─ postprocess.py              # Watershed-based froth post-processing
│
├─ scripts/
│   ├─ train.py                        # Train + validate SBSNet
│   ├─ eval.py                         # Evaluate IoU / Dice on eval split
│   ├─ predict.py                      # Generate soft segmentation maps (logits)
│   └─ postprocess.py                  # Apply watershed to predicted maps
│
└─ data/                               # LOCAL ONLY (ignored by git)
    ├─ train/
    ├─ val/
    └─ eval/
