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

(You can later add a formal citation here if you publish a thesis or paper.)

---

## 3. Quickstart – How to Run the Code

This is the minimal sequence to go from **fresh clone → training → evaluation → post-processing**.


3.1. Put your data in place
Create this structure locally (not pushed to git):

text
Copy code
data/
├─ train/
├─ val/
└─ eval/
Each of these folders should contain pairs like:

text
Copy code
img_0001.tiff
img_0001.json
img_0002.tiff
img_0002.json
...
.tiff = image
.json = polygon annotations (LabelMe-style)

3.2. Check config.py
Open config.py and make sure at least these are correct:

python
Copy code
train_dir = ROOT / "data" / "train"
val_dir   = ROOT / "data" / "val"
eval_dir  = ROOT / "data" / "eval"

image_size = (512, 512)
batch_size = 2
epochs = 50
You can also change device to "cuda" if you want to force GPU.

3.3. Train SBSNet
From the repo root:


Copy code
python scripts/train.py
This will:

Train on data/train/

Validate on data/val/

Save checkpoints under checkpoints/

model_epoch_XX.pth

model_best.pth (best val pixel accuracy)

3.4. Evaluate (IoU & Dice)
Set in config.py (optional but recommended):


Copy code
resume_checkpoint = ROOT / "checkpoints" / "model_best.pth"
Then run:


Copy code
python scripts/eval.py
This prints mean IoU and Dice over the data/eval/ split.

3.5. Generate soft segmentation maps

Copy code
python scripts/predict.py
This reads images from eval_dir and writes normalized logits (soft maps) to:


Copy code
outputs/raw_masks/mask_0000.png
outputs/raw_masks/mask_0001.png
...
3.6. Run watershed post-processing

Copy code
python scripts/postprocess.py
This reads the soft maps from outputs/raw_masks/, runs the watershed-based separation pipeline, and saves final binary masks to:


Copy code
outputs/postprocessed_masks/
It also prints how many froths were detected before and after splitting.

4. Repository Structure
text
Copy code
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


5. Features
SBSNet Implementation

Modular architecture for semantic segmentation, adapted for froth images.

Polygon-based Dataset Loader

Converts polygon annotations from JSON into binary masks.

Training / Validation Pipeline

Standard PyTorch loop with logging and checkpointing.

Evaluation (IoU & Dice)

Segmentation quality metrics on a separate eval split.

Prediction / Inference

Generates soft segmentation maps from a trained model (matching original notebook behavior).

Watershed Post-processing

Splits merged froths, filters small blobs, and produces final froth counts.

6. Configuration (config.py)
All settings live in config.py.

Most common changes:

Adjust data directories (train_dir, val_dir, eval_dir).

Change image_size, batch_size, epochs, lr for your experiments.

Set resume_checkpoint when evaluating or predicting.

7. Data Format & Layout
Dataset is not included. This section defines how your own data should look.

7.1. Folder Layout
text
Copy code
data/
├─ train/
│   ├─ img_0001.tiff
│   ├─ img_0001.json
│   ├─ img_0002.tiff
│   ├─ img_0002.json
│   └─ ...
├─ val/
│   ├─ ...
└─ eval/
    ├─ ...
Each .tiff has a corresponding .json annotation file with polygons.

7.2. JSON Annotation Format (LabelMe-like)
Example structure:

json
Copy code
{
  "shapes": [
    {
      "label": "froth",
      "points": [[x1, y1], [x2, y2], ..., [xn, yn]]
    }
  ]
}
points define the polygon vertices.

All shapes in data['shapes'] are used to fill the froth mask.

7.3. PolygonDataset Behavior
PolygonDataset:

Loads the .tiff image and converts to RGB.

Applies a center crop of 1080 × 1080.

Renders polygons into a binary mask: 1 = froth, 0 = background.

Resizes image and mask to Config.image_size.

Returns:

image: (3, H, W) float tensor in [0,1]

mask: (1, H, W) float tensor in {0,1}

8. Training :

Copy code
python scripts/train.py
What happens:

Seeds and device setup from Config.

Training & validation datasets created from train_dir and val_dir.

SBSNet instantiated:

python
Copy code
from sbsnet_froth.models import SBSNet
model = SBSNet(num_classes=Config.num_classes).to(device)
Loss: BCEWithLogitsLoss, with the binary mask repeated to match (B, 2, H, W) outputs.

Pixel accuracy computed as:

python
Copy code
preds = (torch.sigmoid(outputs) > Config.pixel_threshold).float()
Checkpoints written to checkpoints/:

model_epoch_XX.pth every Config.save_every epochs

model_best.pth when validation pixel accuracy improves

9. Evaluation (IoU & Dice)
Set:

python
Copy code
resume_checkpoint = ROOT / "checkpoints/model_best.pth"
Run:


Copy code
python scripts/eval.py
What it does:

Loads SBSNet + checkpoint.

Runs on eval_dir.

Gets predicted class map via argmax over channels.

Computes IoU and Dice per image using compute_iou_and_dice.

Prints mean IoU and mean Dice over the eval set.

10. Prediction (Soft Segmentation Maps)

Copy code
python scripts/predict.py
What it does:

Loads images from eval_dir.

Runs SBSNet + checkpoint.

For each image:

Takes the raw logits outputs[b] (shape (C, H, W)),

Normalizes:

python
Copy code
output_image = output_image - output_image.min()
output_image = output_image / output_image.max()
seg = output_image[0]  # channel 0
Saves seg as a grayscale image [0,255].

Outputs go to:

text
Copy code
outputs/raw_masks/mask_0000.png
outputs/raw_masks/mask_0001.png
...
These soft maps are directly compatible with the watershed post-process.

11. Watershed Post-processing

Copy code
python scripts/postprocess.py
What it does:

Reads all predicted soft masks from outputs/raw_masks/.

For each mask:

Dynamically thresholds based on mean and standard deviation.

Extracts connected components.

For each component:

Computes distance transform.

Uses a statistical threshold on distances to find seeds.

If multiple seeds present → runs local watershed to split merged froths.

Filters out small components by minimum area.

Saves final cleaned binary masks to:

text
Copy code
outputs/postprocessed_masks/
Prints:

count_before (pre-watershed blobs),

count_after (final froths),

removed_small_objects.

plot_froth_regions can also be used to visualize labeled froths with bounding boxes and centroids.

12. Reproducibility
Config.seed is used to seed:

Python random

NumPy

PyTorch (CPU & CUDA)

Deterministic operations for cropping, resizing, mask generation for the same data.

To make experiments reproducible:

Fix Config.seed.

Keep training hyperparameters (image_size, batch_size, lr, epochs) fixed.

Use the same dataset and repository commit.

13. Extending the Project
You can extend this repository in several directions:

New Architectures: add more models under sbsnet_froth/models/ and select via config.

Augmentations: integrate Albumentations or torchvision transforms.

Advanced Logging: plug in TensorBoard, Weights & Biases, etc.

Multi-class Segmentation: support multiple froth classes with num_classes > 2.

Hyperparameter Tuning: wrap training in Optuna or similar for automatic tuning.