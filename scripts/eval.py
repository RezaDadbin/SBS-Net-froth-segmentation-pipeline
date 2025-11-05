# scripts/eval.py
import sys
from pathlib import Path

# make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config as C
from sbsnet_froth.data import PolygonDataset
from sbsnet_froth.models import SBSNet
from sbsnet_froth.utils import compute_iou_and_dice


def get_device():
    if C.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(C.device)


def main():
    device = get_device()
    print("Using device:", device)

    # transforms must match training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    eval_dataset = PolygonDataset(
        image_folder=str(C.eval_dir),
        transform=transform,
        image_size=C.image_size,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=C.batch_size,
        shuffle=False,
        num_workers=C.num_workers,
    )

    print(f"Eval samples: {len(eval_dataset)}")

    # model
    model = SBSNet(num_classes=C.num_classes).to(device)

    # load checkpoint
    ckpt_path = C.resume_checkpoint or (C.checkpoint_dir / "model_best.pth")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    ious = []
    dices = []

    with torch.no_grad():
        for images, masks in tqdm(eval_loader, desc="Eval"):
            images = images.to(device)
            masks = masks.to(device)  # (B,1,H,W)

            outputs = model(images)               # (B,2,H,W)
            preds = torch.argmax(outputs, dim=1)  # (B,H,W)

            preds_np = preds.cpu().numpy()
            masks_np = masks[:, 0].cpu().numpy()  # (B,H,W)

            for p, t in zip(preds_np, masks_np):
                iou, dice = compute_iou_and_dice(p, t)
                ious.append(iou)
                dices.append(dice)

    print(f"Mean IoU:  {np.mean(ious):.4f}")
    print(f"Mean Dice: {np.mean(dices):.4f}")


if __name__ == "__main__":
    main()
