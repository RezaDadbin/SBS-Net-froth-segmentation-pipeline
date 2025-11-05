# scripts/predict.py
import sys
from pathlib import Path

# --- add project root to path ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import Config as C
from sbsnet_froth.data import PolygonDataset
from sbsnet_froth.models import SBSNet


def get_device():
    if C.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(C.device)


def main():
    device = get_device()
    print("Using device:", device)

    # same transform as training
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # use eval_dir for inference, same as data_loader_report
    dataset = PolygonDataset(
        image_folder=str(C.eval_dir),
        transform=transform,
        image_size=C.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=C.batch_size,
        shuffle=False,
        num_workers=C.num_workers,
    )

    # where to save the maps
    C.pred_masks_dir.mkdir(parents=True, exist_ok=True)

    # model + checkpoint
    model = SBSNet(num_classes=C.num_classes).to(device)
    ckpt_path = C.resume_checkpoint or (C.checkpoint_dir / "model_best.pth")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # --- this mirrors your notebook loop ---
    idx_global = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Predict"):
            images = images.to(device)
            outputs = model(images)              # (B, C, H, W)

            for b in range(outputs.size(0)):
                output_image = outputs[b].detach().cpu().numpy()  # (C, H, W)

                # normalize whole tensor to [0,1]
                output_image = output_image - output_image.min()
                max_val = output_image.max()
                if max_val > 0:
                    output_image = output_image / max_val

                # use channel 0, like output_image[0] in your notebook
                seg = output_image[0]            # (H, W) in [0,1]
                seg_255 = (seg * 255).astype("uint8")

                out_path = C.pred_masks_dir / f"mask_{idx_global:04d}.png"
                cv2.imwrite(str(out_path), seg_255)
                idx_global += 1

    print(f"Saved {idx_global} segmentation maps to {C.pred_masks_dir}")


if __name__ == "__main__":
    main()
