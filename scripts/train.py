# scripts/train.py
import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config as C
from sbsnet_froth.data import PolygonDataset
from sbsnet_froth.models import SBSNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if C.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(C.device)


def main():
    # ---- setup ----
    set_seed(C.seed)
    device = get_device()
    print("Using device:", device)

    C.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- transforms ----
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # ---- datasets & loaders ----
    train_dataset = PolygonDataset(
        image_folder=str(C.train_dir),
        transform=transform,
        image_size=C.image_size,
    )
    val_dataset = PolygonDataset(
        image_folder=str(C.val_dir),
        transform=transform,
        image_size=C.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=C.batch_size,
        shuffle=C.shuffle_train,
        num_workers=C.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=C.batch_size,
        shuffle=C.shuffle_val,
        num_workers=C.num_workers,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ---- model, optimizer, loss ----
    model = SBSNet(num_classes=C.num_classes).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=C.lr,
        weight_decay=C.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = C.epochs
    best_val_acc = 0.0

    # ---- training loop ----
    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        running_loss_train = 0.0
        correct_pixels_train = 0
        total_pixels_train = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            images = images.to(device)
            masks = masks.to(device)  # (B,1,H,W)
            masks_exp = masks.repeat(1, C.num_classes, 1, 1)  # (B,2,H,W)

            optimizer.zero_grad()
            outputs = model(images)

            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > C.pixel_threshold).float()
                correct_pixels_train += (preds == masks_exp).sum().item()
                total_pixels_train += masks_exp.numel()

            loss = criterion(outputs, masks_exp)
            running_loss_train += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = running_loss_train / len(train_loader)
        train_acc = correct_pixels_train / total_pixels_train

        # VAL
        model.eval()
        running_loss_val = 0.0
        correct_pixels_val = 0
        total_pixels_val = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"):
                images = images.to(device)
                masks = masks.to(device)
                masks_exp = masks.repeat(1, C.num_classes, 1, 1)

                outputs = model(images)

                preds = (torch.sigmoid(outputs) > C.pixel_threshold).float()
                correct_pixels_val += (preds == masks_exp).sum().item()
                total_pixels_val += masks_exp.numel()

                loss = criterion(outputs, masks_exp)
                running_loss_val += loss.item()

        val_loss = running_loss_val / len(val_loader)
        val_acc = correct_pixels_val / total_pixels_val

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # checkpoints
        if (epoch + 1) % C.save_every == 0:
            ckpt_path = C.checkpoint_dir / f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = C.checkpoint_dir / "model_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best model (val_acc={val_acc:.4f}), saved to {best_path}")


if __name__ == "__main__":
    main()
