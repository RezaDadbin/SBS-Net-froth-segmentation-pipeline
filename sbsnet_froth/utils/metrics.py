# sbsnet_froth/utils/metrics.py
import numpy as np


def compute_iou_and_dice(pred, target, smooth: float = 1e-6):
    """
    pred, target: numpy arrays (H, W), binary 0/1
    returns: (iou, dice)
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return iou, dice
