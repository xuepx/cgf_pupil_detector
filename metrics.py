"""
Evaluation metrics for pupil detection.
"""
import numpy as np
import time


def compute_iou(mask_pred, mask_gt):
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask_pred > 0, mask_gt > 0).sum()
    union = np.logical_or(mask_pred > 0, mask_gt > 0).sum()
    if union == 0:
        return float('nan')
    return float(intersection / union)


def compute_dice(mask_pred, mask_gt):
    """Compute Dice coefficient (F1 of masks)."""
    pred_sum = (mask_pred > 0).sum()
    gt_sum = (mask_gt > 0).sum()
    if pred_sum + gt_sum == 0:
        return float('nan')
    intersection = np.logical_and(mask_pred > 0, mask_gt > 0).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def compute_center_error(center_pred, center_gt):
    """Euclidean distance between two centers (px). Returns NaN if either is None."""
    if center_pred is None or center_gt is None:
        return float('nan')
    dx = center_pred[0] - center_gt[0]
    dy = center_pred[1] - center_gt[1]
    return float(np.sqrt(dx * dx + dy * dy))


class SpeedTimer:
    """Context manager to measure processing time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
