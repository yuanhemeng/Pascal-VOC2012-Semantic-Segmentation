"""Segmentation metrics and original-resolution validation geometry."""

import torch
from torch.nn import functional as F

from src.data_preprocessing import IGNORE_INDEX, NUM_CLASSES


def restore_logits(logits: torch.Tensor, valid_size: tuple[int, int], original_size: tuple[int, int]) -> torch.Tensor:
    """Remove right/bottom padding, then resize class scores to the raw mask."""
    height, width = valid_size
    logits = logits[..., :height, :width]
    return F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)


class SegmentationMetrics:
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target must have the same shape")
        prediction = prediction.reshape(-1).long().cpu()
        target = target.reshape(-1).long().cpu()
        valid = target != IGNORE_INDEX
        prediction, target = prediction[valid], target[valid]
        if ((prediction < 0) | (prediction >= self.num_classes)).any():
            raise ValueError("Prediction contains an invalid class index")
        if ((target < 0) | (target >= self.num_classes)).any():
            raise ValueError("Target contains an invalid class index")
        counts = torch.bincount(target * self.num_classes + prediction, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self) -> torch.Tensor:
        intersection = self.confusion.diag().double()
        union = self.confusion.sum(0) + self.confusion.sum(1) - self.confusion.diag()
        return (intersection / union.clamp(min=1)).masked_fill(union == 0, float("nan"))

    def mean_iou(self) -> float:
        scores = self.per_class_iou()
        return scores[~torch.isnan(scores)].mean().item() if (~torch.isnan(scores)).any() else float("nan")

    def pixel_accuracy(self) -> float:
        total = self.confusion.sum().item()
        return self.confusion.diag().sum().item() / total if total else float("nan")
