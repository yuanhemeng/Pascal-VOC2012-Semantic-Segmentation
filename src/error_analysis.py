"""Pure helpers for semantic-segmentation error analysis."""

from __future__ import annotations

import math

import torch

from src.data_preprocessing import IGNORE_INDEX


VOC_CLASS_NAMES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)


def class_metrics(confusion: torch.Tensor, class_names=VOC_CLASS_NAMES) -> list[dict]:
    """Return per-class metrics from a target-row/prediction-column matrix."""
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if confusion.shape[0] != len(class_names):
        raise ValueError("class_names length does not match confusion matrix")
    confusion = confusion.double()
    rows = []
    for class_id, name in enumerate(class_names):
        true_positive = confusion[class_id, class_id].item()
        support = confusion[class_id].sum().item()
        predicted = confusion[:, class_id].sum().item()
        union = support + predicted - true_positive
        precision = true_positive / predicted if predicted else math.nan
        recall = true_positive / support if support else math.nan
        iou = true_positive / union if union else math.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan

        errors = confusion[class_id].clone()
        errors[class_id] = -1
        confused_id = int(errors.argmax().item()) if errors.numel() > 1 else class_id
        confused_pixels = max(0, int(errors[confused_id].item()))
        rows.append({
            "class_id": class_id,
            "class_name": name,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_pixels": int(support),
            "predicted_pixels": int(predicted),
            "most_confused_with_id": confused_id,
            "most_confused_with": class_names[confused_id],
            "confused_pixels": confused_pixels,
            "confused_fraction": confused_pixels / support if support else math.nan,
        })
    return rows


def sample_class_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_id: str,
    class_names=VOC_CLASS_NAMES,
) -> list[dict]:
    """Measure each ground-truth class present in one validation image."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = target != IGNORE_INDEX
    prediction = prediction[valid].long().cpu()
    target = target[valid].long().cpu()
    rows = []
    for class_id in torch.unique(target).tolist():
        gt = target == class_id
        pred = prediction == class_id
        true_positive = (gt & pred).sum().item()
        support = gt.sum().item()
        predicted = pred.sum().item()
        union = (gt | pred).sum().item()

        wrong_predictions = prediction[gt & ~pred]
        if wrong_predictions.numel():
            counts = torch.bincount(wrong_predictions, minlength=len(class_names))
            confused_id = int(counts.argmax().item())
            confused_pixels = int(counts[confused_id].item())
        else:
            confused_id = class_id
            confused_pixels = 0
        rows.append({
            "sample_id": sample_id,
            "class_id": class_id,
            "class_name": class_names[class_id],
            "gt_pixels": support,
            "predicted_pixels": predicted,
            "true_positive_pixels": true_positive,
            "recall": true_positive / support,
            "iou": true_positive / union if union else math.nan,
            "most_confused_with_id": confused_id,
            "most_confused_with": class_names[confused_id],
            "confused_pixels": confused_pixels,
            "confused_fraction": confused_pixels / support,
        })
    return rows


def select_worst_samples(
    rows: list[dict], target_class_ids: list[int], samples_per_class: int, min_gt_pixels: int,
) -> list[dict]:
    """Select low-recall examples independently for each requested class."""
    selected = []
    for class_id in target_class_ids:
        candidates = [
            row for row in rows
            if row["class_id"] == class_id and row["gt_pixels"] >= min_gt_pixels
        ]
        candidates.sort(key=lambda row: (row["recall"], row["iou"], -row["gt_pixels"], row["sample_id"]))
        for rank, row in enumerate(candidates[:samples_per_class], start=1):
            selected.append({**row, "rank": rank})
    return selected
