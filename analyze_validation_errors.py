"""Run full VOC2012 validation error analysis for a saved checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.data_preprocessing import IGNORE_INDEX, NUM_CLASSES, VOCPairedDataset
from src.error_analysis import (
    VOC_CLASS_NAMES,
    class_metrics,
    sample_class_metrics,
    select_worst_samples,
)
from src.metrics import SegmentationMetrics, restore_logits
from src.model import build_segmentation_model


VOC_COLORS = np.array([
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
    (0, 64, 128),
], dtype=np.uint8)

SELECTED_FIELDS = [
    "sample_id", "class_id", "class_name", "gt_pixels", "predicted_pixels",
    "true_positive_pixels", "recall", "iou", "most_confused_with_id",
    "most_confused_with", "confused_pixels", "confused_fraction", "rank",
    "visualization",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "VOCdevkit" / "VOC2012")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "experiments" / "finetune_gpu_02" / "best.pth")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "error_analysis" / "finetune_gpu_02")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-classes", default="chair,bicycle,sofa,pottedplant",
                        help="Comma-separated VOC class names or numeric ids")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--min-gt-pixels", type=int, default=64,
                        help="Ignore tiny class instances when selecting visual examples")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=None, help="Debug only: analyze a validation prefix")
    args = parser.parse_args()
    if args.samples_per_class <= 0 or args.min_gt_pixels < 0 or args.progress_every <= 0:
        parser.error("sample/progress arguments must be positive (min-gt-pixels may be zero)")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def parse_target_classes(value: str) -> list[int]:
    lookup = {name: index for index, name in enumerate(VOC_CLASS_NAMES)}
    result = []
    for token in (item.strip().lower() for item in value.split(",")):
        if not token:
            continue
        if token.isdigit():
            class_id = int(token)
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"Class id outside 0..{NUM_CLASSES - 1}: {class_id}")
        elif token in lookup:
            class_id = lookup[token]
        else:
            raise ValueError(f"Unknown VOC class: {token}")
        if class_id not in result:
            result.append(class_id)
    if not result:
        raise ValueError("No target classes were provided")
    return result


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {"model", "config", "epoch", "best_miou"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    return checkpoint


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV columns for empty rows: {path}")
    fieldnames = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, confusion: torch.Tensor, normalize: bool) -> None:
    values = confusion.double()
    if normalize:
        values = values / values.sum(1, keepdim=True).clamp(min=1) * 100
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth\\prediction", *VOC_CLASS_NAMES])
        for index, name in enumerate(VOC_CLASS_NAMES):
            row = values[index].tolist()
            writer.writerow([name, *([f"{value:.6f}" for value in row] if normalize else [int(value) for value in row])])


def colorize(mask: np.ndarray) -> Image.Image:
    visible = mask.copy()
    void = visible == IGNORE_INDEX
    visible[void] = 0
    colored = VOC_COLORS[visible]
    colored[void] = (224, 224, 224)
    return Image.fromarray(colored, mode="RGB")


def save_comparison(image_path: Path, target: torch.Tensor, prediction: torch.Tensor, output: Path, title: str) -> None:
    with Image.open(image_path) as raw:
        raw = raw.convert("RGB")
    panels = [raw, colorize(target.numpy()), colorize(prediction.numpy())]
    header = 54
    canvas = Image.new("RGB", (raw.width * 3, raw.height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, ("Image", "Ground truth", "Prediction"))):
        canvas.paste(panel, (index * raw.width, header))
        draw.text((index * raw.width + raw.width // 2, 6), label, fill="black", anchor="ma")
    draw.text((8, 27), title, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.inference_mode()
def predict_item(model, dataset: VOCPairedDataset, index: int, device: torch.device):
    image, mask, original_size, valid_size, sample_id = dataset[index]
    logits = model(image.unsqueeze(0).to(device))["out"]
    logits = restore_logits(logits, valid_size, original_size)
    prediction = logits.argmax(1).squeeze(0).cpu()
    if prediction.shape != mask.shape:
        raise RuntimeError(f"Prediction/target shape mismatch for {sample_id}")
    return prediction, mask, sample_id


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    target_class_ids = parse_target_classes(args.target_classes)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    backbone = config.get("backbone")
    architecture = config.get("architecture", "deeplabv3")
    image_size = int(config.get("image_size", 384))
    if backbone not in ("mobilenet_v3_large", "resnet50"):
        raise ValueError(f"Unsupported checkpoint backbone: {backbone}")

    dataset = VOCPairedDataset(args.data_root, "val", image_size=image_size)
    limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
    model = build_segmentation_model(
        architecture=architecture, pretrained_backbone=False, backbone=backbone,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Checkpoint epoch={checkpoint['epoch']}, recorded best mIoU={checkpoint['best_miou']:.6f}", flush=True)
    print(f"Analyzing {limit} validation images on {device} at {image_size}x{image_size}", flush=True)
    aggregate = SegmentationMetrics(NUM_CLASSES)
    sample_rows = []
    id_to_index = {}
    for index in range(limit):
        prediction, target, sample_id = predict_item(model, dataset, index, device)
        aggregate.update(prediction, target)
        sample_rows.extend(sample_class_metrics(prediction, target, sample_id))
        id_to_index[sample_id] = index
        if (index + 1) % args.progress_every == 0 or index + 1 == limit:
            print(f"  analyzed {index + 1}/{limit}", flush=True)

    per_class = class_metrics(aggregate.confusion)
    selected = select_worst_samples(
        sample_rows, target_class_ids, args.samples_per_class, args.min_gt_pixels,
    )
    write_csv(args.output_dir / "class_metrics.csv", per_class)
    write_csv(args.output_dir / "sample_class_metrics.csv", sample_rows)
    write_csv(args.output_dir / "selected_samples.csv", selected, SELECTED_FIELDS)
    write_confusion_csv(args.output_dir / "confusion_counts.csv", aggregate.confusion, normalize=False)
    write_confusion_csv(args.output_dir / "confusion_row_percent.csv", aggregate.confusion, normalize=True)

    for row in selected:
        prediction, target, sample_id = predict_item(model, dataset, id_to_index[row["sample_id"]], device)
        class_dir = args.output_dir / "visualizations" / row["class_name"]
        output = class_dir / f"{row['rank']:02d}_{sample_id}.png"
        title = (
            f"{row['class_name']} recall={row['recall']:.3f} IoU={row['iou']:.3f}; "
            f"mostly confused with {row['most_confused_with']}"
        )
        save_comparison(
            args.data_root / "JPEGImages" / f"{sample_id}.jpg", target, prediction, output, title,
        )
        row["visualization"] = str(output.relative_to(args.output_dir))
        print(f"  saved {row['class_name']} rank {row['rank']}: {sample_id}", flush=True)
    write_csv(args.output_dir / "selected_samples.csv", selected, SELECTED_FIELDS)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_recorded_best_miou": float(checkpoint["best_miou"]),
        "computed_miou": aggregate.mean_iou(),
        "computed_pixel_accuracy": aggregate.pixel_accuracy(),
        "analyzed_images": limit,
        "partial_analysis": limit != len(dataset),
        "image_size": image_size,
        "architecture": architecture,
        "device": str(device),
        "target_classes": [VOC_CLASS_NAMES[index] for index in target_class_ids],
        "samples_per_class": args.samples_per_class,
        "min_gt_pixels": args.min_gt_pixels,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Done: mIoU={summary['computed_miou']:.4f}, pixel accuracy={summary['computed_pixel_accuracy']:.4f}\n"
        f"Results: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
