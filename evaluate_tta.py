"""Evaluate a saved segmentation checkpoint with multi-scale flip TTA."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data_preprocessing import NUM_CLASSES, VOCPreprocessor, mask_array
from src.error_analysis import class_metrics
from src.metrics import SegmentationMetrics, restore_logits
from src.model import build_segmentation_model


PROJECT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT / "experiments" / "deeplabv3plus_resnet50_finetune_01" / "best.pth"
DEFAULT_OUTPUT = PROJECT / "evaluations" / "resnet50_finetune_tta_01"


def parse_scales(value: str) -> tuple[float, ...]:
    try:
        scales = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("scales must be comma-separated numbers") from error
    if not scales or any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("all scales must be positive finite numbers")
    if len(set(scales)) != len(scales):
        raise argparse.ArgumentTypeError("scales must not contain duplicates")
    return scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "VOCdevkit" / "VOC2012")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scales", type=parse_scales, default=parse_scales("0.75,1.0,1.25"))
    parser.add_argument("--no-flip", action="store_true", help="Disable horizontal-flip TTA")
    parser.add_argument("--max-images", type=int, help="Debug only: evaluate a validation prefix")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images <= 0:
        parser.error("max-images must be positive")
    if args.progress_every <= 0:
        parser.error("progress-every must be positive")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


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


@torch.inference_mode()
def tta_logits(model, image: Image.Image, device: torch.device, base_size: int,
               scales: tuple[float, ...], horizontal_flip: bool) -> torch.Tensor:
    """Average original-resolution logits across scale and flip augmentations."""
    logits_sum = None
    count = 0
    for scale in scales:
        canvas_size = max(32, round(base_size * scale))
        item = VOCPreprocessor(image_size=canvas_size, mode="test")(image)
        tensor = item["image"].unsqueeze(0).to(device)
        variants = [(tensor, False)]
        if horizontal_flip:
            variants.append((torch.flip(tensor, dims=(-1,)), True))
        for variant, flipped in variants:
            logits = model(variant)["out"]
            if flipped:
                logits = torch.flip(logits, dims=(-1,))
            logits = restore_logits(logits, item["valid_size"], item["original_size"])
            logits = logits.squeeze(0).float().cpu()
            logits_sum = logits if logits_sum is None else logits_sum + logits
            count += 1
    if logits_sum is None or count == 0:
        raise RuntimeError("No TTA logits were produced")
    return logits_sum / count


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    architecture = config.get("architecture", "deeplabv3")
    backbone = config.get("backbone")
    base_size = int(config.get("image_size", 384))
    model = build_segmentation_model(
        architecture=architecture, backbone=backbone, pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    split_path = args.data_root / "ImageSets" / "Segmentation" / "val.txt"
    sample_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    limit = min(len(sample_ids), args.max_images) if args.max_images else len(sample_ids)
    metrics = SegmentationMetrics(NUM_CLASSES)
    flip = not args.no_flip
    augmentations = len(args.scales) * (2 if flip else 1)
    print(
        f"Checkpoint epoch={checkpoint['epoch']}, baseline mIoU={checkpoint['best_miou']:.6f}\n"
        f"TTA scales={args.scales}, horizontal_flip={flip}, passes/image={augmentations}",
        flush=True,
    )

    for index, sample_id in enumerate(sample_ids[:limit], start=1):
        with Image.open(args.data_root / "JPEGImages" / f"{sample_id}.jpg") as raw:
            image = raw.convert("RGB")
        with Image.open(args.data_root / "SegmentationClass" / f"{sample_id}.png") as raw_mask:
            target = torch.from_numpy(mask_array(raw_mask).astype(np.int64, copy=True))
        logits = tta_logits(model, image, device, base_size, args.scales, flip)
        prediction = logits.argmax(dim=0)
        if prediction.shape != target.shape:
            raise RuntimeError(f"Prediction/target shape mismatch for {sample_id}")
        metrics.update(prediction, target)
        if index % args.progress_every == 0 or index == limit:
            print(f"  evaluated {index}/{limit}", flush=True)

    rows = class_metrics(metrics.confusion)
    baseline_scores_path = args.checkpoint.parent / "best_per_class_iou.json"
    if baseline_scores_path.is_file():
        baseline_scores = json.loads(baseline_scores_path.read_text(encoding="utf-8"))
        for row in rows:
            baseline = baseline_scores.get(str(row["class_id"]))
            row["baseline_iou"] = baseline
            row["tta_iou_delta"] = row["iou"] - baseline if baseline is not None else None
    write_csv(args.output_dir / "class_metrics.csv", rows)

    computed_miou = metrics.mean_iou()
    baseline_miou = float(checkpoint["best_miou"])
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "baseline_miou": baseline_miou,
        "tta_miou": computed_miou,
        "tta_miou_delta": computed_miou - baseline_miou,
        "tta_pixel_accuracy": metrics.pixel_accuracy(),
        "analyzed_images": limit,
        "partial_analysis": limit != len(sample_ids),
        "base_image_size": base_size,
        "scales": list(args.scales),
        "horizontal_flip": flip,
        "passes_per_image": augmentations,
        "architecture": architecture,
        "backbone": backbone,
        "device": str(device),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Done: TTA mIoU={computed_miou:.6f}, delta={computed_miou - baseline_miou:+.6f}, "
        f"pixel accuracy={summary['tta_pixel_accuracy']:.6f}\nResults: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
