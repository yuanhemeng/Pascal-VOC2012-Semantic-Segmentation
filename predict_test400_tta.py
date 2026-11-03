"""Predict course test images with multi-scale flip TTA and package masks."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_tta import load_checkpoint, resolve_device, tta_logits
from src.data_preprocessing import NUM_CLASSES
from src.model import build_segmentation_model


PROJECT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT / "experiments" / "resnet50_512_finetune_01" / "best.pth"
DEFAULT_INPUT_DIR = PROJECT / "test400" / "images"
DEFAULT_OUTPUT_DIR = PROJECT / "predictions" / "resnet50_512_tta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive-path", type=Path, required=True,
                        help="Final masks zip, e.g. deliverables/predict_NAME_ID.zip")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--scales", default="0.75,1.0,1.25")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--expected-count", type=int, default=400)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    if args.expected_count <= 0 or args.progress_every <= 0:
        parser.error("expected-count and progress-every must be positive")
    return args


def list_test_images(input_dir: Path, expected_count: int) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {input_dir}")
    images = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) != expected_count:
        raise ValueError(f"Expected {expected_count} test images, found {len(images)}")
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Test image stems are not unique")
    return images


def validate_mask(path: Path, expected_size: tuple[int, int]) -> dict:
    with Image.open(path) as mask:
        array = np.asarray(mask)
        mode = mask.mode
        size = mask.size
    if mode != "L" or array.ndim != 2:
        raise ValueError(f"Mask must be one-channel L mode: {path}")
    if size != expected_size:
        raise ValueError(f"Size mismatch for {path.name}: got {size}, expected {expected_size}")
    minimum, maximum = int(array.min()), int(array.max())
    if minimum < 0 or maximum >= NUM_CLASSES:
        raise ValueError(f"Class values must be in 0..{NUM_CLASSES - 1}: {path}")
    return {
        "filename": path.name,
        "width": size[0],
        "height": size[1],
        "mode": mode,
        "dtype": str(array.dtype),
        "min_class": minimum,
        "max_class": maximum,
    }


def package_masks(output_dir: Path, archive_path: Path, expected_names: set[str]) -> None:
    files = sorted(output_dir.glob("*.png"))
    names = {path.name for path in files}
    if names != expected_names or len(files) != len(expected_names):
        missing = sorted(expected_names - names)[:5]
        extra = sorted(names - expected_names)[:5]
        raise RuntimeError(f"Prediction set mismatch; missing={missing}, extra={extra}")
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=path.name)
        temporary_archive.replace(archive_path)
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    scales = tuple(float(value.strip()) for value in args.scales.split(",") if value.strip())
    if not scales or any(value <= 0 for value in scales) or len(set(scales)) != len(scales):
        raise ValueError("scales must be unique positive comma-separated numbers")
    horizontal_flip = not args.no_flip

    images = list_test_images(args.input_dir, args.expected_count)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    architecture = config.get("architecture", "deeplabv3")
    backbone = config.get("backbone")
    image_size = int(config.get("image_size", 384))

    output_dir = args.output_dir.resolve()
    archive_path = args.archive_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Prediction directory is not empty: {output_dir}")
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")
    manifest_path = output_dir.parent / f"{output_dir.name}_manifest.csv"
    summary_path = output_dir.parent / f"{output_dir.name}_summary.json"
    for artifact in (manifest_path, summary_path):
        if artifact.exists():
            raise FileExistsError(f"Prediction metadata already exists: {artifact}")
    if archive_path == output_dir or output_dir in archive_path.parents:
        raise ValueError("Archive path must be outside the prediction directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_segmentation_model(
        architecture=architecture, backbone=backbone, pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(
        f"Checkpoint epoch={checkpoint['epoch']}, validation mIoU={checkpoint['best_miou']:.6f}\n"
        f"Predicting {len(images)} images on {device}; size={image_size}, scales={scales}, "
        f"flip={horizontal_flip}",
        flush=True,
    )

    manifest = []
    try:
        with torch.inference_mode():
            for index, image_path in enumerate(images, start=1):
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                    original_size = image.size
                logits = tta_logits(model, image, device, image_size, scales, horizontal_flip)
                prediction = logits.argmax(dim=0).to(torch.uint8).cpu().numpy()
                output_path = output_dir / f"{image_path.stem}.png"
                Image.fromarray(prediction, mode="L").save(output_path)
                row = validate_mask(output_path, original_size)
                row["source_filename"] = image_path.name
                manifest.append(row)
                if index % args.progress_every == 0 or index == len(images):
                    print(f"  predicted and checked {index}/{len(images)}", flush=True)

        expected_names = {f"{path.stem}.png" for path in images}
        package_masks(output_dir, archive_path, expected_names)
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_validation_miou": float(checkpoint["best_miou"]),
            "prediction_count": len(manifest),
            "scales": list(scales),
            "horizontal_flip": horizontal_flip,
            "passes_per_image": len(scales) * (2 if horizontal_flip else 1),
            "mask_format": "single-channel 8-bit PNG class-index mask",
            "class_range": [0, NUM_CLASSES - 1],
            "device": str(device),
            "archive": str(archive_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        for path in output_dir.glob("*.png"):
            path.unlink(missing_ok=True)
        raise

    print(
        f"Complete: {len(manifest)} masks validated.\n"
        f"Masks: {output_dir}\nArchive: {archive_path}\n"
        f"Manifest: {manifest_path}\nSummary: {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
