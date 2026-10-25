"""Audit the raw VOC segmentation splits without modifying the dataset."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_preprocessing import IGNORE_INDEX, NUM_CLASSES

PROJECT = Path(__file__).resolve().parent
VOC = PROJECT / "VOCdevkit" / "VOC2012"
TEST_IMAGES = PROJECT / "test400" / "images"
OUTPUT = PROJECT / "experiments" / "data_check" / "dataset_audit.json"


def read_ids(split: str) -> list[str]:
    path = VOC / "ImageSets" / "Segmentation" / f"{split}.txt"
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def inspect_split(ids: list[str]) -> dict:
    issues: list[str] = []
    counts: Counter[int] = Counter()
    for sample_id in ids:
        image_path = VOC / "JPEGImages" / f"{sample_id}.jpg"
        mask_path = VOC / "SegmentationClass" / f"{sample_id}.png"
        if not image_path.is_file() or not mask_path.is_file():
            issues.append(f"{sample_id}: missing image or mask")
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                issues.append(f"{sample_id}: size mismatch {image.size} != {mask.size}")
            if mask.mode not in ("P", "L"):
                issues.append(f"{sample_id}: unsupported mask mode {mask.mode}")
                continue
            values, frequencies = np.unique(np.asarray(mask), return_counts=True)
            for value, frequency in zip(values.tolist(), frequencies.tolist()):
                counts[value] += frequency
                if value not in range(NUM_CLASSES) and value != IGNORE_INDEX:
                    issues.append(f"{sample_id}: invalid label {value}")
    return {
        "samples": len(ids),
        "label_values": sorted(counts),
        "pixel_counts": {str(k): v for k, v in sorted(counts.items())},
        "issues": issues,
    }


def main() -> None:
    train_ids, val_ids = read_ids("train"), read_ids("val")
    report = {
        "voc_root": str(VOC),
        "test_images": str(TEST_IMAGES),
        "test_image_count": len(list(TEST_IMAGES.glob("*.jpg"))),
        "train_val_overlap": sorted(set(train_ids) & set(val_ids)),
        "train": inspect_split(train_ids),
        "val": inspect_split(val_ids),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train={len(train_ids)}, val={len(val_ids)}, test={report['test_image_count']}")
    print(f"overlap={len(report['train_val_overlap'])}, train issues={len(report['train']['issues'])}, val issues={len(report['val']['issues'])}")
    print(f"Details: {OUTPUT}")
    if report["train_val_overlap"] or report["train"]["issues"] or report["val"]["issues"]:
        raise SystemExit("Dataset audit found issues; see the report")


if __name__ == "__main__":
    main()
