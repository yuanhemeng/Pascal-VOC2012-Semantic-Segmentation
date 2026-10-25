"""保存一组可复现的、经过处理的VOC图像/掩码预览文件."""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.data_preprocessing import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, VOCPreprocessor

PROJECT = Path(__file__).resolve().parent
VOC = PROJECT / "VOCdevkit" / "VOC2012"
OUTPUT = PROJECT / "experiments" / "data_check"


def palette() -> np.ndarray:
    colors = np.zeros((256, 3), dtype=np.uint8)
    for label in range(21):
        value = label
        for bit in range(8):
            for channel in range(3):
                colors[label, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    return colors


def denormalize(tensor) -> Image.Image:
    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(IMAGE_STD, dtype=np.float32)
    pixels = tensor.permute(1, 2, 0).numpy() * std + mean
    return Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8), mode="RGB")


def save_preview(sample_id: str, index: int, preprocessor: VOCPreprocessor, colors: np.ndarray, folder: Path) -> set[int]:
    with Image.open(VOC / "JPEGImages" / f"{sample_id}.jpg") as image:
        with Image.open(VOC / "SegmentationClass" / f"{sample_id}.png") as mask:
            item = preprocessor(image, mask)
    labels = item["mask"].numpy()
    image_panel = denormalize(item["image"])
    mask_panel = Image.fromarray(colors[labels], mode="RGB")
    overlay = Image.blend(image_panel, mask_panel, alpha=0.45)
    width, height = image_panel.size
    canvas = Image.new("RGB", (width * 3, height + 28), "white")
    for column, panel in enumerate((image_panel, mask_panel, overlay)):
        canvas.paste(panel, (column * width, 28))
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate((f"{sample_id} | image", "mask", "overlay")):
        draw.text((column * width + 8, 7), title, fill="black")
    canvas.save(folder / f"{index:02d}_{sample_id}.png")
    return set(np.unique(labels).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    random.seed(args.seed)
    ids = [line.strip() for line in (VOC / "ImageSets" / "Segmentation" / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.count > len(ids):
        parser.error(f"--count exceeds the train split size ({len(ids)})")
    folder = OUTPUT / "preprocessing_samples"
    folder.mkdir(parents=True, exist_ok=True)
    preprocessor = VOCPreprocessor(image_size=args.image_size, mode="train")
    colors = palette()
    observed: set[int] = set()
    for index, sample_id in enumerate(ids[:args.count], start=1):
        observed.update(save_preview(sample_id, index, preprocessor, colors, folder))
    summary = OUTPUT / "preprocessing_check.txt"
    summary.write_text(
        f"Samples checked: {args.count}\n"
        f"Random seed: {args.seed}\n"
        f"Image shape: [3, {args.image_size}, {args.image_size}]\n"
        f"Training mask shape: [{args.image_size}, {args.image_size}]\n"
        f"Observed values in sampled masks: {sorted(observed)}\n"
        "Image interpolation: bilinear; mask interpolation: nearest\n"
        "Synchronized augmentation: scale, crop, horizontal flip\n",
        encoding="utf-8",
    )
    print(f"Saved {args.count} previews to {folder}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
