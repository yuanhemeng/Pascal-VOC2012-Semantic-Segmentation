"""用于VOC2012分割图像和类别索引掩码的配对预处理。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

Mode = Literal["train", "val", "test"]
NUM_CLASSES = 21
IGNORE_INDEX = 255
IMAGE_SIZE = 384
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
PAD_COLOR = (124, 116, 104)


def mask_array(mask: Image.Image) -> np.ndarray:
    if mask.mode not in ("P", "L"):
        raise ValueError(f"Expected a single-channel class-index mask, got {mask.mode}")
    values = np.asarray(mask)
    invalid = set(np.unique(values).tolist()) - set(range(NUM_CLASSES)) - {IGNORE_INDEX}
    if invalid:
        raise ValueError(f"Invalid mask class values: {sorted(invalid)}")
    return values


def image_tensor(image: Image.Image) -> torch.Tensor:
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
    pixels = pixels / 255.0
    mean = pixels.new_tensor(IMAGE_MEAN).view(3, 1, 1)
    std = pixels.new_tensor(IMAGE_STD).view(3, 1, 1)
    return (pixels - mean) / std


class VOCPreprocessor:
    """Use random paired geometry for train and deterministic geometry for val/test.

    Validation masks retain original resolution. To evaluate predictions, remove
    padding using ``valid_size`` and resize logits to ``original_size`` first.
    """

    def __init__(self, image_size: int = IMAGE_SIZE, mode: Mode = "train") -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.image_size = image_size
        self.mode = mode

    def __call__(self, image: Image.Image, mask: Image.Image | None = None) -> dict:
        image = image.convert("RGB")
        if self.mode == "test":
            if mask is not None:
                raise ValueError("Test mode does not accept a mask")
        elif mask is None:
            raise ValueError(f"A mask is required in {self.mode} mode")
        else:
            mask_array(mask)
            if image.size != mask.size:
                raise ValueError(f"Image/mask size mismatch: {image.size} vs {mask.size}")

        original_size = (image.height, image.width)
        size = self.image_size

        if self.mode == "train":
            short_side = max(1, round(size * random.uniform(0.75, 1.25)))
            scale = short_side / min(image.size)
            resized = tuple(max(1, round(length * scale)) for length in image.size)
            image = image.resize(resized, Image.Resampling.BILINEAR)
            mask = mask.resize(resized, Image.Resampling.NEAREST)
            pad = (0, 0, max(0, size - image.width), max(0, size - image.height))
            image = ImageOps.expand(image, border=pad, fill=PAD_COLOR)
            mask = ImageOps.expand(mask, border=pad, fill=IGNORE_INDEX)
            left = random.randint(0, image.width - size)
            top = random.randint(0, image.height - size)
            crop = (left, top, left + size, top + size)
            image, mask = image.crop(crop), mask.crop(crop)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            valid_size = (size, size)
        else:
            scale = size / max(image.size)
            resized = tuple(max(1, round(length * scale)) for length in image.size)
            image = image.resize(resized, Image.Resampling.BILINEAR)
            valid_size = (image.height, image.width)
            image = ImageOps.expand(
                image,
                border=(0, 0, size - image.width, size - image.height),
                fill=PAD_COLOR,
            )

        return {
            "image": image_tensor(image),
            "mask": torch.from_numpy(mask_array(mask).astype(np.int64, copy=True)) if mask is not None else None,
            "original_size": original_size,
            "valid_size": valid_size,
        }


class VOCPairedDataset(Dataset):
    """Read a VOC train/val split; validation items have original-size masks."""

    def __init__(self, voc_root: str | Path, split: str, image_size: int = IMAGE_SIZE):
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.root = Path(voc_root)
        path = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        self.ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.preprocess = VOCPreprocessor(image_size=image_size, mode=split)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        sample_id = self.ids[index]
        with Image.open(self.root / "JPEGImages" / f"{sample_id}.jpg") as image:
            with Image.open(self.root / "SegmentationClass" / f"{sample_id}.png") as mask:
                item = self.preprocess(image, mask)
        return item["image"], item["mask"], item["original_size"], item["valid_size"], sample_id
