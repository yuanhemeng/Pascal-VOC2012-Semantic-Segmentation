from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT = Path(
    r"D:\学习记录\3-1\AI_technology\code\Pascal-VOC2012-Semantic-Segmentation"
)

VOC = PROJECT / "VOCdevkit" / "VOC2012"
TRAIN_LIST = VOC / "ImageSets" / "Segmentation" / "train.txt"
OUTPUT_DIR = PROJECT / "experiments" / "data_check" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def voc_palette(num_classes=256):
    """
    为每个类别编号生成固定颜色。
    0 到 20 是 VOC 类别，255 作为忽略区域显示为黑色。
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)

    for label in range(num_classes):
        value = label
        red = 0
        green = 0
        blue = 0
        bit = 0

        while value:
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
            bit += 1

        palette[label] = [red, green, blue]

    # 255 表示忽略区域，显示为黑色
    palette[255] = [0, 0, 0]

    return palette


def colorize_mask(mask):
    """
    把类别编号标签转换成 RGB 彩色图。
    """
    palette = voc_palette()
    return palette[mask]


def read_sample_ids(count=4):
    ids = [
        line.strip()
        for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(ids) < count:
        raise ValueError(f"训练集只有 {len(ids)} 张，无法选择 {count} 张样本")

    return ids[:count]


def visualize_sample(sample_id, index):
    image_path = VOC / "JPEGImages" / f"{sample_id}.jpg"
    mask_path = VOC / "SegmentationClass" / f"{sample_id}.png"

    image = Image.open(image_path).convert("RGB")
    mask = np.array(Image.open(mask_path), dtype=np.uint8)
    colored_mask = colorize_mask(mask)

    image_array = np.array(image)

    # 透明叠加图，帮助观察标签是否与原图位置对齐
    overlay = (
        0.55 * image_array.astype(np.float32)
        + 0.45 * colored_mask.astype(np.float32)
    ).astype(np.uint8)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation Label")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    figure.suptitle(
        f"VOC2012 sample: {sample_id}",
        fontsize=14,
    )

    figure.tight_layout()

    output_path = OUTPUT_DIR / f"{index:02d}_{sample_id}.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {output_path}")
    print(f"Image size: {image.size}")
    print(f"Mask size: {mask.shape[1], mask.shape[0]}")
    print(f"Mask values: {sorted(np.unique(mask).tolist())}")


sample_ids = read_sample_ids(count=4)

for index, sample_id in enumerate(sample_ids, start=1):
    visualize_sample(sample_id, index)