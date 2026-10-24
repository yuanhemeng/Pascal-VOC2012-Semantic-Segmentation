from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

project = Path(r"D:\学习记录\3-1\AI_technology\code\Pascal-VOC2012-Semantic-Segmentation")
voc = project / "VOCdevkit" / "VOC2012"

train_ids = [
    x.strip()
    for x in (voc / "ImageSets" / "Segmentation" / "train.txt").read_text().splitlines()
    if x.strip()
]

val_ids = [
    x.strip()
    for x in (voc / "ImageSets" / "Segmentation" / "val.txt").read_text().splitlines()
    if x.strip()
]

all_values = Counter()
invalid_files = []
size_mismatch = []

for sample_id in train_ids + val_ids:
    image_path = voc / "JPEGImages" / f"{sample_id}.jpg"
    mask_path = voc / "SegmentationClass" / f"{sample_id}.png"

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    if image.size != mask.size:
        size_mismatch.append(sample_id)

    values = np.array(mask)
    all_values.update(values.reshape(-1).tolist())

    invalid = set(np.unique(values).tolist()) - set(range(21)) - {255}
    if invalid:
        invalid_files.append((sample_id, sorted(invalid)))

print("Train:", len(train_ids))
print("Val:", len(val_ids))
print("Most common label values:", all_values.most_common())
print("All label values:", sorted(all_values))
print("Invalid label files:", invalid_files[:10])
print("Size mismatches:", size_mismatch[:10])