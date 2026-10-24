from pathlib import Path
from PIL import Image

PROJECT = Path(
    r"D:\学习记录\3-1\AI_technology\code\Pascal-VOC2012-Semantic-Segmentation"
)

VOC = PROJECT / "VOCdevkit" / "VOC2012"
SEGMENTATION = VOC / "ImageSets" / "Segmentation"

train_file = SEGMENTATION / "train.txt"
val_file = SEGMENTATION / "val.txt"

train_ids = [
    line.strip()
    for line in train_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

val_ids = [
    line.strip()
    for line in val_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def check_split(split_name, sample_ids):
    mismatches = []
    checked = 0

    for sample_id in sample_ids:
        image_path = VOC / "JPEGImages" / f"{sample_id}.jpg"
        mask_path = VOC / "SegmentationClass" / f"{sample_id}.png"

        image_size = Image.open(image_path).size
        mask_size = Image.open(mask_path).size

        checked += 1

        if image_size != mask_size:
            mismatches.append(
                {
                    "id": sample_id,
                    "image_size": image_size,
                    "mask_size": mask_size,
                }
            )

    print(f"{split_name} checked: {checked}")
    print(f"{split_name} size mismatches: {len(mismatches)}")

    if mismatches:
        print(f"\n{split_name} mismatch details:")
        for item in mismatches[:20]:
            print(
                item["id"],
                "image=",
                item["image_size"],
                "mask=",
                item["mask_size"],
            )

    return mismatches


train_mismatches = check_split("train", train_ids)
val_mismatches = check_split("val", val_ids)

all_mismatches = train_mismatches + val_mismatches

result_dir = PROJECT / "experiments" / "data_check"
result_dir.mkdir(parents=True, exist_ok=True)

result_file = result_dir / "image_mask_size_check.txt"

with result_file.open("w", encoding="utf-8") as file:
    file.write("Image and mask size check\n")
    file.write(f"Train samples: {len(train_ids)}\n")
    file.write(f"Val samples: {len(val_ids)}\n")
    file.write(f"Train mismatches: {len(train_mismatches)}\n")
    file.write(f"Val mismatches: {len(val_mismatches)}\n")
    file.write(f"Total mismatches: {len(all_mismatches)}\n\n")

    for item in all_mismatches:
        file.write(
            f'{item["id"]}: '
            f'image={item["image_size"]}, '
            f'mask={item["mask_size"]}\n'
        )

print(f"\nResult saved to: {result_file}")