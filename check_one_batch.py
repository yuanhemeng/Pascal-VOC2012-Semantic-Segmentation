"""单批次模型更新检查"""

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data_preprocessing import IGNORE_INDEX, NUM_CLASSES, VOCPairedDataset
from src.model import build_deeplabv3


PROJECT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.image_size <= 0 or args.batch_size < 2 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("image size and learning rate must be positive; batch size must be at least 2")

    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")

    data = VOCPairedDataset(PROJECT / "VOCdevkit" / "VOC2012", "train", args.image_size)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, masks, _, _, sample_ids = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    if masks.ne(IGNORE_INDEX).sum().item() == 0:
        raise RuntimeError("The sampled batch has no valid labeled pixels")

    model = build_deeplabv3(pretrained_backbone=False).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = model(images)["out"]
    expected_shape = (args.batch_size, NUM_CLASSES, args.image_size, args.image_size)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Expected logits {expected_shape}, got {tuple(output.shape)}")
    loss = F.cross_entropy(output, masks, ignore_index=IGNORE_INDEX)
    if not torch.isfinite(loss):
        raise RuntimeError("Loss is not finite")

    final_weight = model.classifier[-1].weight
    before = final_weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if final_weight.grad is None or not torch.isfinite(final_weight.grad).all():
        raise RuntimeError("The final classifier did not receive a finite gradient")
    optimizer.step()
    changed = not torch.equal(before, final_weight.detach())
    if not changed:
        raise RuntimeError("Optimizer step did not update the final classifier")

    print(f"device: {device}")
    print(f"samples: {list(sample_ids)}")
    print(f"images: {tuple(images.shape)}, labels: {tuple(masks.shape)}")
    print(f"logits: {tuple(output.shape)}")
    print(f"valid labeled pixels: {masks.ne(IGNORE_INDEX).sum().item()}")
    print(f"loss: {loss.item():.6f}")
    print("gradient: finite; optimizer update: successful")
    print("This is a pipeline check, not a trained model or a validation score.")


if __name__ == "__main__":
    main()
