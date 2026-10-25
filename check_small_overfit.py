"""Repeat updates on the same two VOC samples before full baseline training."""

import argparse
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data_preprocessing import IGNORE_INDEX, VOCPairedDataset
from src.model import build_deeplabv3


PROJECT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.steps < 2 or args.image_size <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps must be >= 2; image size and learning rate must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    dataset = VOCPairedDataset(PROJECT / "VOCdevkit" / "VOC2012", "train", args.image_size)
    images, masks, _, _, sample_ids = next(iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)))
    images, masks = images.to(device), masks.to(device)
    if masks.ne(IGNORE_INDEX).sum().item() == 0:
        raise RuntimeError("No valid labels in the selected samples")
    model = build_deeplabv3(pretrained_backbone=False).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    for step in range(1, args.steps + 1):
        logits = model(images)["out"]
        loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"step {step}/{args.steps}: loss={loss.item():.6f}", flush=True)
    print(f"samples: {list(sample_ids)}")
    print(f"first loss: {losses[0]:.6f}; last loss: {losses[-1]:.6f}")
    if losses[-1] >= losses[0]:
        raise RuntimeError("Loss did not decrease on the fixed batch; investigate before full training")


if __name__ == "__main__":
    main()
