"""Train a DeepLabV3 baseline on the full VOC2012 train and val splits."""

import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data_preprocessing import IGNORE_INDEX, IMAGE_SIZE, NUM_CLASSES, VOCPairedDataset
from src.metrics import SegmentationMetrics, restore_logits
from src.model import build_deeplabv3


PROJECT = Path(__file__).resolve().parent
CSV_COLUMNS = (
    "epoch", "train_loss", "val_loss", "val_miou", "val_pixel_accuracy",
    "train_images", "val_images", "learning_rate", "epoch_seconds", "partial_run",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "VOCdevkit" / "VOC2012")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "experiments" / "baseline_01")
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--backbone", choices=("mobilenet_v3_large", "resnet50"), default="mobilenet_v3_large")
    parser.add_argument("--batch-size", type=positive_int, default=2)
    parser.add_argument("--image-size", type=positive_int, default=IMAGE_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine"),
        default="none",
        help="Learning-rate schedule; cosine decays from --lr to --min-lr over all epochs",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=positive_int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--random-init", action="store_true", help="Skip ImageNet backbone download; for debugging only")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the segmentation head")
    parser.add_argument("--resume", type=Path, help="Resume from this run's last.pth")
    parser.add_argument("--max-train-batches", type=positive_int, help="Debug only: limit train batches")
    parser.add_argument("--max-val-batches", type=positive_int, help="Debug only: limit validation images")
    parser.add_argument("--no-checkpoints", action="store_true", help="Debug only: do not save model weights")
    parser.add_argument("--log-every", type=positive_int, default=50, help="Print progress every N training batches")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("DeepLabV3 training needs batch size >= 2 because of BatchNorm in global pooling")
    if args.num_workers < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("num-workers must be nonnegative and lr must be positive and finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be nonnegative and finite")
    if not math.isfinite(args.min_lr) or args.min_lr < 0 or args.min_lr > args.lr:
        parser.error("min-lr must be finite, nonnegative, and no greater than lr")
    if (args.max_train_batches or args.max_val_batches) and not args.no_checkpoints:
        parser.error("Debug batch limits require --no-checkpoints to avoid saving partial-run weights")
    if args.resume and args.no_checkpoints:
        parser.error("--resume and --no-checkpoints cannot be combined")
    return args


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def single_sample(batch):
    return batch[0]


def train_epoch(model, loader, optimizer, device, max_batches, log_every, freeze_backbone):
    model.train()
    if freeze_backbone:
        model.backbone.eval()
    loss_total = 0.0
    valid_total = 0
    images_seen = 0
    started = time.perf_counter()
    for batch_index, (images, masks, _, _, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)
        valid = masks.ne(IGNORE_INDEX).sum().item()
        if valid == 0:
            continue
        logits = model(images)["out"]
        if logits.shape[:2] != (images.shape[0], NUM_CLASSES) or logits.shape[-2:] != masks.shape[-2:]:
            raise RuntimeError(f"Unexpected output shape: {tuple(logits.shape)}")
        loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss in batch {batch_index}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * valid
        valid_total += valid
        images_seen += images.shape[0]
        if (batch_index + 1) % log_every == 0:
            print(f"  train batch {batch_index + 1}/{len(loader)}, images={images_seen}, elapsed={time.perf_counter() - started:.0f}s", flush=True)
    if valid_total == 0:
        raise RuntimeError("No labeled training pixels were processed")
    return loss_total / valid_total, images_seen


@torch.inference_mode()
def validate(model, loader, device, max_batches, writer, epoch):
    model.eval()
    metrics = SegmentationMetrics(NUM_CLASSES)
    loss_total = 0.0
    valid_total = 0
    images_seen = 0
    for batch_index, (image, mask, original_size, valid_size, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        logits = model(image.unsqueeze(0).to(device))["out"]
        logits = restore_logits(logits, valid_size, original_size)
        target = mask.unsqueeze(0).to(device)
        if logits.shape[-2:] != target.shape[-2:]:
            raise RuntimeError("Validation logits and original mask have different sizes")
        loss = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss in image {batch_index}")
        prediction = logits.argmax(dim=1).squeeze(0).cpu()
        metrics.update(prediction, mask)
        valid = mask.ne(IGNORE_INDEX).sum().item()
        loss_total += loss.item() * valid
        valid_total += valid
        images_seen += 1
        if images_seen % 200 == 0:
            print(f"  val image {images_seen}/{len(loader)}", flush=True)
        if batch_index == 0:
            writer.add_image("val/sample_prediction", prediction.float().unsqueeze(0) / 20, epoch)
            visible_target = mask.masked_fill(mask == IGNORE_INDEX, 0)
            writer.add_image("val/sample_label", visible_target.float().unsqueeze(0) / 20, epoch)
    if valid_total == 0:
        raise RuntimeError("No labeled validation pixels were processed")
    return loss_total / valid_total, metrics, images_seen


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_miou, config):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch,
        "best_miou": best_miou,
        "config": config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(PROJECT / "weights" / "torch"))
    device = device_from_arg(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.set_num_threads(args.cpu_threads)

    train_data = VOCPairedDataset(args.data_root, "train", args.image_size)
    val_data = VOCPairedDataset(args.data_root, "val", args.image_size)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=single_sample, num_workers=0)
    if len(train_loader) == 0:
        raise ValueError("Batch size exceeds the train split size")
    config = {
        "model": f"DeepLabV3-{args.backbone}", "backbone": args.backbone,
        "pretrained_backbone": not args.random_init,
        "freeze_backbone": args.freeze_backbone,
        "image_size": args.image_size, "batch_size": args.batch_size,
        "lr": args.lr, "lr_scheduler": args.lr_scheduler, "min_lr": args.min_lr,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "train_images": len(train_data), "val_images": len(val_data),
        "max_train_batches": args.max_train_batches, "max_val_batches": args.max_val_batches,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "python": platform.python_version(), "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
    }
    if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    if args.resume and args.resume.resolve().parent != args.output_dir.resolve():
        raise ValueError("Resume checkpoint must be inside --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (args.output_dir / "command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")

    model = build_deeplabv3(
        pretrained_backbone=not args.random_init and args.resume is None,
        backbone=args.backbone,
    ).to(device)
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr,
        )
    start_epoch, best_miou = 1, -1.0
    if args.resume:
        # The first checkpoint may contain TorchVersion rather than a plain str.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint configuration does not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        random.setstate(checkpoint["python_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint["best_miou"])
    if start_epoch > args.epochs:
        raise ValueError("--epochs must exceed the last completed checkpoint epoch")

    with (args.output_dir / "command_history.txt").open("a", encoding="utf-8") as history:
        history.write(subprocess.list2cmdline(sys.argv) + "\n")

    csv_path = args.output_dir / "metrics.csv"
    csv_exists = csv_path.exists()
    partial = bool(args.max_train_batches or args.max_val_batches)
    writer = SummaryWriter(log_dir=str(args.output_dir / "tensorboard"), purge_step=start_epoch if args.resume else None)
    try:
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
            if not csv_exists:
                csv_writer.writeheader()
            for epoch in range(start_epoch, args.epochs + 1):
                started = time.perf_counter()
                train_loss, train_seen = train_epoch(
                    model, train_loader, optimizer, device, args.max_train_batches,
                    args.log_every, args.freeze_backbone,
                )
                val_loss, metrics, val_seen = validate(model, val_loader, device, args.max_val_batches, writer, epoch)
                miou, accuracy = metrics.mean_iou(), metrics.pixel_accuracy()
                if not math.isfinite(miou) or not math.isfinite(accuracy):
                    raise RuntimeError("Validation metrics are not finite")
                record = {
                    "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "val_miou": miou, "val_pixel_accuracy": accuracy,
                    "train_images": train_seen, "val_images": val_seen,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - started, "partial_run": partial,
                }
                csv_writer.writerow(record)
                stream.flush()
                writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
                writer.add_scalar("val/mIoU", miou, epoch)
                writer.add_scalar("val/pixel_accuracy", accuracy, epoch)
                writer.add_scalar("train/learning_rate", record["learning_rate"], epoch)
                writer.flush()
                print(f"epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, mIoU={miou:.4f}, accuracy={accuracy:.4f}, val_images={val_seen}, partial={partial}", flush=True)
                if scheduler is not None:
                    scheduler.step()
                if not args.no_checkpoints:
                    improved = miou > best_miou
                    best_miou = max(best_miou, miou)
                    save_checkpoint(args.output_dir / "last.pth", model, optimizer, scheduler, epoch, best_miou, config)
                    if improved:
                        save_checkpoint(args.output_dir / "best.pth", model, optimizer, scheduler, epoch, best_miou, config)
                        scores = metrics.per_class_iou().tolist()
                        (args.output_dir / "best_per_class_iou.json").write_text(
                            json.dumps({str(i): score if math.isfinite(score) else None for i, score in enumerate(scores)}, indent=2),
                            encoding="utf-8",
                        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
