"""Launch the controlled DeepLabV3+ stage-B backbone fine-tuning experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT / "experiments" / "deeplabv3plus_decoder_01" / "best.pth"
DEFAULT_OUTPUT = PROJECT / "experiments" / "deeplabv3plus_finetune_01"
DEFAULT_SMOKE_OUTPUT = PROJECT / "experiments" / "deeplabv3plus_finetune_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run one train batch and one validation image without saving checkpoints",
    )
    return parser.parse_args()


def load_and_validate_source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Stage-A checkpoint not found: {path}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "config", "epoch", "best_miou"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Source checkpoint is missing keys: {sorted(missing)}")

    config = checkpoint["config"]
    expected = {
        "architecture": "deeplabv3plus",
        "backbone": "mobilenet_v3_large",
        "image_size": 384,
        "loss": "ce_dice",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Source checkpoint is not the stage-A model: {details}")
    return checkpoint


def build_command(*, source: Path, output_dir: Path, device: str, smoke: bool) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT / "train_baseline.py"),
        "--device", device,
        "--architecture", "deeplabv3plus",
        "--backbone", "mobilenet_v3_large",
        "--freeze-backbone",
        "--unfreeze-last-blocks", "3",
        "--init-from", str(source),
        "--image-size", "384",
        "--batch-size", "4",
        "--lr", "0.0001",
        "--backbone-lr", "0.00001",
        "--lr-scheduler", "cosine",
        "--min-lr", "0.000001",
        "--loss", "ce_dice",
        "--dice-weight", "1.0",
        "--epochs", "1" if smoke else "10",
        "--log-every", "20",
        "--output-dir", str(output_dir),
    ]
    if smoke:
        command.extend((
            "--max-train-batches", "1",
            "--max-val-batches", "1",
            "--no-checkpoints",
        ))
    return command


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    checkpoint = load_and_validate_source(source)
    default_output = DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT
    output_dir = (args.output_dir or default_output).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if output_dir == source.parent.resolve():
        raise ValueError("Output directory cannot be the stage-A experiment directory")

    mode = "smoke check" if args.smoke else "10-epoch stage-B experiment"
    print(
        f"Source verified: epoch={checkpoint['epoch']}, "
        f"mIoU={float(checkpoint['best_miou']):.6f}\n"
        f"Launching {mode}: {output_dir}",
        flush=True,
    )
    command = build_command(
        source=source, output_dir=output_dir, device=args.device, smoke=args.smoke,
    )
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
