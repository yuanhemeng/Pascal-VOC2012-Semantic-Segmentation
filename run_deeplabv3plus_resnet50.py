"""Launch the controlled frozen-ResNet50 DeepLabV3+ experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT / "experiments" / "deeplabv3plus_resnet50_01"
DEFAULT_SMOKE_OUTPUT = PROJECT / "experiments" / "deeplabv3plus_resnet50_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def build_command(*, output_dir: Path, device: str, batch_size: int, smoke: bool) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT / "train_baseline.py"),
        "--device", device,
        "--architecture", "deeplabv3plus",
        "--backbone", "resnet50",
        "--freeze-backbone",
        "--image-size", "384",
        "--batch-size", str(batch_size),
        "--lr", "0.001",
        "--lr-scheduler", "cosine",
        "--min-lr", "0.00001",
        "--loss", "ce_dice",
        "--dice-weight", "1.0",
        "--epochs", "1" if smoke else "20",
        "--log-every", "20",
        "--output-dir", str(output_dir),
    ]
    if smoke:
        command.extend((
            "--max-train-batches", "1", "--max-val-batches", "1", "--no-checkpoints",
        ))
    return command


def main() -> None:
    args = parse_args()
    default_output = DEFAULT_SMOKE_OUTPUT if args.smoke else DEFAULT_OUTPUT
    output_dir = (args.output_dir or default_output).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    mode = "smoke check" if args.smoke else "20-epoch ResNet50 experiment"
    print(f"Launching {mode}: {output_dir}", flush=True)
    command = build_command(
        output_dir=output_dir, device=args.device,
        batch_size=args.batch_size, smoke=args.smoke,
    )
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
