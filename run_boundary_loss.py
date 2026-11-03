"""Launch boundary-weighted CE + Dice fine-tuning from the 70.21% checkpoint."""

import argparse
import subprocess
import sys
from pathlib import Path
import torch

PROJECT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT / "experiments" / "deeplabv3plus_resnet50_finetune_01" / "best.pth"


def validate_source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {path}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    expected = {"architecture": "deeplabv3plus", "backbone": "resnet50",
                "image_size": 384, "loss": "ce_dice", "unfreeze_last_blocks": 1}
    mismatches = {k: (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    if mismatches:
        raise ValueError(f"Source is not the 70.21% ResNet50 checkpoint: {mismatches}")
    return checkpoint


def build_command(source: Path, output: Path, device: str, smoke: bool) -> list[str]:
    command = [sys.executable, str(PROJECT / "train_baseline.py"),
        "--device", device, "--architecture", "deeplabv3plus", "--backbone", "resnet50",
        "--freeze-backbone", "--unfreeze-last-blocks", "1", "--init-from", str(source),
        "--image-size", "384", "--batch-size", "4", "--lr", "0.00001",
        "--backbone-lr", "0.000001", "--lr-scheduler", "cosine", "--min-lr", "0.0000001",
        "--loss", "boundary_ce_dice", "--dice-weight", "1.0",
        "--boundary-weight", "2.0", "--boundary-width", "3",
        "--epochs", "1" if smoke else "5", "--log-every", "20", "--output-dir", str(output)]
    if smoke:
        command += ["--max-train-batches", "1", "--max-val-batches", "1", "--no-checkpoints"]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(); checkpoint = validate_source(source)
    name = "resnet50_boundary_loss_smoke" if args.smoke else "resnet50_boundary_loss_01"
    output = (args.output_dir or PROJECT / "experiments" / name).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    print(f"Source verified: epoch={checkpoint['epoch']}, mIoU={checkpoint['best_miou']:.6f}", flush=True)
    command = build_command(source, output, args.device, args.smoke)
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()
