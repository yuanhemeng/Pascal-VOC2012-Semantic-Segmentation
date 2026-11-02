import tempfile
import unittest
from pathlib import Path

import torch

from run_deeplabv3plus_polish import build_command, load_and_validate_source


class TestPolishRunner(unittest.TestCase):
    def test_formal_command_uses_five_low_lr_epochs(self):
        command = build_command(
            source=Path("source.pth"), output_dir=Path("output"), device="cuda", smoke=False,
        )
        text = " ".join(command)
        self.assertIn("--epochs 5", text)
        self.assertIn("--lr 0.00001", text)
        self.assertIn("--backbone-lr 0.000001", text)
        self.assertIn("--min-lr 0.0000001", text)
        self.assertNotIn("--resume", command)

    def test_smoke_command_disables_checkpoints(self):
        command = build_command(
            source=Path("source.pth"), output_dir=Path("output"), device="cpu", smoke=True,
        )
        self.assertIn("--no-checkpoints", command)
        self.assertEqual(command[command.index("--epochs") + 1], "1")

    def test_source_validation_requires_stage_b_unfreezing(self):
        checkpoint = {
            "model": {}, "epoch": 20, "best_miou": 0.58,
            "config": {
                "architecture": "deeplabv3plus", "backbone": "mobilenet_v3_large",
                "image_size": 384, "loss": "ce_dice", "unfreeze_last_blocks": 0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "not the stage-B model"):
                load_and_validate_source(path)


if __name__ == "__main__":
    unittest.main()
