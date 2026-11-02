import tempfile
import unittest
from pathlib import Path
import torch

from run_deeplabv3plus_finetune import build_command, load_and_validate_source


class TestStageBRunner(unittest.TestCase):
    def test_command_has_controlled_finetune_settings(self):
        command = build_command(
            source=Path("source.pth"), output_dir=Path("output"), device="cuda", smoke=False,
        )
        text = " ".join(command)
        self.assertIn("--architecture deeplabv3plus", text)
        self.assertIn("--unfreeze-last-blocks 3", text)
        self.assertIn("--lr 0.0001", text)
        self.assertIn("--backbone-lr 0.00001", text)
        self.assertIn("--epochs 10", text)
        self.assertNotIn("--max-train-batches", command)

    def test_smoke_command_cannot_save_partial_checkpoint(self):
        command = build_command(
            source=Path("source.pth"), output_dir=Path("output"), device="cpu", smoke=True,
        )
        self.assertIn("--no-checkpoints", command)
        self.assertIn("--max-train-batches", command)
        self.assertEqual(command[command.index("--epochs") + 1], "1")

    def test_source_validation_rejects_wrong_architecture(self):
        checkpoint = {
            "model": {}, "epoch": 1, "best_miou": 0.5,
            "config": {
                "architecture": "deeplabv3", "backbone": "mobilenet_v3_large",
                "image_size": 384, "loss": "ce_dice",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "not the stage-A model"):
                load_and_validate_source(path)


if __name__ == "__main__":
    unittest.main()
