import unittest
from pathlib import Path

from run_deeplabv3plus_resnet50 import build_command


class TestResNet50Runner(unittest.TestCase):
    def test_formal_command_changes_only_backbone(self):
        command = build_command(
            output_dir=Path("output"), device="cuda", batch_size=4, smoke=False,
        )
        text = " ".join(command)
        self.assertIn("--architecture deeplabv3plus", text)
        self.assertIn("--backbone resnet50", text)
        self.assertIn("--freeze-backbone", command)
        self.assertIn("--epochs 20", text)
        self.assertNotIn("--unfreeze-last-blocks", command)

    def test_smoke_command_disables_checkpoints(self):
        command = build_command(
            output_dir=Path("output"), device="cpu", batch_size=2, smoke=True,
        )
        self.assertIn("--no-checkpoints", command)
        self.assertEqual(command[command.index("--epochs") + 1], "1")


if __name__ == "__main__":
    unittest.main()
