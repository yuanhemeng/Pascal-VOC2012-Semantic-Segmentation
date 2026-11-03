import unittest
from pathlib import Path
from run_boundary_loss import build_command

class TestBoundaryRunner(unittest.TestCase):
    def test_formal_command(self):
        command = build_command(Path("source.pth"), Path("output"), "cuda", False)
        text = " ".join(command)
        self.assertIn("--loss boundary_ce_dice", text)
        self.assertIn("--boundary-weight 2.0", text)
        self.assertIn("--boundary-width 3", text)
        self.assertIn("--epochs 5", text)
        self.assertNotIn("--resume", command)

    def test_smoke_disables_checkpoints(self):
        command = build_command(Path("source.pth"), Path("output"), "cpu", True)
        self.assertIn("--no-checkpoints", command)
        self.assertEqual(command[command.index("--epochs") + 1], "1")

if __name__ == "__main__": unittest.main()
