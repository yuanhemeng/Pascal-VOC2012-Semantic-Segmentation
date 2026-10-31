import unittest

from src.model import build_deeplabv3
from train_baseline import configure_backbone_training


class TestBackboneFinetune(unittest.TestCase):
    def test_only_last_three_blocks_are_trainable(self):
        model = build_deeplabv3(backbone="mobilenet_v3_large")
        selected = configure_backbone_training(model, freeze_backbone=True, unfreeze_last_blocks=3)
        self.assertEqual(selected, ["14", "15", "16"])
        for name, block in model.backbone.named_children():
            self.assertEqual(any(p.requires_grad for p in block.parameters()), name in selected)
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))


if __name__ == "__main__":
    unittest.main()
