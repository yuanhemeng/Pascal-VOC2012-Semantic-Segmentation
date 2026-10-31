import unittest

import torch

from train_baseline import multiclass_dice_loss, segmentation_loss


class TestSegmentationLosses(unittest.TestCase):
    def test_dice_loss_is_finite_and_ignores_void_pixels(self):
        logits = torch.randn(1, 21, 4, 4, requires_grad=True)
        target = torch.tensor([[[0, 1, 255, 2], [0, 1, 2, 2], [3, 3, 3, 3], [0, 0, 1, 1]]])
        loss = multiclass_dice_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_combined_loss_is_at_least_cross_entropy(self):
        logits = torch.randn(1, 21, 4, 4)
        target = torch.randint(0, 21, (1, 4, 4))
        ce = segmentation_loss(logits, target, "ce", 1.0)
        combined = segmentation_loss(logits, target, "ce_dice", 1.0)
        self.assertGreaterEqual(combined.item(), ce.item())


if __name__ == "__main__":
    unittest.main()
