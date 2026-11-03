import unittest

import torch

from train_baseline import (
    boundary_weighted_cross_entropy,
    multiclass_dice_loss,
    segmentation_boundary_mask,
    segmentation_loss,
)


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

    def test_boundary_mask_detects_and_dilates_transition(self):
        target = torch.zeros(1, 7, 7, dtype=torch.long)
        target[:, :, 4:] = 1
        narrow = segmentation_boundary_mask(target, width=0)
        wide = segmentation_boundary_mask(target, width=2)
        self.assertTrue(narrow[:, :, 3:5].all())
        self.assertGreater(wide.sum().item(), narrow.sum().item())

    def test_boundary_weighted_ce_ignores_void_pixels(self):
        logits = torch.zeros(1, 2, 3, 3, requires_grad=True)
        target = torch.tensor([[[0, 0, 1], [0, 255, 1], [0, 0, 1]]])
        loss = boundary_weighted_cross_entropy(logits, target, 2.0, 1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_boundary_ce_dice_is_finite(self):
        logits = torch.randn(2, 21, 8, 8, requires_grad=True)
        target = torch.randint(0, 21, (2, 8, 8))
        loss = segmentation_loss(logits, target, "boundary_ce_dice", 1.0, 2.0, 3)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
