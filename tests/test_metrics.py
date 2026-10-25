import unittest

import torch

from src.metrics import SegmentationMetrics, restore_logits


class TestMetrics(unittest.TestCase):
    def test_ignore_value_and_absent_class(self):
        metric = SegmentationMetrics(num_classes=3)
        metric.update(torch.tensor([[0, 0, 2]]), torch.tensor([[0, 1, 255]]))
        self.assertAlmostEqual(metric.mean_iou(), 0.25)
        self.assertAlmostEqual(metric.pixel_accuracy(), 0.5)
        self.assertTrue(torch.isnan(metric.per_class_iou()[2]))

    def test_restore_excludes_padding(self):
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 1, 2:, :] = 100
        restored = restore_logits(logits, valid_size=(2, 4), original_size=(3, 6))
        self.assertEqual(tuple(restored.shape), (1, 2, 3, 6))
        self.assertEqual(restored[:, 1].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
