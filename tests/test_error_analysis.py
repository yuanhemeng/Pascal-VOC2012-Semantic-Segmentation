import math
import unittest

import torch

from src.error_analysis import class_metrics, sample_class_metrics, select_worst_samples


class TestErrorAnalysis(unittest.TestCase):
    def test_class_metrics_reports_recall_iou_and_confusion(self):
        confusion = torch.tensor([[8, 2, 0], [1, 3, 1], [0, 2, 3]])
        rows = class_metrics(confusion, ("a", "b", "c"))
        self.assertAlmostEqual(rows[1]["recall"], 3 / 5)
        self.assertAlmostEqual(rows[1]["precision"], 3 / 7)
        self.assertAlmostEqual(rows[1]["iou"], 3 / 9)
        self.assertEqual(rows[1]["most_confused_with"], "a")

    def test_sample_metrics_ignores_void_and_finds_dominant_error(self):
        target = torch.tensor([[1, 1, 1, 255], [0, 0, 2, 2]])
        prediction = torch.tensor([[1, 0, 0, 2], [0, 2, 2, 1]])
        rows = sample_class_metrics(prediction, target, "sample", ("a", "b", "c"))
        class_one = next(row for row in rows if row["class_id"] == 1)
        self.assertAlmostEqual(class_one["recall"], 1 / 3)
        self.assertEqual(class_one["most_confused_with"], "a")
        self.assertEqual(class_one["confused_pixels"], 2)

    def test_worst_selection_applies_pixel_threshold_and_rank(self):
        rows = [
            {"sample_id": "a", "class_id": 1, "gt_pixels": 100, "recall": 0.2, "iou": 0.2},
            {"sample_id": "b", "class_id": 1, "gt_pixels": 10, "recall": 0.0, "iou": 0.0},
            {"sample_id": "c", "class_id": 1, "gt_pixels": 100, "recall": 0.1, "iou": 0.1},
        ]
        selected = select_worst_samples(rows, [1], samples_per_class=2, min_gt_pixels=64)
        self.assertEqual([row["sample_id"] for row in selected], ["c", "a"])
        self.assertEqual([row["rank"] for row in selected], [1, 2])


if __name__ == "__main__":
    unittest.main()
