"""针对分割数据的小型测试"""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.data_preprocessing import VOCPreprocessor


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.zeros((4, 6, 3), dtype=np.uint8), mode="RGB")
        self.mask = Image.fromarray(np.tile(np.array([1, 1, 1, 2, 2, 2], dtype=np.uint8), (4, 1)), mode="P")

    def test_train_mask_keeps_class_ids_and_shape(self):
        with patch("src.data_preprocessing.random.uniform", return_value=1.0), patch(
            "src.data_preprocessing.random.randint", side_effect=[1, 0]
        ), patch("src.data_preprocessing.random.random", return_value=1.0):
            item = VOCPreprocessor(4, "train")(self.image, self.mask)
        self.assertEqual(tuple(item["image"].shape), (3, 4, 4))
        self.assertEqual(item["mask"].tolist()[0], [1, 1, 2, 2])

    def test_validation_keeps_original_mask(self):
        item = VOCPreprocessor(8, "val")(self.image, self.mask)
        self.assertEqual(tuple(item["image"].shape), (3, 8, 8))
        self.assertEqual(tuple(item["mask"].shape), (4, 6))
        self.assertEqual(item["valid_size"], (5, 8))
        self.assertEqual(item["original_size"], (4, 6))

    def test_test_mode_has_no_mask(self):
        item = VOCPreprocessor(8, "test")(self.image)
        self.assertIsNone(item["mask"])
        self.assertEqual(item["original_size"], (4, 6))

    def test_rejects_rgb_mask(self):
        with self.assertRaisesRegex(ValueError, "class-index mask"):
            VOCPreprocessor(8, "val")(self.image, self.image)


if __name__ == "__main__":
    unittest.main()
