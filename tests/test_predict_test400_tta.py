import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from predict_test400_tta import list_test_images, package_masks, validate_mask


class TestTest400TTAOutput(unittest.TestCase):
    def test_requires_exact_input_count_and_sorts_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 3)).save(root / "b.jpg")
            Image.new("RGB", (4, 3)).save(root / "a.jpg")
            self.assertEqual([p.name for p in list_test_images(root, 2)], ["a.jpg", "b.jpg"])
            with self.assertRaisesRegex(ValueError, "Expected 3"):
                list_test_images(root, 3)

    def test_validates_class_mask_shape_mode_and_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.png"
            Image.fromarray(np.array([[0, 20]], dtype=np.uint8), mode="L").save(good)
            result = validate_mask(good, (2, 1))
            self.assertEqual(result["dtype"], "uint8")
            rgb = root / "rgb.png"
            Image.new("RGB", (2, 1)).save(rgb)
            with self.assertRaisesRegex(ValueError, "one-channel"):
                validate_mask(rgb, (2, 1))
            bad = root / "bad.png"
            Image.fromarray(np.array([[0, 255]], dtype=np.uint8), mode="L").save(bad)
            with self.assertRaisesRegex(ValueError, "0..20"):
                validate_mask(bad, (2, 1))

    def test_packages_flat_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "masks"
            masks.mkdir()
            for name in ("a.png", "b.png"):
                Image.fromarray(np.zeros((2, 3), dtype=np.uint8), mode="L").save(masks / name)
            archive_path = root / "predict.zip"
            package_masks(masks, archive_path, {"a.png", "b.png"})
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(sorted(archive.namelist()), ["a.png", "b.png"])


if __name__ == "__main__":
    unittest.main()
