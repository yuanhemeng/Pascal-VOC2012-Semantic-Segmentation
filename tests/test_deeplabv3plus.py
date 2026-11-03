import unittest

import torch

from src.model import DeepLabV3PlusMobileNetV3, build_segmentation_model
from train_baseline import configure_backbone_training


class TestDeepLabV3Plus(unittest.TestCase):
    def test_output_matches_input_resolution(self):
        model = build_segmentation_model(
            architecture="deeplabv3plus", backbone="mobilenet_v3_large",
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 128, 128))["out"]
        self.assertEqual(tuple(output.shape), (2, 21, 128, 128))

    def test_low_level_feature_is_output_stride_four(self):
        model = DeepLabV3PlusMobileNetV3().eval()
        features = torch.zeros(1, 3, 128, 128)
        with torch.inference_mode():
            for index, block in enumerate(model.backbone):
                features = block(features)
                if index == model.low_level_index:
                    break
        self.assertEqual(tuple(features.shape), (1, 24, 32, 32))

    def test_partial_unfreeze_keeps_decoder_trainable(self):
        model = DeepLabV3PlusMobileNetV3()
        selected = configure_backbone_training(model, freeze_backbone=True, unfreeze_last_blocks=3)
        self.assertEqual(selected, ["14", "15", "16"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_rejects_unknown_backbone(self):
        with self.assertRaisesRegex(ValueError, "Unsupported DeepLabV3\+ backbone"):
            build_segmentation_model(architecture="deeplabv3plus", backbone="unknown")


if __name__ == "__main__":
    unittest.main()
