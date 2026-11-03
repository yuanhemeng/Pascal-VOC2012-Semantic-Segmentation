import unittest

import torch

from src.model import DeepLabV3PlusResNet50, build_segmentation_model
from train_baseline import configure_backbone_training


class TestDeepLabV3PlusResNet50(unittest.TestCase):
    def test_feature_scales_and_channels(self):
        model = DeepLabV3PlusResNet50().eval()
        features = torch.zeros(1, 3, 128, 128)
        low = high = None
        with torch.inference_mode():
            for name, block in model.backbone.named_children():
                features = block(features)
                if name == "layer1":
                    low = features
                if name == "layer4":
                    high = features
        self.assertEqual(tuple(low.shape), (1, 256, 32, 32))
        self.assertEqual(tuple(high.shape), (1, 2048, 8, 8))

    def test_output_matches_input_resolution(self):
        model = build_segmentation_model(
            architecture="deeplabv3plus", backbone="resnet50",
        ).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 128, 128))["out"]
        self.assertEqual(tuple(output.shape), (1, 21, 128, 128))

    def test_frozen_encoder_keeps_decoder_trainable(self):
        model = DeepLabV3PlusResNet50()
        selected = configure_backbone_training(model, freeze_backbone=True, unfreeze_last_blocks=0)
        self.assertEqual(selected, [])
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))


if __name__ == "__main__":
    unittest.main()
