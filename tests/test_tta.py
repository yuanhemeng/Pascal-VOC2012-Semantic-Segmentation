import argparse
import unittest

import torch
from PIL import Image

from evaluate_tta import parse_scales, tta_logits


class RedChannelModel(torch.nn.Module):
    def forward(self, inputs):
        red = inputs[:, :1]
        zeros = torch.zeros_like(red)
        return {"out": torch.cat((red, zeros), dim=1)}


class TestTTA(unittest.TestCase):
    def test_parse_scales(self):
        self.assertEqual(parse_scales("0.75,1,1.25"), (0.75, 1.0, 1.25))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_scales("1,1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_scales("0,-1")

    def test_flip_ensemble_restores_orientation(self):
        image = Image.new("RGB", (8, 6), color=(255, 0, 0))
        logits = tta_logits(
            RedChannelModel().eval(), image, torch.device("cpu"),
            base_size=8, scales=(1.0,), horizontal_flip=True,
        )
        self.assertEqual(tuple(logits.shape), (2, 6, 8))
        self.assertTrue(torch.all(logits[0] > logits[1]))


if __name__ == "__main__":
    unittest.main()
