"""Segmentation model builders shared by training and analysis scripts."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights, mobilenet_v3_large
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP

from src.data_preprocessing import NUM_CLASSES


ARCHITECTURES = ("deeplabv3", "deeplabv3plus")


class DeepLabV3PlusDecoder(nn.Module):
    """Fuse ASPP context with output-stride-4 MobileNetV3 features."""

    def __init__(self, high_channels: int = 960, low_channels: int = 24) -> None:
        super().__init__()
        self.aspp = ASPP(high_channels, atrous_rates=(12, 24, 36), out_channels=256)
        self.low_projection = nn.Sequential(
            nn.Conv2d(low_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, NUM_CLASSES, kernel_size=1),
        )

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_projection(low)
        return self.fusion(torch.cat((high, low), dim=1))


class DeepLabV3PlusMobileNetV3(nn.Module):
    """A lightweight DeepLabV3+ style model with a MobileNetV3-Large encoder."""

    low_level_index = 3

    def __init__(self, pretrained_backbone: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.backbone = mobilenet_v3_large(weights=weights, dilated=True).features
        self.classifier = DeepLabV3PlusDecoder(high_channels=960, low_channels=24)

    def forward(self, inputs: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        input_size = inputs.shape[-2:]
        low = None
        features = inputs
        for index, block in enumerate(self.backbone):
            features = block(features)
            if index == self.low_level_index:
                low = features
        if low is None:
            raise RuntimeError("MobileNetV3 low-level feature was not produced")
        logits = self.classifier(features, low)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return OrderedDict(out=logits)


def build_deeplabv3(*, pretrained_backbone: bool = False, backbone: str = "resnet50"):
    """Build the existing 21-class torchvision DeepLabV3 baseline."""
    if backbone == "resnet50":
        builder = deeplabv3_resnet50
        backbone_weights = ResNet50_Weights.IMAGENET1K_V1
    elif backbone == "mobilenet_v3_large":
        builder = deeplabv3_mobilenet_v3_large
        backbone_weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return builder(
        weights=None,
        weights_backbone=backbone_weights if pretrained_backbone else None,
        num_classes=NUM_CLASSES,
        aux_loss=False,
    )


def build_segmentation_model(
    *, architecture: str = "deeplabv3", pretrained_backbone: bool = False,
    backbone: str = "mobilenet_v3_large",
):
    """Build a supported segmentation architecture without loading head weights."""
    if architecture == "deeplabv3":
        return build_deeplabv3(pretrained_backbone=pretrained_backbone, backbone=backbone)
    if architecture == "deeplabv3plus":
        if backbone != "mobilenet_v3_large":
            raise ValueError("deeplabv3plus currently supports only mobilenet_v3_large")
        return DeepLabV3PlusMobileNetV3(pretrained_backbone=pretrained_backbone)
    raise ValueError(f"Unsupported architecture: {architecture}")
