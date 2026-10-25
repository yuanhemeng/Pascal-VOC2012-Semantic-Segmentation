"""烟雾检查和后续训练脚本共享的模型构建器."""

from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50

from src.data_preprocessing import NUM_CLASSES


def build_deeplabv3(*, pretrained_backbone: bool = False, backbone: str = "resnet50"):
    """Build a 21-class DeepLabV3 model with the chosen backbone.

    The one-batch smoke check uses a randomly initialized backbone. Set
    pretrained_backbone=True for a future experiment after documenting and
    obtaining the ImageNet backbone weights.
    """
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
