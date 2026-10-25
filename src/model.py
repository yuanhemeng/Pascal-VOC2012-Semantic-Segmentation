"""烟雾检查和后续训练脚本共享的模型构建器."""

from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

from src.data_preprocessing import NUM_CLASSES


def build_deeplabv3(*, pretrained_backbone: bool = False):
    """Build a 21-class model without downloading segmentation weights.

    The one-batch smoke check uses a randomly initialized backbone. Set
    pretrained_backbone=True for a future experiment after documenting and
    obtaining the ImageNet backbone weights.
    """
    return deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
        num_classes=NUM_CLASSES,
        aux_loss=False,
    )
