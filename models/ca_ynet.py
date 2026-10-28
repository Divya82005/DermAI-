import torch
import torch.nn as nn

from .encoder import ResNet34Encoder
from .classification_decoder import ClassificationDecoder
from .segmentation_decoder import SegmentationDecoder


class CAYNet(nn.Module):

    def __init__(
        self,
        num_classes=3,
        pretrained_encoder=True
    ):
        super().__init__()

        # Shared ResNet-34 encoder
        self.encoder = ResNet34Encoder(
            pretrained=pretrained_encoder
        )

        # Classification branch
        self.classification = ClassificationDecoder(
            num_classes=num_classes
        )

        # Segmentation branch
        self.segmentation = SegmentationDecoder()

    def forward(self, x):

        # Shared encoder
        e1, e2, e3, e4, e5 = self.encoder(x)

        # Classification
        z_cls, class_logits = self.classification(e5)

        # Segmentation
        z_seg, seg_logits, aux_masks = self.segmentation(
            e1,
            e2,
            e3,
            e4,
            e5
        )

        return {
            "z_cls": z_cls,
            "class_logits": class_logits,

            "z_seg": z_seg,
            "seg_logits": seg_logits,

            "aux_masks": aux_masks,

            "features": (
                e1,
                e2,
                e3,
                e4,
                e5
            )
        }