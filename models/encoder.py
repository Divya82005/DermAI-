import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Encoder(nn.Module):
    """
    Shared ResNet-34 encoder.

    Outputs:

    e1: H/2  × W/2  × 64
    e2: H/4  × W/4  × 64
    e3: H/8  × W/8  × 128
    e4: H/16 × W/16 × 256
    e5: H/32 × W/32 × 512
    """

    def __init__(self, pretrained=False):
        super().__init__()

        weights = (
            ResNet34_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = resnet34(
            weights=weights
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):

        # -----------------------------------------
        # e1
        # -----------------------------------------

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        e1 = x

        # -----------------------------------------
        # e2
        # -----------------------------------------

        x = self.maxpool(x)

        x = self.layer1(x)

        e2 = x

        # -----------------------------------------
        # e3
        # -----------------------------------------

        x = self.layer2(x)

        e3 = x

        # -----------------------------------------
        # e4
        # -----------------------------------------

        x = self.layer3(x)

        e4 = x

        # -----------------------------------------
        # e5
        # -----------------------------------------

        x = self.layer4(x)

        e5 = x

        return e1, e2, e3, e4, e5