import torch
import torch.nn as nn

from .cbam import CBAM


class ClassificationAttentionBlock(nn.Module):
    """
    Classification Attention Block from CA Y-Net.

    Input:
        e5 -> [B, 512, H/32, W/32]

    The block contains:
        bottleneck residual processing
        -> 1x1 convolution
        -> BatchNorm
        -> CBAM
        -> residual addition with e5
        -> ReLU
    """

    def __init__(self, channels=512, bottleneck_channels=128):
        super().__init__()

        # Bottleneck residual branch
        self.conv1 = nn.Conv2d(
            channels,
            bottleneck_channels,
            kernel_size=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(bottleneck_channels)

        self.dropout = nn.Dropout2d(p=0.2)

        # Expand back to 512 channels before CBAM
        self.expand = nn.Conv2d(
            bottleneck_channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.bn_expand = nn.BatchNorm2d(channels)

        # CBAM
        self.cbam = CBAM(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, e5):

        # Bottleneck residual processing
        x = self.conv1(e5)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.dropout(x)

        # Restore original channel dimension
        x = self.expand(x)
        x = self.bn_expand(x)

        # Attention
        x = self.cbam(x)

        # Residual connection with e5
        x = x + e5

        x = self.relu(x)

        return x


class ClassificationDecoder(nn.Module):
    """
    CA Y-Net classification decoder.

    Input:
        e5: [B, 512, H/32, W/32]

    Outputs:
        z_cls:
            [B, 128]

        class_logits:
            [B, 3]
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.attention_block = ClassificationAttentionBlock(
            channels=512,
            bottleneck_channels=128
        )

        # Paper:
        # classification attention output
        # -> average pooling
        # -> 512-dimensional vector
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 512 -> 512
        self.fc1 = nn.Linear(
            512,
            512
        )

        # 512 -> 128
        self.fc2 = nn.Linear(
            512,
            128
        )

        self.relu = nn.ReLU(inplace=True)

        # Dropout before final classification layer
        self.dropout = nn.Dropout(p=0.2)

        # 128 -> 3 lesion classes
        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self, e5):

        # Classification attention block
        x = self.attention_block(e5)

        # Global average pooling
        x = self.global_pool(x)

        # [B, 512, 1, 1]
        # -> [B, 512]
        x = torch.flatten(
            x,
            start_dim=1
        )

        # First fully connected layer
        x = self.fc1(x)
        x = self.relu(x)

        # Second fully connected layer
        # This is the 128-D classification representation
        z_cls = self.fc2(x)
        z_cls = self.relu(z_cls)

        # Classification prediction
        x = self.dropout(z_cls)

        class_logits = self.classifier(x)

        return z_cls, class_logits