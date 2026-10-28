import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module.

    Learns which feature channels are important.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_attention = self.mlp(
            self.avg_pool(x)
        )

        max_attention = self.mlp(
            self.max_pool(x)
        )

        attention = avg_attention + max_attention

        attention = self.sigmoid(attention)

        return x * attention


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module.

    Learns which spatial regions are important.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        attention = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        attention = self.conv(attention)

        attention = self.sigmoid(attention)

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Channel attention is applied first,
    followed by spatial attention.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channel = ChannelAttention(
            channels,
            reduction
        )

        self.spatial = SpatialAttention(7)

    def forward(self, x):

        x = self.channel(x)

        x = self.spatial(x)

        return x