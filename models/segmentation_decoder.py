import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM


class DoubleConv(nn.Module):
    """
    Double convolution block used in the CA Y-Net
    segmentation decoder.

    3x3 Conv
    -> BatchNorm
    -> ReLU
    -> 3x3 Conv
    -> BatchNorm
    -> ReLU
    -> Dropout2d
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.2
    ):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout2d(
                p=dropout
            )
        )

    def forward(self, x):
        return self.block(x)


class SegmentationDecoder(nn.Module):
    """
    CA Y-Net segmentation decoder.

    Encoder features:

        e1 = [B,  64, H/2,  W/2]
        e2 = [B,  64, H/4,  W/4]
        e3 = [B, 128, H/8,  W/8]
        e4 = [B, 256, H/16, W/16]
        e5 = [B, 512, H/32, W/32]

    Decoder:

        d1 = 512 channels
        d2 = 256 channels
        d3 = 128 channels
        d4 = 64 channels
        d5 = 64 channels

    Four shortcut pathways:
        d1 -> 32 channels
        d2 -> 32 channels
        d3 -> 32 channels
        d4 -> 32 channels

    Final fusion:
        d5      = 64
        shortcut1 = 32
        shortcut2 = 32
        shortcut3 = 32
        shortcut4 = 32

        total = 192 channels

        192 -> 128 -> 1
    """

    def __init__(self):

        super().__init__()

        # -------------------------------------------------
        # Dropout on deepest encoder feature e5
        # -------------------------------------------------

        self.dropout_e5 = nn.Dropout2d(
            p=0.2
        )

        # -------------------------------------------------
        # Five CBAM blocks
        #
        # One for each encoder feature e1-e5
        # -------------------------------------------------

        self.cbam_e1 = CBAM(64)

        self.cbam_e2 = CBAM(64)

        self.cbam_e3 = CBAM(128)

        self.cbam_e4 = CBAM(256)

        self.cbam_e5 = CBAM(512)

        # -------------------------------------------------
        # Decoder stage 1
        #
        # d1: 512 channels
        #
        # 512 -> 256 through transposed convolution
        # concatenate with e4 (256)
        # 512 -> 256 through DoubleConv
        # -------------------------------------------------

        self.up1 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.double_conv1 = DoubleConv(
            512,
            256
        )

        # -------------------------------------------------
        # Decoder stage 2
        #
        # d2: 256
        #
        # 256 -> 128
        # concatenate with e3 (128)
        # 256 -> 128
        # -------------------------------------------------

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.double_conv2 = DoubleConv(
            256,
            128
        )

        # -------------------------------------------------
        # Decoder stage 3
        #
        # d3: 128
        #
        # 128 -> 64
        # concatenate with e2 (64)
        # 128 -> 64
        # -------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.double_conv3 = DoubleConv(
            128,
            64
        )

        # -------------------------------------------------
        # Decoder stage 4
        #
        # d4: 64
        #
        # 64 -> 64
        # concatenate with e1 (64)
        # 128 -> 64
        # -------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            64,
            64,
            kernel_size=2,
            stride=2
        )

        self.double_conv4 = DoubleConv(
            128,
            64
        )

        # -------------------------------------------------
        # Four shortcut pathways
        #
        # Each decoder output -> 32 channels
        # -------------------------------------------------

        self.shortcut1 = nn.Conv2d(
            512,
            32,
            kernel_size=1
        )

        self.shortcut2 = nn.Conv2d(
            256,
            32,
            kernel_size=1
        )

        self.shortcut3 = nn.Conv2d(
            128,
            32,
            kernel_size=1
        )

        self.shortcut4 = nn.Conv2d(
            64,
            32,
            kernel_size=1
        )

        # -------------------------------------------------
        # Four auxiliary segmentation heads
        #
        # Each 32-channel shortcut -> 1 channel
        # -------------------------------------------------

        self.aux_head1 = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        self.aux_head2 = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        self.aux_head3 = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        self.aux_head4 = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

        # -------------------------------------------------
        # Final fusion
        #
        # 64 + 32 + 32 + 32 + 32 = 192
        #
        # 192 -> 128 = z_seg
        # 128 -> 1 = lesion map
        # -------------------------------------------------

        self.fusion = nn.Conv2d(
            192,
            128,
            kernel_size=1
        )

        self.segmentation_head = nn.Conv2d(
            128,
            1,
            kernel_size=1
        )

    @staticmethod
    def bicubic_resize(
        x,
        size
    ):
        """
        Bicubic interpolation exactly as described
        for the shortcut pathways and d5.
        """

        return F.interpolate(
            x,
            size=size,
            mode="bicubic",
            align_corners=False
        )

    def forward(
        self,
        e1,
        e2,
        e3,
        e4,
        e5
    ):

        # =================================================
        # 1. CBAM processing of encoder features
        # =================================================

        e1_att = self.cbam_e1(e1)

        e2_att = self.cbam_e2(e2)

        e3_att = self.cbam_e3(e3)

        e4_att = self.cbam_e4(e4)

        # Paper applies Dropout2D to e5 before CBAM
        e5_drop = self.dropout_e5(e5)

        e5_att = self.cbam_e5(e5_drop)

        # =================================================
        # 2. d1
        # =================================================

        d1 = e5_att

        # =================================================
        # 3. d1 -> d2
        # =================================================

        x = self.up1(d1)

        x = torch.cat(
            [x, e4_att],
            dim=1
        )

        d2 = self.double_conv1(x)

        # =================================================
        # 4. d2 -> d3
        # =================================================

        x = self.up2(d2)

        x = torch.cat(
            [x, e3_att],
            dim=1
        )

        d3 = self.double_conv2(x)

        # =================================================
        # 5. d3 -> d4
        # =================================================

        x = self.up3(d3)

        x = torch.cat(
            [x, e2_att],
            dim=1
        )

        d4 = self.double_conv3(x)

        # =================================================
        # 6. d4 -> d5
        # =================================================

        x = self.up4(d4)

        x = torch.cat(
            [x, e1_att],
            dim=1
        )

        d5 = self.double_conv4(x)

        # =================================================
        # 7. Full image resolution
        # =================================================

        # e1 is H/2 x W/2.
        # Therefore original input resolution is:
        #
        # H x W

        full_size = (
            e1.shape[-2] * 2,
            e1.shape[-1] * 2
        )

        # =================================================
        # 8. Shortcut pathway 1
        # =================================================

        s1 = self.shortcut1(d1)

        s1 = self.bicubic_resize(
            s1,
            full_size
        )

        aux1 = self.aux_head1(s1)

        # =================================================
        # 9. Shortcut pathway 2
        # =================================================

        s2 = self.shortcut2(d2)

        s2 = self.bicubic_resize(
            s2,
            full_size
        )

        aux2 = self.aux_head2(s2)

        # =================================================
        # 10. Shortcut pathway 3
        # =================================================

        s3 = self.shortcut3(d3)

        s3 = self.bicubic_resize(
            s3,
            full_size
        )

        aux3 = self.aux_head3(s3)

        # =================================================
        # 11. Shortcut pathway 4
        # =================================================

        s4 = self.shortcut4(d4)

        s4 = self.bicubic_resize(
            s4,
            full_size
        )

        aux4 = self.aux_head4(s4)

        # =================================================
        # 12. Upsample d5 to original image resolution
        # =================================================

        d5_full = self.bicubic_resize(
            d5,
            full_size
        )

        # =================================================
        # 13. Feature fusion
        #
        # d5       = 64
        # s1       = 32
        # s2       = 32
        # s3       = 32
        # s4       = 32
        #
        # total    = 192
        # =================================================

        fused = torch.cat(
            [
                d5_full,
                s1,
                s2,
                s3,
                s4
            ],
            dim=1
        )

        # =================================================
        # 14. Generate z_seg
        # =================================================

        z_seg = self.fusion(
            fused
        )

        # =================================================
        # 15. Generate final lesion map
        # =================================================

        seg_logits = self.segmentation_head(
            z_seg
        )

        # =================================================
        # Return
        # =================================================

        aux_masks = [
            aux1,
            aux2,
            aux3,
            aux4
        ]

        return (
            z_seg,
            seg_logits,
            aux_masks
        )