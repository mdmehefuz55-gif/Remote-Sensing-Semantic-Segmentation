"""DeepLabV3+ Semantic Segmentation Architecture for Remote Sensing.

Implements DeepLabV3+ with:
- ResNet backbone (ResNet-34 / ResNet-50 with dilated/atrous convolutions).
- Atrous Spatial Pyramid Pooling (ASPP) module with multi-rate dilations and image pooling.
- Low-level feature fusion decoder.
- High-resolution segmentation head tailored for LoveDA remote sensing imagery.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPPConv(nn.Sequential):
    """Standard 3x3 atrous convolution block with specified dilation rate."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    """Global average pooling branch for ASPP."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (ASPP) module."""

    def __init__(self, in_channels: int, atrous_rates: List[int] = [6, 12, 18], out_channels: int = 256):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ Architecture for Semantic Segmentation.

    Parameters
    ----------
    num_classes : int
        Number of output segmentation classes (default: 8).
    backbone : str
        Backbone architecture ('resnet34' or 'resnet50').
    pretrained : bool
        Whether to load torchvision pretrained weights for the backbone.
    atrous_rates : List[int]
        Dilation rates for the ASPP module (default: [6, 12, 18] for output stride 16).
    """

    def __init__(
        self,
        num_classes: int = 8,
        backbone: str = "resnet50",
        pretrained: bool = False,
        atrous_rates: List[int] = [6, 12, 18],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone.lower()

        # Build backbone with dilated convolutions
        if self.backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            bb = models.resnet34(weights=weights)
            low_level_channels = 64
            high_level_channels = 512
        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            # Replace stride with dilation in layer4 for output stride 16
            bb = models.resnet50(
                weights=weights,
                replace_stride_with_dilation=[False, False, True],
            )
            low_level_channels = 256
            high_level_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet34' or 'resnet50'.")

        self.stem = nn.Sequential(
            bb.conv1,
            bb.bn1,
            bb.relu,
            bb.maxpool,
        )
        self.layer1 = bb.layer1  # Low-level features (1/4 spatial resolution)
        self.layer2 = bb.layer2  # (1/8 resolution)
        self.layer3 = bb.layer3  # (1/16 resolution)
        self.layer4 = bb.layer4  # High-level features (1/16 resolution via dilation)

        # ASPP Module
        self.aspp = ASPP(
            in_channels=high_level_channels,
            atrous_rates=atrous_rates,
            out_channels=256,
        )

        # Low-level feature projection
        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        # Backbone feature extraction
        x_stem = self.stem(x)
        low_level_feat = self.layer1(x_stem)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        high_level_feat = self.layer4(x)

        # ASPP feature extraction
        aspp_feat = self.aspp(high_level_feat)
        aspp_feat_up = F.interpolate(
            aspp_feat,
            size=low_level_feat.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        # Low-level feature projection and concatenation
        low_level_proj = self.low_level_project(low_level_feat)
        fused = torch.cat([aspp_feat_up, low_level_proj], dim=1)

        # Decoder and final classification
        decoder_out = self.decoder(fused)
        logits = F.interpolate(
            decoder_out,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )

        return logits
