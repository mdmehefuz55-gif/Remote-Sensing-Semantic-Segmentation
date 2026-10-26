"""U-Net Semantic Segmentation Architecture for Remote Sensing.

Implements classic U-Net encoder-decoder network with skip connections
as well as a ResNet-backbone U-Net option for LoveDA segmentation.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2's spatial dimensions
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping features to number of classes."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Classic U-Net architecture for semantic segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 3 for RGB).
    num_classes : int
        Number of segmentation classes (default: 8 for LoveDA).
    base_features : int
        Number of feature maps in the first layer (default: 64).
    bilinear : bool
        Whether to use bilinear interpolation for upsampling (default: True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        base_features: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        f = base_features
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, (f * 16) // factor)

        self.up1 = Up(f * 16, (f * 8) // factor, bilinear)
        self.up2 = Up(f * 8, (f * 4) // factor, bilinear)
        self.up3 = Up(f * 4, (f * 2) // factor, bilinear)
        self.up4 = Up(f * 2, f, bilinear)
        self.outc = OutConv(f, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResNetUNet(nn.Module):
    """U-Net with a pretrained ResNet-34 backbone for remote sensing segmentation."""

    def __init__(
        self,
        num_classes: int = 8,
        pretrained: bool = False,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ResNet34 encoder
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base_model = models.resnet34(weights=weights)

        if in_channels != 3:
            self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.first_conv = base_model.conv1

        self.first_bn = base_model.bn1
        self.first_relu = base_model.relu
        self.first_pool = base_model.maxpool

        self.encoder1 = base_model.layer1  # 64 channels, 1/4 size
        self.encoder2 = base_model.layer2  # 128 channels, 1/8 size
        self.encoder3 = base_model.layer3  # 256 channels, 1/16 size
        self.encoder4 = base_model.layer4  # 512 channels, 1/32 size

        # Decoder blocks
        self.center = DoubleConv(512, 512)
        self.dec4 = Up(512 + 256, 256)
        self.dec3 = Up(256 + 128, 128)
        self.dec2 = Up(128 + 64, 64)
        self.dec1 = Up(64 + 64, 64)

        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Sequential(
            DoubleConv(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.size()[2:]

        # Initial stem
        x0 = self.first_relu(self.first_bn(self.first_conv(x)))  # 1/2 size, 64 ch
        x_pool = self.first_pool(x0)                             # 1/4 size, 64 ch

        # Encoder stages
        e1 = self.encoder1(x_pool)  # 1/4 size, 64 ch
        e2 = self.encoder2(e1)      # 1/8 size, 128 ch
        e3 = self.encoder3(e2)      # 1/16 size, 256 ch
        e4 = self.encoder4(e3)      # 1/32 size, 512 ch

        # Center bridge & Decoder with skip connections
        c = self.center(e4)
        d4 = self.dec4(c, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        logits = self.final_conv(out)

        if logits.size()[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=True)

        return logits
