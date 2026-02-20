from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .parts import DoubleConv, Down, OutConv, Up


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self._use_checkpointing = False

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Masking convolution
        self.outc = OutConv(64, n_classes)

    def use_checkpointing(self, enabled: bool = True) -> None:
        self._use_checkpointing = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_checkpointing:
            x1 = checkpoint(self.inc, x)
            x2 = checkpoint(self.down1, x1)
            x3 = checkpoint(self.down2, x2)
            x4 = checkpoint(self.down3, x3)
            x5 = checkpoint(self.down4, x4)

            x = checkpoint(self.up1, x5, x4)
            x = checkpoint(self.up2, x, x3)
            x = checkpoint(self.up3, x, x2)
            x = checkpoint(self.up4, x, x1)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        return self.outc(x)
