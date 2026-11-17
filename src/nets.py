"""Network definitions for models 2 and 3."""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from common import DATA


def _conv_bn(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """Compact CNN for 12 x 20 x 20 Sentinel-2 patches (about 0.29 M parameters)."""

    def __init__(self, in_ch: int = 12, n_classes: int = 4, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _conv_bn(in_ch, w), _conv_bn(w, w), nn.MaxPool2d(2),  # 20 -> 10
            _conv_bn(w, 2 * w), _conv_bn(2 * w, 2 * w), nn.MaxPool2d(2),  # 10 -> 5
            _conv_bn(2 * w, 4 * w), _conv_bn(4 * w, 4 * w),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# Aerial band order (BRIEF.md and DECISIONS.md D10): 0 = NIR, 1 = green, 2 = blue, 3 = red.
AERIAL_BANDS = ["NIR", "G", "B", "R"]


def resnet18_4band(n_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ImageNet ResNet-18 with the first convolution widened to 4 input bands.

    Pretrained RGB filters are copied to the matching aerial bands; the NIR
    filter starts as the mean of the RGB filters. All four are scaled by 3/4
    so the first-layer activation scale stays about the same as with 3 bands.
    """
    os.environ.setdefault("TORCH_HOME", str(DATA / "torch_home"))  # keep weights inside the project
    from torchvision.models import ResNet18_Weights, resnet18

    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    w = m.conv1.weight.data  # (64, 3, 7, 7), RGB
    new = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        r, g, b = w[:, 0], w[:, 1], w[:, 2]
        new.weight[:, 0] = w.mean(1)  # NIR
        new.weight[:, 1] = g
        new.weight[:, 2] = b
        new.weight[:, 3] = r
        new.weight *= 0.75
    m.conv1 = new
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def freeze_resnet_except(m: nn.Module, trainable: tuple[str, ...]) -> list[nn.Parameter]:
    """Freeze all parameters except those in the named top-level children. Returns trainable params."""
    params = []
    for name, child in m.named_children():
        req = name in trainable
        for p in child.parameters():
            p.requires_grad = req
            if req:
                params.append(p)
    return params
