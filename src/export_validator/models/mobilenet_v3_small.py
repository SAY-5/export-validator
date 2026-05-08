"""MobileNet-V3-Small builder. Different op coverage (depth-wise convs, hardswish)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_mobilenet_v3_small(pretrained: bool = True) -> tuple[nn.Module, torch.Tensor]:
    """Return an eval-mode MobileNet-V3-Small plus a deterministic sample input."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.eval()
    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(1, 3, 224, 224, generator=generator)
    return model, sample
