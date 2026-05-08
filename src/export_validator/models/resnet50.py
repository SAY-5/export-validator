"""ResNet-50 builder. Surfaces accumulated drift from a deeper ResNet."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_resnet50(pretrained: bool = True) -> tuple[nn.Module, torch.Tensor]:
    """Return an eval-mode ResNet-50 plus a deterministic sample input."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.eval()
    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(1, 3, 224, 224, generator=generator)
    return model, sample
