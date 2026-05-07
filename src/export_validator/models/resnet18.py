"""ResNet-18 builder using torchvision weights (cached via ``torch.hub``)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_resnet18(pretrained: bool = True) -> tuple[nn.Module, torch.Tensor]:
    """Return an eval-mode ResNet-18 plus a deterministic sample input."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.eval()
    # Deterministic sample. Real ImageNet preprocessing is not relevant for
    # numerical parity; only that PyTorch and ONNX Runtime see the same bytes.
    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(1, 3, 224, 224, generator=generator)
    return model, sample
