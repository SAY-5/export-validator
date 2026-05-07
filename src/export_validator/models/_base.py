"""Shared model-builder protocol."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class ModelBuilder(Protocol):
    """A function that returns ``(model, sample_input)``."""

    def __call__(self) -> tuple[nn.Module, torch.Tensor]: ...
