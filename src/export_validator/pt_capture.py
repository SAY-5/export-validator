"""Capture per-layer activations from a PyTorch model via forward hooks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .instrument import NamingHooks


def run(model: nn.Module, inputs: torch.Tensor) -> dict[str, npt.NDArray[np.float32]]:
    """Run ``inputs`` through ``model`` and return ``{layer_name: ndarray}``.

    The returned arrays are detached, on CPU, and converted to ``float32``.
    """
    model.eval()
    with NamingHooks(model) as hooks, torch.no_grad():
        model(inputs)
        return {
            name: tensor.detach().cpu().to(torch.float32).numpy()
            for name, tensor in hooks.captures.items()
        }
