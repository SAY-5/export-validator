"""ViT-B/16 builder. Transformer architecture — fundamentally different graph shape.

Layers whose output is a tuple (e.g. ``MultiheadAttention`` returning
``(attn_output, attn_weights)``) are filtered out of the named-output set by
``select_exportable_layers`` in the same way they are for CNNs; the
named-output strategy still produces one named ONNX output per **tensor-only**
leaf.

ViT-B/16 has roughly 86M parameters and produces ~100 named leaf outputs in
the torchvision implementation (each encoder block's LayerNorm + MLP
Linear/GELU are leaves; the ``MultiheadAttention`` submodules return tuples
and are skipped).

A subtlety surfaces here that a CNN-only validator never sees:
``nn.MultiheadAttention`` in PyTorch's eval mode hits a fused
``aten::_native_multi_head_attention`` fast-path that the legacy ONNX
exporter cannot lower at opset 17 (it raises ``UnsupportedOperatorError``).
The build step disables that fast-path globally for the lifetime of the
process so the slow-path (which the exporter handles) executes instead.
This is the same workaround that the upstream torch ONNX issue tracker
recommends for ViT-style transformers; it is documented at
``docs/transformer-export.md`` in this repo.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_vit_b_16(pretrained: bool = True) -> tuple[nn.Module, torch.Tensor]:
    """Return an eval-mode ViT-B/16 plus a deterministic sample input.

    Disables the global MHA fast-path so ONNX export does not hit
    ``UnsupportedOperatorError: aten::_native_multi_head_attention``. The
    flag is process-global, but turning it off does not affect any
    already-built CNN — it only changes the kernel selected by
    ``MultiheadAttention.forward``.
    """
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    model.eval()
    generator = torch.Generator().manual_seed(42)
    sample = torch.randn(1, 3, 224, 224, generator=generator)
    return model, sample
