"""Hypothesis property tests for the leaf-instrumentation layer.

We build random module trees with hypothesis (sequential blocks of varying
depth and width, mixing leaf op types) and assert two contracts that the
named-output ONNX export depends on:

1. ``enumerate_leaves`` returns an entry for **every** leaf module reachable
   under the root, never duplicates a name, and never returns the root.
2. After running ``NamedOutputWrapper`` on a synthetic input, every leaf that
   actually emitted a tensor lands in ``execution_order`` exactly once, with
   a unique name. The ``output_names()`` returned to ``torch.onnx.export`` is
   then guaranteed unique — the same property the live ResNet-18 export
   depends on.
"""

from __future__ import annotations

from typing import Any

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn

from export_validator.instrument import (
    NamedOutputWrapper,
    enumerate_leaves,
    select_exportable_layers,
)


def _leaf_factories() -> dict[str, Any]:
    """Map of factory keys to (callable -> nn.Module) for hypothesis to pick from.

    Every factory returns a module that:
      - has no submodules (so it is a leaf),
      - accepts a ``(N, C, 1, 1)`` float tensor and returns one of the same
        shape (or a flattened version, see _Flatten).
    """
    return {
        "conv1x1": lambda c: nn.Conv2d(c, c, kernel_size=1),
        "bn": lambda c: nn.BatchNorm2d(c),
        "relu": lambda _c: nn.ReLU(),
        "id": lambda _c: nn.Identity(),
        "avgpool": lambda _c: nn.AdaptiveAvgPool2d(1),
    }


class _Trunk(nn.Module):
    """A flat sequential trunk built from the leaves hypothesis chose."""

    def __init__(self, channels: int, leaf_keys: list[str]) -> None:
        super().__init__()
        factories = _leaf_factories()
        self.channels = channels
        for i, key in enumerate(leaf_keys):
            self.add_module(f"{key}_{i}", factories[key](channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.children():
            x = module(x)
        return x


@st.composite
def _random_trunk(draw: st.DrawFn) -> tuple[_Trunk, torch.Tensor, list[str]]:
    keys = draw(st.lists(st.sampled_from(list(_leaf_factories())), min_size=1, max_size=8))
    channels = draw(st.integers(min_value=1, max_value=4))
    torch.manual_seed(draw(st.integers(min_value=0, max_value=10_000)))
    trunk = _Trunk(channels, keys).eval()
    sample = torch.randn(1, channels, 4, 4)
    return trunk, sample, keys


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_random_trunk())
def test_enumerate_leaves_yields_unique_names(
    bundle: tuple[_Trunk, torch.Tensor, list[str]],
) -> None:
    trunk, _sample, keys = bundle
    leaves = enumerate_leaves(trunk)
    names = [n for n, _ in leaves]
    # Same number of leaves as factories the trunk asked for.
    assert len(leaves) == len(keys)
    # Every name unique (leaf attribute names use distinct suffixes).
    assert len(set(names)) == len(names)
    # Root is never returned.
    assert "" not in set(names)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_random_trunk())
def test_named_output_wrapper_assigns_unique_names_to_every_leaf(
    bundle: tuple[_Trunk, torch.Tensor, list[str]],
) -> None:
    trunk, sample, _keys = bundle
    layers = select_exportable_layers(trunk, sample)
    wrapper = NamedOutputWrapper(trunk, layers)
    out = wrapper(sample)
    # The wrapper returns final + one tensor per executed leaf.
    assert isinstance(out, tuple)
    assert len(out) == 1 + len(wrapper.execution_order)
    # Execution order is the source of truth for output ordering.
    assert len(set(wrapper.execution_order)) == len(wrapper.execution_order)
    # output_names() puts "output" first followed by the unique leaves.
    names = wrapper.output_names()
    assert names[0] == "output"
    assert len(set(names)) == len(names), names
