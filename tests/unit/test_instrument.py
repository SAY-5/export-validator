"""Unit tests for the hook-based instrumentation."""

from __future__ import annotations

import torch
from torch import nn

from export_validator.instrument import (
    NamedOutputWrapper,
    NamingHooks,
    enumerate_leaves,
    select_exportable_layers,
)


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _net() -> tuple[_TinyNet, torch.Tensor]:
    torch.manual_seed(0)
    net = _TinyNet().eval()
    x = torch.randn(1, 1, 4, 4)
    return net, x


def test_enumerate_leaves_returns_only_leaf_modules() -> None:
    net, _ = _net()
    leaves = enumerate_leaves(net)
    names = [n for n, _ in leaves]
    assert names == ["conv", "bn", "relu", "pool", "fc"]


def test_naming_hooks_capture_one_tensor_per_leaf() -> None:
    net, x = _net()
    with NamingHooks(net) as hooks:
        net(x)
    assert set(hooks.captures) == {"conv", "bn", "relu", "pool", "fc"}
    for name, t in hooks.captures.items():
        assert isinstance(t, torch.Tensor), name


def test_select_exportable_layers_matches_leaves_for_tensor_outputs() -> None:
    net, x = _net()
    layers = select_exportable_layers(net, x)
    assert layers == ["conv", "bn", "relu", "pool", "fc"]


def test_named_output_wrapper_returns_final_plus_intermediates() -> None:
    net, x = _net()
    layers = select_exportable_layers(net, x)
    wrapper = NamedOutputWrapper(net, layers)
    out = wrapper(x)
    assert isinstance(out, tuple)
    # final + 5 leaves
    assert len(out) == 6
    assert wrapper.execution_order == ["conv", "bn", "relu", "pool", "fc"]
    assert wrapper.output_names() == ["output", "conv", "bn", "relu", "pool", "fc"]


def test_named_output_wrapper_clones_to_survive_inplace_relu() -> None:
    """In-place ReLU after BN must not retroactively mutate the buffered tensor.

    We rig a tiny BN whose running stats force the output to land partly
    negative, then verify the wrapper's buffered BN tensor still contains those
    negatives after the in-place ReLU has executed. Cloning in the stash hook
    is what keeps PyTorch and ONNX Runtime captures aligned on real models.
    """

    class WithInplace(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            bn = nn.BatchNorm2d(1)
            # Force eval-mode BN to emit values centred near zero so the
            # in-place ReLU clearly zeros half of them.
            bn.eval()
            with torch.no_grad():
                bn.running_mean.fill_(0.0)
                bn.running_var.fill_(1.0)
                bn.weight.fill_(1.0)
                bn.bias.fill_(0.0)
            self.bn = bn
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.bn(x))

    m = WithInplace().eval()
    # Mix of negative and positive inputs so the BN output (with identity
    # affine) is also mixed sign.
    x = torch.tensor([[[[-1.5, -0.5], [0.5, 1.5]]]])
    layers = select_exportable_layers(m, x)
    wrapper = NamedOutputWrapper(m, layers)
    out = wrapper(x)
    bn_buffered = out[1 + layers.index("bn")]
    # The buffered BN output must include negative values. Without the clone,
    # in-place ReLU would have zeroed them.
    assert (bn_buffered < 0).any().item()
