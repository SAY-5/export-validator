"""Forward-hook instrumentation and named-output ONNX export wrapper.

The strategy:
- ``NamingHooks`` walks ``model.named_modules()``, picks leaf modules (no children),
  and attaches a forward hook on each that records the output tensor under the
  module's dotted path name.
- ``NamedOutputWrapper`` wraps the model so that ``forward`` returns
  ``(final_output, *intermediate_outputs)`` in a deterministic order. This lets
  ``torch.onnx.export`` emit named graph outputs we can later read from
  onnxruntime ``session.run`` and align with the PyTorch capture.

Only floating-point tensor outputs are tracked. Modules whose output is a
non-tensor (e.g. tuple-returning custom blocks) are skipped from the named
output list but the hook still records them for the Python capture path.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn


def _is_leaf(module: nn.Module) -> bool:
    """A leaf module has no submodules of its own."""
    return next(module.children(), None) is None


def enumerate_leaves(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return ``[(dotted_name, module), ...]`` for every leaf module.

    The empty-string root is excluded. Order is deterministic (depth-first as
    yielded by ``named_modules``).
    """
    leaves: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if _is_leaf(module):
            leaves.append((name, module))
    return leaves


@dataclass
class NamingHooks:
    """Attach forward hooks to every leaf module and capture outputs by name."""

    model: nn.Module
    captures: dict[str, Tensor] = field(default_factory=dict)
    _handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def __enter__(self) -> NamingHooks:
        self.attach()
        return self

    def __exit__(self, *exc: object) -> None:
        self.detach()

    def attach(self) -> None:
        self.detach()
        self.captures.clear()
        for name, module in enumerate_leaves(self.model):
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str) -> Any:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, Tensor):
                self.captures[name] = output.detach().clone()

        return hook


class NamedOutputWrapper(nn.Module):
    """Wrap ``model`` so ``forward`` returns ``(final, *named_intermediates)``.

    The intermediate outputs are gathered via in-graph forward hooks that stash
    ``(name, tensor)`` pairs into a list, in the order modules actually
    execute. The wrapper's ``forward`` discards the names at runtime (only the
    tuple of tensors can flow through ONNX) but records the execution-order
    name list in ``self.execution_order`` after the first forward pass.
    ``output_names`` then reflects that order, which is what
    ``torch.onnx.export`` will use to label graph outputs.
    """

    def __init__(self, model: nn.Module, layer_names: Iterable[str]) -> None:
        super().__init__()
        self.model = model
        self.layer_names: list[str] = list(layer_names)
        self._buffer: list[tuple[str, Tensor]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.execution_order: list[str] = []
        leaves = dict(enumerate_leaves(model))
        wanted = set(self.layer_names)
        # Hook every wanted leaf with a name-aware closure.
        for name, module in enumerate_leaves(model):
            if name not in wanted:
                continue
            if name not in leaves:
                raise KeyError(f"layer {name!r} not found among leaves")
            self._handles.append(module.register_forward_hook(self._make_stash(name)))

    def _make_stash(self, name: str) -> Any:
        def stash(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, Tensor):
                # Clone so a downstream in-place op (e.g. nn.ReLU(inplace=True))
                # cannot retroactively alter what we expose as a named ONNX
                # output. Without this, BN outputs on ResNet show post-ReLU
                # values because in-place ReLU mutates the same storage.
                self._buffer.append((name, output.clone()))

        return stash

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        self._buffer.clear()
        final = self.model(x)
        # Record execution order so the caller can align it with ONNX outputs.
        self.execution_order = [name for name, _ in self._buffer]
        return (final, *(tensor for _, tensor in self._buffer))

    def output_names(self) -> list[str]:
        if self.execution_order:
            return ["output", *self.execution_order]
        return ["output", *self.layer_names]


def select_exportable_layers(model: nn.Module, sample_input: Tensor) -> list[str]:
    """Run a dry pass to discover which leaves emit a single Tensor output.

    Some layers (e.g. modules that return tuples) cannot be cleanly named in
    the ONNX graph with this strategy; those are filtered out.
    """
    leaves = enumerate_leaves(model)
    saw_tensor: dict[str, bool] = {name: False for name, _ in leaves}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_probe(name: str) -> Any:
        def probe(_m: nn.Module, _i: tuple[Any, ...], output: Any) -> None:
            saw_tensor[name] = isinstance(output, Tensor)

        return probe

    for name, module in leaves:
        handles.append(module.register_forward_hook(make_probe(name)))
    try:
        with torch.no_grad():
            model.eval()
            model(sample_input)
    finally:
        for h in handles:
            h.remove()
    return [name for name, _ in leaves if saw_tensor[name]]
