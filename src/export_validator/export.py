"""ONNX export with per-layer named outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from .instrument import NamedOutputWrapper, select_exportable_layers


def export_with_named_layers(
    model: nn.Module,
    sample_input: torch.Tensor,
    onnx_path: Path,
    layer_map_path: Path,
    opset_version: int = 17,
) -> dict[str, Any]:
    """Export ``model`` to ONNX with one named output per leaf layer.

    Returns a dict with ``layer_names`` and ``output_names``. Also writes a
    JSON layer map alongside the ``.onnx`` file so consumers (Python, C++) can
    line up tensors by name.
    """
    model.eval()
    layer_names = select_exportable_layers(model, sample_input)
    wrapper = NamedOutputWrapper(model, layer_names)
    # One dry pass to populate execution_order (which may differ from
    # module-iteration order when blocks reuse activations across branches).
    with torch.no_grad():
        wrapper(sample_input)
    output_names = wrapper.output_names()
    layer_names = wrapper.execution_order

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    layer_map_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Pin to the legacy TorchScript-based exporter. The dynamo-based
        # exporter introduced in torch 2.5 does not preserve the
        # tuple-of-tensors signature we rely on for per-layer named outputs,
        # and additionally pulls in onnxscript as a hard dependency.
        export_kwargs: dict[str, Any] = {
            "input_names": ["input"],
            "output_names": output_names,
            "opset_version": opset_version,
            "dynamic_axes": {"input": {0: "batch"}},
            "do_constant_folding": False,
        }
        if "dynamo" in torch.onnx.export.__code__.co_varnames:
            export_kwargs["dynamo"] = False
        torch.onnx.export(wrapper, (sample_input,), str(onnx_path), **export_kwargs)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    layer_map = {
        "model": model.__class__.__name__,
        "opset_version": opset_version,
        "input_name": "input",
        "final_output": "output",
        "layers": list(layer_names),
        "all_outputs": list(output_names),
    }
    layer_map_path.write_text(json.dumps(layer_map, indent=2) + "\n")
    return layer_map
