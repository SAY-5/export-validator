"""Smoke tests for the ONNX exporter on a tiny model.

The full ResNet-18 path is exercised by the integration suite, gated on
``RUN_INTEGRATION=1``. Here we use a small model so the unit suite stays
fast and does not need to download torchvision weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from export_validator.export import export_with_named_layers
from export_validator.pt_capture import run as pt_run


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def test_export_emits_named_outputs(tmp_path: Path) -> None:
    torch.manual_seed(0)
    m = _Tiny().eval()
    x = torch.randn(1, 1, 4, 4)
    onnx_path = tmp_path / "tiny.onnx"
    layer_map_path = tmp_path / "tiny_map.json"
    info = export_with_named_layers(m, x, onnx_path, layer_map_path, opset_version=17)
    assert onnx_path.exists()
    assert layer_map_path.exists()
    payload = json.loads(layer_map_path.read_text())
    assert payload["input_name"] == "input"
    assert payload["final_output"] == "output"
    assert "layers" in payload and len(payload["layers"]) == 5
    assert info["all_outputs"][0] == "output"


def test_exported_layer_names_match_pytorch_capture(tmp_path: Path) -> None:
    torch.manual_seed(0)
    m = _Tiny().eval()
    x = torch.randn(1, 1, 4, 4)
    onnx_path = tmp_path / "tiny.onnx"
    layer_map_path = tmp_path / "tiny_map.json"
    info = export_with_named_layers(m, x, onnx_path, layer_map_path)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_names = {o.name for o in sess.get_outputs()}
    pt_caps = pt_run(m, x)

    # Every layer recorded by the exporter must have a matching ONNX named
    # output and a matching PyTorch hook capture (by exact name).
    for name in info["layers"]:
        assert name in ort_names, f"{name} missing from ORT outputs"
        assert name in pt_caps, f"{name} missing from PyTorch captures"


def test_exported_outputs_match_within_tolerance(tmp_path: Path) -> None:
    torch.manual_seed(0)
    m = _Tiny().eval()
    x = torch.randn(1, 1, 4, 4)
    onnx_path = tmp_path / "tiny.onnx"
    layer_map_path = tmp_path / "tiny_map.json"
    info = export_with_named_layers(m, x, onnx_path, layer_map_path)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outs = sess.run(None, {"input": x.numpy()})
    pt_caps = pt_run(m, x)

    by_name = dict(zip([o.name for o in sess.get_outputs()], ort_outs, strict=True))
    for name in info["layers"]:
        diff = float(
            np.max(np.abs(by_name[name].astype(np.float64) - pt_caps[name].astype(np.float64)))
        )
        assert diff < 1e-4, f"{name} drift {diff:g}"
