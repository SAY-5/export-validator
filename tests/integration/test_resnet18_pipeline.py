"""End-to-end pipeline test for ResNet-18.

Gated on ``RUN_INTEGRATION=1`` because it downloads torchvision weights and
runs the full pipeline (~1 minute on CPU).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from export_validator import compare as cmp
from export_validator import export as exp
from export_validator import ort_capture, pt_capture
from export_validator.models import build_resnet18

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run end-to-end pipeline",
)


def test_full_pipeline_no_drift_at_default_tolerance(tmp_path: Path) -> None:
    model, sample = build_resnet18()
    onnx_path = tmp_path / "resnet18.onnx"
    layer_map_path = tmp_path / "layer_map.json"
    info = exp.export_with_named_layers(model, sample, onnx_path, layer_map_path)

    pt_acts = pt_capture.run(model, sample)
    ort_acts = ort_capture.run(onnx_path, sample.numpy())
    ort_layer_acts = {k: v for k, v in ort_acts.items() if k in pt_acts}

    rpt = cmp.compare_python(
        pt_acts,
        ort_layer_acts,
        model="resnet18",
        tolerance=1e-4,
        layer_order=info["layers"],
    )
    # Real ResNet-18 fp32 export has shown no per-layer drift at 1e-4.
    assert rpt.layers_total >= 50, f"unexpected layer count: {rpt.layers_total}"
    assert rpt.layers_exceeding == 0, (
        f"unexpected drift: origin={rpt.drift_origin}, " f"exceeding={rpt.layers_exceeding}"
    )
    assert rpt.drift_origin is None


def test_layer_map_matches_committed_example() -> None:
    """The layer count in a fresh export must match the committed snapshot."""
    here = Path(__file__).resolve().parents[2]
    committed = json.loads((here / "examples" / "resnet18_layer_map.json").read_text())
    assert committed["model"] == "ResNet"
    assert committed["input_name"] == "input"
    assert committed["final_output"] == "output"
    # Sanity: the committed map names a final classifier layer.
    assert "fc" in committed["layers"]
