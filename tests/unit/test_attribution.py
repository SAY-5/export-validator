"""Unit tests for layer-mismatch root cause attribution.

The strategy is to produce an ONNX export from a tiny model, then
synthetically mutate one of three things to produce a known cause class:

- ``weight_bit_mismatch``: rewrite an initializer in the saved ONNX file so
  it no longer matches the PyTorch state_dict, then re-run attribution and
  assert the affected layer is flagged.
- ``op_implementation``: do not mutate anything; instead drive the
  PyTorch-side activation away from the ONNX-side one by tweaking the BN
  ``eps`` after export, so the BN op_type is recognised as a known
  implementation-quirk source.
- ``precision_loss``: declare the activation dtype as ``float16`` and
  assert every violating layer is attributed to precision loss.
- ``unknown``: feed a synthetic divergence on a layer whose op type is
  outside the quirk allow-list and whose weights match.

The tests use the same _Tiny model as ``test_export.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from export_validator import attribution as attr
from export_validator import compare as cmp
from export_validator import export as exp
from export_validator import ort_capture, pt_capture


class _Tiny(nn.Module):
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


def _exported(tmp_path: Path) -> tuple[nn.Module, Path, dict, torch.Tensor]:
    torch.manual_seed(0)
    model = _Tiny().eval()
    sample = torch.randn(1, 1, 4, 4)
    onnx_path = tmp_path / "tiny.onnx"
    layer_map_path = tmp_path / "tiny_map.json"
    info = exp.export_with_named_layers(model, sample, onnx_path, layer_map_path)
    return model, onnx_path, info, sample


def _mutate_initializer(onnx_path: Path, init_name: str, scale: float) -> None:
    """Multiply every element of the named initializer by ``scale`` in place."""
    onnx_model = onnx.load(str(onnx_path))
    found = False
    for init in onnx_model.graph.initializer:
        if init.name != init_name:
            continue
        arr = onnx.numpy_helper.to_array(init).copy()
        arr = arr.astype(np.float32) * np.float32(scale)
        new_init = onnx.numpy_helper.from_array(arr, name=init_name)
        init.CopyFrom(new_init)
        found = True
        break
    if not found:
        raise AssertionError(f"{init_name!r} not in initializers")
    onnx.save(onnx_model, str(onnx_path))


def _run_compare(
    model: nn.Module, onnx_path: Path, info: dict, sample: torch.Tensor, tolerance: float = 1e-4
) -> cmp.CompareReport:
    pt_acts = pt_capture.run(model, sample)
    ort_acts = ort_capture.run(onnx_path, sample.numpy())
    ort_layer_acts = {k: v for k, v in ort_acts.items() if k in pt_acts}
    return cmp.compare_python(
        pt_acts,
        ort_layer_acts,
        model="tiny",
        tolerance=tolerance,
        layer_order=info["layers"],
    )


def test_clean_export_has_no_violators_to_attribute(tmp_path: Path) -> None:
    """Sanity: a clean export produces no violations, so no causes are emitted."""
    model, onnx_path, info, sample = _exported(tmp_path)
    rpt = _run_compare(model, onnx_path, info, sample, tolerance=1e-4)
    assert rpt.layers_exceeding == 0
    causes = attr.attribute_causes(rpt, model=model, onnx_path=onnx_path)
    assert causes == []


def test_weight_mutation_is_attributed_to_weight_bit_mismatch(tmp_path: Path) -> None:
    """Rewriting a Conv weight in the ONNX file → weight_bit_mismatch on conv."""
    model, onnx_path, info, sample = _exported(tmp_path)
    _mutate_initializer(onnx_path, "model.conv.weight", scale=1.01)
    rpt = _run_compare(model, onnx_path, info, sample, tolerance=1e-4)
    assert rpt.layers_exceeding > 0, "mutation should produce drift"
    causes = attr.attribute_causes(rpt, model=model, onnx_path=onnx_path)
    by_layer = {c.layer: c for c in causes}
    # The conv layer itself must be flagged as weight_bit_mismatch.
    assert "conv" in by_layer, list(by_layer)
    assert by_layer["conv"].cause == "weight_bit_mismatch"
    assert "weight" in by_layer["conv"].detail


def test_bn_eps_mutation_is_attributed_to_op_implementation(tmp_path: Path) -> None:
    """Tweaking BN running stats *after* export: the PT side computes a
    different value from the ONNX graph (which captured the pre-mutation
    stats as initializers). The bn output's op_type is BatchNormalization,
    a known op_implementation quirk → cause should be op_implementation,
    not weight_bit_mismatch (because we mutate state_dict not the ONNX).
    """
    model, onnx_path, info, sample = _exported(tmp_path)
    # Skew BN running stats on the PyTorch side; the ONNX file already
    # captured the pre-mutation values. The PT vs ORT activations now
    # diverge on the BN layer. The Python state_dict no longer matches
    # the ONNX initializers — so weight_bit_mismatch fires first. To
    # truly isolate op_implementation we have to mutate something the
    # heuristic does *not* read (bn.eps), then synthesise a divergence.
    # Instead of trying to hide the mutation from the heuristic, we write
    # a more direct test: inject a fake CompareReport whose violating
    # layer is "bn" with no weight mismatch in the ONNX file. That tests
    # the heuristic in isolation.
    rpt_layers = [
        cmp.LayerStat(
            layer="bn", shape=[1, 2, 4, 4], max_abs_diff=1e-3, mean_abs_diff=1e-4, exceeds_tol=True
        ),
    ]
    fake_rpt = cmp.CompareReport(
        model="tiny",
        tolerance=1e-4,
        layers=rpt_layers,
        drift_origin="bn",
        layers_total=1,
        layers_exceeding=1,
    )
    causes = attr.attribute_causes(fake_rpt, model=model, onnx_path=onnx_path)
    assert len(causes) == 1
    assert causes[0].layer == "bn"
    assert causes[0].cause == "op_implementation"
    assert "BatchNormalization" in causes[0].detail


def test_precision_loss_when_activation_dtype_is_fp16(tmp_path: Path) -> None:
    """Same fake report as above, but pass activation_dtype='float16'.

    Even on a BN layer (which would be op_implementation at fp32), the
    fp16 dtype outranks op_implementation in the cause hierarchy.
    """
    model, onnx_path, _info, _sample = _exported(tmp_path)
    fake_rpt = cmp.CompareReport(
        model="tiny",
        tolerance=1e-4,
        layers=[
            cmp.LayerStat(
                layer="bn",
                shape=[1, 2, 4, 4],
                max_abs_diff=5e-3,
                mean_abs_diff=5e-4,
                exceeds_tol=True,
            )
        ],
        drift_origin="bn",
        layers_total=1,
        layers_exceeding=1,
    )
    causes = attr.attribute_causes(
        fake_rpt, model=model, onnx_path=onnx_path, activation_dtype="float16"
    )
    assert len(causes) == 1
    assert causes[0].cause == "precision_loss"
    assert "float16" in causes[0].detail


def test_unknown_when_op_type_is_unrecognised(tmp_path: Path) -> None:
    """A violator whose op is outside the quirk allow-list and whose
    weights match must be classified as 'unknown'."""
    model, onnx_path, _info, _sample = _exported(tmp_path)
    # The "relu" leaf in our Tiny model maps to a Relu ONNX node, which
    # is not in _OP_IMPL_QUIRKS. Weights for ReLU don't exist (no
    # parameters), so weight_bit_mismatch cannot fire either.
    fake_rpt = cmp.CompareReport(
        model="tiny",
        tolerance=1e-4,
        layers=[
            cmp.LayerStat(
                layer="relu",
                shape=[1, 2, 4, 4],
                max_abs_diff=1e-3,
                mean_abs_diff=1e-4,
                exceeds_tol=True,
            )
        ],
        drift_origin="relu",
        layers_total=1,
        layers_exceeding=1,
    )
    causes = attr.attribute_causes(fake_rpt, model=model, onnx_path=onnx_path)
    assert len(causes) == 1
    assert causes[0].cause == "unknown"


def test_non_violating_layers_are_not_attributed(tmp_path: Path) -> None:
    """Layers with exceeds_tol=False must not appear in the cause output."""
    model, onnx_path, _info, _sample = _exported(tmp_path)
    fake_rpt = cmp.CompareReport(
        model="tiny",
        tolerance=1e-4,
        layers=[
            cmp.LayerStat(
                layer="bn",
                shape=[1, 2, 4, 4],
                max_abs_diff=1e-6,
                mean_abs_diff=1e-7,
                exceeds_tol=False,
            ),
            cmp.LayerStat(
                layer="conv",
                shape=[1, 2, 4, 4],
                max_abs_diff=1.0,
                mean_abs_diff=0.5,
                exceeds_tol=True,
            ),
        ],
        drift_origin="conv",
        layers_total=2,
        layers_exceeding=1,
    )
    causes = attr.attribute_causes(fake_rpt, model=model, onnx_path=onnx_path)
    assert [c.layer for c in causes] == ["conv"]


def test_report_writes_cause_sidecar_when_attribution_present(tmp_path: Path) -> None:
    """``write_outputs`` must produce ``<base>_causes.json`` only when given causes."""
    from export_validator.report import write_outputs

    model, onnx_path, _info, _sample = _exported(tmp_path)
    fake_rpt = cmp.CompareReport(
        model="tiny",
        tolerance=1e-4,
        layers=[
            cmp.LayerStat(
                layer="bn",
                shape=[1, 2, 4, 4],
                max_abs_diff=1e-3,
                mean_abs_diff=1e-4,
                exceeds_tol=True,
            ),
        ],
        drift_origin="bn",
        layers_total=1,
        layers_exceeding=1,
    )
    base = tmp_path / "out"
    causes = attr.attribute_causes(fake_rpt, model=model, onnx_path=onnx_path)
    write_outputs(fake_rpt, base, causes)
    sidecar = tmp_path / "out_causes.json"
    assert sidecar.exists()
    body = sidecar.read_text()
    assert "op_implementation" in body
    assert "bn" in body
    # Primary JSON must NOT mention causes (Python ↔ C++ byte-equal contract).
    primary = (tmp_path / "out.json").read_text()
    assert "cause" not in primary


def test_report_omits_cause_sidecar_when_attribution_disabled(tmp_path: Path) -> None:
    from export_validator.report import write_outputs

    fake_rpt = cmp.CompareReport(
        model="tiny",
        tolerance=1e-4,
        layers=[],
        drift_origin=None,
        layers_total=0,
        layers_exceeding=0,
    )
    base = tmp_path / "out2"
    write_outputs(fake_rpt, base)
    assert not (tmp_path / "out2_causes.json").exists()
