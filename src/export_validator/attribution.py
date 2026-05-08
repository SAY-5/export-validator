"""Root-cause attribution for per-layer divergence.

The compare layer reports *where* the first divergence appears
(``drift_origin``) and the magnitude of every layer's drift. Attribution
classifies *why* each violating layer drifts, using only artefacts the
exporter already produces (the PyTorch state_dict, the ONNX ModelProto, the
declared activation dtype).

Causes are mutually exclusive and assigned in the order below; the first
match wins. A layer that does not exceed ``tolerance`` is never assigned a
cause.

``weight_bit_mismatch``
    The PyTorch parameter named ``layer.<param>`` is not bit-identical to
    the ONNX graph initializer it should correspond to (typically
    ``model.<layer>.<param>``). Rare on the legacy TorchScript exporter,
    but possible with the dynamo path or with constant-folding turned on.
    A bit-mismatch on a weight upstream of the violating layer is enough
    to flag it: the output activation cannot be byte-exact if the
    exported weights are not.

``precision_loss``
    The activation dtype is not fp32 (today only fp16 is recognised).
    Drift in fp16 is dominated by representation, not by op
    implementation; up to ~1e-3 absolute drift is expected and reporting
    it as ``op_implementation`` would mislead the reader.

``op_implementation``
    Same op type, different numerical implementation between PyTorch and
    ONNX Runtime. The classic offenders are ``BatchNormalization``
    (different epsilon-handling, in-place fusion with following ReLU) and
    ``LayerNorm`` (mean/variance accumulation order). The classifier
    checks the ONNX ``op_type`` for the node whose output matches the
    layer name and consults a static allow-list.

``unknown``
    Fallback for layers that exceed tolerance but match none of the
    heuristics above. Real-model runs at fp32 should never hit this; if
    they do, the heuristic table needs an entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnx
from torch import nn

from .compare import CompareReport

Cause = Literal[
    "weight_bit_mismatch",
    "precision_loss",
    "op_implementation",
    "unknown",
]

# Op types whose ONNX ↔ PyTorch numerical implementation is known to differ
# enough to produce >1e-4 drift on common-sized activations. The list is
# deliberately conservative — only ops where we have first-hand evidence
# (literature, debugging notes, or the existing drift-debugging.md doc in
# this repo) are listed. Unrecognised ops fall through to ``unknown``.
_OP_IMPL_QUIRKS: frozenset[str] = frozenset(
    {
        "BatchNormalization",
        "LayerNormalization",
        # InstanceNormalization shares the same epsilon-handling quirk.
        "InstanceNormalization",
        # Softmax over large axes accumulates differently on the ORT side.
        "Softmax",
    }
)


@dataclass(frozen=True)
class CauseAssignment:
    """Per-layer attribution result."""

    layer: str
    cause: Cause
    detail: str


def _state_dict_to_numpy(model: nn.Module) -> dict[str, npt.NDArray[np.generic]]:
    out: dict[str, npt.NDArray[np.generic]] = {}
    for name, tensor in model.state_dict().items():
        out[name] = tensor.detach().cpu().numpy()
    return out


def _onnx_initializers(graph: onnx.GraphProto) -> dict[str, npt.NDArray[np.generic]]:
    out: dict[str, npt.NDArray[np.generic]] = {}
    for init in graph.initializer:
        out[init.name] = onnx.numpy_helper.to_array(init)
    return out


def _node_output_to_op_type(graph: onnx.GraphProto) -> dict[str, str]:
    """Map ONNX node *output* names to their producing op_type."""
    out: dict[str, str] = {}
    for node in graph.node:
        for output_name in node.output:
            out[output_name] = node.op_type
    return out


def _initializers_for_layer(
    layer: str,
    sd: dict[str, npt.NDArray[np.generic]],
    inits: dict[str, npt.NDArray[np.generic]],
) -> list[tuple[str, npt.NDArray[np.generic], npt.NDArray[np.generic]]]:
    """Return ``[(suffix, pt_array, onnx_array), ...]`` for every parameter
    that exists for ``layer`` in both the state_dict and the ONNX initializers.

    The exporter prefixes every state_dict key with ``model.`` (because the
    ``NamedOutputWrapper.model`` attribute holds the original model), so
    ``conv1.weight`` in PyTorch becomes ``model.conv1.weight`` in ONNX.
    """
    matches: list[tuple[str, npt.NDArray[np.generic], npt.NDArray[np.generic]]] = []
    suffixes = ("weight", "bias", "running_mean", "running_var")
    for suffix in suffixes:
        pt_key = f"{layer}.{suffix}"
        onnx_key = f"model.{layer}.{suffix}"
        if pt_key in sd and onnx_key in inits:
            matches.append((suffix, sd[pt_key], inits[onnx_key]))
    return matches


def _classify_layer(
    layer: str,
    *,
    sd: dict[str, npt.NDArray[np.generic]],
    inits: dict[str, npt.NDArray[np.generic]],
    op_type: str | None,
    activation_dtype: str,
) -> CauseAssignment:
    # 1. weight_bit_mismatch
    for suffix, pt_arr, onnx_arr in _initializers_for_layer(layer, sd, inits):
        if pt_arr.shape != onnx_arr.shape:
            return CauseAssignment(
                layer,
                "weight_bit_mismatch",
                f"{suffix}: shape {pt_arr.shape} vs {onnx_arr.shape}",
            )
        # Same dtype required for bit-identical comparison; if the exporter
        # cast a float param, that *is* the mismatch signature.
        if pt_arr.dtype != onnx_arr.dtype:
            return CauseAssignment(
                layer,
                "weight_bit_mismatch",
                f"{suffix}: dtype {pt_arr.dtype} vs {onnx_arr.dtype}",
            )
        if pt_arr.tobytes() != onnx_arr.tobytes():
            max_abs = float(np.max(np.abs(pt_arr.astype(np.float64) - onnx_arr.astype(np.float64))))
            return CauseAssignment(
                layer,
                "weight_bit_mismatch",
                f"{suffix}: max_abs={max_abs:.3e}",
            )

    # 2. precision_loss
    if activation_dtype != "float32":
        return CauseAssignment(
            layer,
            "precision_loss",
            f"activation dtype = {activation_dtype}",
        )

    # 3. op_implementation
    if op_type in _OP_IMPL_QUIRKS:
        return CauseAssignment(
            layer,
            "op_implementation",
            f"op_type = {op_type}",
        )

    # 4. unknown
    return CauseAssignment(
        layer,
        "unknown",
        f"op_type = {op_type or 'unmapped'}",
    )


def attribute_causes(
    report: CompareReport,
    *,
    model: nn.Module,
    onnx_path: Path,
    activation_dtype: str = "float32",
) -> list[CauseAssignment]:
    """Classify the cause of every violating layer in ``report``.

    Returns one :class:`CauseAssignment` per layer that exceeded the
    tolerance, in the same order they appear in the report. Non-violating
    layers are not included; the caller can render them with ``cause=None``
    if needed.
    """
    onnx_model = onnx.load(str(onnx_path))
    sd = _state_dict_to_numpy(model)
    inits = _onnx_initializers(onnx_model.graph)
    op_types = _node_output_to_op_type(onnx_model.graph)
    out: list[CauseAssignment] = []
    for stat in report.layers:
        if not stat.exceeds_tol:
            continue
        op_type = op_types.get(stat.layer)
        out.append(
            _classify_layer(
                stat.layer,
                sd=sd,
                inits=inits,
                op_type=op_type,
                activation_dtype=activation_dtype,
            )
        )
    return out
