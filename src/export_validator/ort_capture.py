"""Capture per-layer activations from an ONNX Runtime session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort


def run(onnx_path: Path, inputs: npt.NDArray[np.float32]) -> dict[str, npt.NDArray[np.float32]]:
    """Run ``inputs`` through the ONNX model at ``onnx_path``.

    Returns a dict mapping every named graph output to its ndarray. The final
    classifier output is included under the key ``"output"``.
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [o.name for o in session.get_outputs()]
    input_name = session.get_inputs()[0].name
    feeds = {input_name: inputs.astype(np.float32, copy=False)}
    results = session.run(output_names, feeds)
    return dict(zip(output_names, results, strict=True))
