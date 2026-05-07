"""Python <-> C++ comparator parity test.

Skipped when the C++ binary has not been built. Constructs synthetic
activation tensors (so no network download is needed), runs both backends,
and asserts byte-identical JSON output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from export_validator import compare as cmp
from export_validator.report import write_outputs

pytestmark = pytest.mark.skipif(
    cmp.native_binary() is None,
    reason="C++ comparator binary not built (run 'make build-cpp')",
)


def _fixtures(seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Generate three synthetic 'layers' with varying drift."""
    rng = np.random.default_rng(seed)
    pt = {
        "conv1": rng.standard_normal((1, 4, 8, 8)).astype(np.float32),
        "bn1": rng.standard_normal((1, 4, 8, 8)).astype(np.float32),
        "fc": rng.standard_normal((1, 10)).astype(np.float32),
    }
    ort = {
        "conv1": pt["conv1"] + rng.standard_normal(pt["conv1"].shape).astype(np.float32) * 1e-7,
        "bn1": pt["bn1"] + rng.standard_normal(pt["bn1"].shape).astype(np.float32) * 1e-3,
        "fc": pt["fc"].copy(),
    }
    return pt, ort, ["conv1", "bn1", "fc"]


def test_native_and_python_emit_byte_identical_json(tmp_path: Path) -> None:
    pt, ort, order = _fixtures()
    py_rpt = cmp.compare_python(
        pt,
        ort,
        model="synthetic",
        tolerance=1e-4,
        layer_order=order,
    )
    py_json, _ = write_outputs(py_rpt, tmp_path / "py")

    pt_path = tmp_path / "pt.evl1"
    ort_path = tmp_path / "ort.evl1"
    cmp.save_evl1(pt_path, pt)
    cmp.save_evl1(ort_path, ort)
    cpp_rpt = cmp.compare_native(
        pt_path,
        ort_path,
        model="synthetic",
        tolerance=1e-4,
        layer_order=order,
    )
    cpp_json, _ = write_outputs(cpp_rpt, tmp_path / "cpp")

    py_bytes = py_json.read_bytes()
    cpp_bytes = cpp_json.read_bytes()
    assert py_bytes == cpp_bytes, (
        "Python and C++ comparator JSON output diverged.\n"
        f"py:  {py_bytes[:200]!r}...\n"
        f"cpp: {cpp_bytes[:200]!r}..."
    )

    payload = json.loads(py_bytes)
    assert payload["drift_origin"] == "bn1"
    assert payload["layers_exceeding"] == 1
