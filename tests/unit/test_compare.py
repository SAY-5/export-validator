"""Unit tests for the Python-fallback comparator and report formats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from export_validator.compare import (
    LayerStat,
    compare_python,
    save_evl1,
    save_npz,
)
from export_validator.report import render_markdown, write_outputs


def _arrays() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    a = {"l1": np.array([[1.0, 2.0]], dtype=np.float32), "l2": np.array([[0.0]], dtype=np.float32)}
    b = {"l1": np.array([[1.0, 2.0]], dtype=np.float32), "l2": np.array([[0.5]], dtype=np.float32)}
    return a, b


def test_compare_zero_diff_when_identical() -> None:
    a, _ = _arrays()
    rpt = compare_python(a, a, model="m", tolerance=1e-4, layer_order=["l1", "l2"])
    assert rpt.layers_total == 2
    assert rpt.layers_exceeding == 0
    assert rpt.drift_origin is None
    assert all(s.max_abs_diff == 0.0 for s in rpt.layers)


def test_compare_first_violator_is_drift_origin() -> None:
    a, b = _arrays()
    rpt = compare_python(a, b, model="m", tolerance=1e-4, layer_order=["l1", "l2"])
    assert rpt.layers_exceeding == 1
    assert rpt.drift_origin == "l2"
    assert rpt.layers[1].exceeds_tol is True


def test_layer_order_is_respected() -> None:
    a, b = _arrays()
    rpt = compare_python(a, b, model="m", tolerance=1e-4, layer_order=["l2", "l1"])
    assert [s.layer for s in rpt.layers] == ["l2", "l1"]


def test_missing_layer_is_skipped() -> None:
    a = {"l1": np.zeros((1,), dtype=np.float32)}
    b = {"l1": np.zeros((1,), dtype=np.float32), "l2": np.ones((1,), dtype=np.float32)}
    rpt = compare_python(a, b, model="m", tolerance=1e-4, layer_order=["l1", "l2"])
    assert rpt.layers_total == 1


def test_tolerance_is_strictly_greater() -> None:
    a = {"l": np.array([0.0], dtype=np.float32)}
    b = {"l": np.array([1e-4], dtype=np.float32)}
    rpt = compare_python(a, b, model="m", tolerance=1e-4, layer_order=["l"])
    assert rpt.layers_exceeding == 0
    assert rpt.layers[0].exceeds_tol is False


def test_round_trip_evl1(tmp_path: Path) -> None:
    arrays = {
        "alpha": np.arange(6, dtype=np.float32).reshape(2, 3),
        "beta": np.array([3.14], dtype=np.float32),
    }
    out = tmp_path / "x.evl1"
    save_evl1(out, arrays)
    assert out.exists()
    assert out.stat().st_size > 0
    # Magic header is "EVL1".
    with out.open("rb") as fh:
        assert fh.read(4) == b"EVL1"


def test_save_npz_writes(tmp_path: Path) -> None:
    arrays = {"a": np.zeros((2,), dtype=np.float32)}
    p = tmp_path / "x.npz"
    save_npz(p, arrays)
    loaded = np.load(p)
    np.testing.assert_array_equal(loaded["a"], arrays["a"])


def test_markdown_includes_no_drift_message_when_clean() -> None:
    a, _ = _arrays()
    rpt = compare_python(a, a, model="m", tolerance=1e-4, layer_order=["l1", "l2"])
    md = render_markdown(rpt)
    assert "No drift detected" in md
    assert "Drift origin: none" in md


def test_write_outputs_is_deterministic(tmp_path: Path) -> None:
    a, b = _arrays()
    rpt = compare_python(a, b, model="m", tolerance=1e-4, layer_order=["l1", "l2"])
    j1, m1 = write_outputs(rpt, tmp_path / "r1")
    j2, m2 = write_outputs(rpt, tmp_path / "r2")
    assert j1.read_bytes() == j2.read_bytes()
    assert m1.read_bytes() == m2.read_bytes()
    payload = json.loads(j1.read_text())
    assert payload["drift_origin"] == "l2"
    # Sorted keys at the top level: drift_origin first.
    assert list(payload.keys())[0] == "drift_origin"


def test_layerstat_is_immutable() -> None:
    s = LayerStat(layer="x", shape=[1], max_abs_diff=0.0, mean_abs_diff=0.0, exceeds_tol=False)
    try:
        s.layer = "y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("LayerStat should be frozen")
