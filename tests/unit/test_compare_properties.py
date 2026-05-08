"""Hypothesis property tests for the Python differ.

Generates random tensor shapes, payloads, and tolerances and asserts that:

- ``compare_python`` returns the same per-layer ``max_abs_diff`` and
  ``mean_abs_diff`` as a direct ``numpy`` reference (modulo the documented
  %.12e rounding).
- The ``exceeds_tol`` classification matches the textual contract
  (``max_abs_diff > tolerance``, strictly greater).
- ``drift_origin`` is the *first* violator in execution order, never the
  smallest-shape one.
- ``layers_total`` and ``layers_exceeding`` are consistent with the per-layer
  flags.

These tests do not exercise the ONNX or PyTorch path; they pin the
mathematical contract of the comparator so any future refactor that touches
the rounding policy or accumulation order shows up immediately.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from export_validator.compare import _round, compare_python

# Keep arrays small: hypothesis spends most of its time shrinking, and the
# differ is O(n) so any n is exercised; large n only slows the suite.
_DTYPE = np.float32
_MAX_DIM = 4
_MAX_NDIM = 3


def _shapes() -> st.SearchStrategy[tuple[int, ...]]:
    return hnp.array_shapes(min_dims=1, max_dims=_MAX_NDIM, min_side=1, max_side=_MAX_DIM)


def _arrays(shape: tuple[int, ...]) -> st.SearchStrategy[np.ndarray]:
    # Bound element magnitude so float32 cancellation is well-defined and
    # hypothesis does not spend cycles on subnormals/NaNs.
    return hnp.arrays(
        dtype=_DTYPE,
        shape=shape,
        elements=st.floats(
            min_value=-1e3,
            max_value=1e3,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
    )


def _layer_payloads() -> st.SearchStrategy[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(name, pt_array, ort_array)`` triples with the same shape."""
    return _shapes().flatmap(
        lambda shape: st.tuples(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="._"
                ),
                min_size=1,
                max_size=12,
            ),
            _arrays(shape),
            _arrays(shape),
        )
    )


@st.composite
def _layer_set(draw: st.DrawFn) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Draw a small set of layers with unique names and consistent shapes."""
    n = draw(st.integers(min_value=1, max_value=5))
    seen_names: set[str] = set()
    pt: dict[str, np.ndarray] = {}
    ort: dict[str, np.ndarray] = {}
    order: list[str] = []
    for _ in range(n):
        name, a, b = draw(_layer_payloads())
        # Hypothesis can re-draw colliding names; force uniqueness deterministically.
        original = name
        bump = 0
        while name in seen_names:
            bump += 1
            name = f"{original}_{bump}"
        seen_names.add(name)
        pt[name] = a
        ort[name] = b
        order.append(name)
    return pt, ort, order


def _numpy_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    if diff.size == 0:
        return 0.0, 0.0
    return float(diff.max()), float(diff.mean())


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(layers=_layer_set(), tol_log=st.floats(min_value=-9.0, max_value=2.0))
def test_max_and_mean_match_numpy_reference(
    layers: tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]],
    tol_log: float,
) -> None:
    pt, ort, order = layers
    tolerance = 10**tol_log
    rpt = compare_python(pt, ort, model="m", tolerance=tolerance, layer_order=order)
    assert rpt.layers_total == len(order)
    for stat in rpt.layers:
        ref_max, ref_mean = _numpy_stats(pt[stat.layer], ort[stat.layer])
        # The compare layer rounds with %.12e; reproduce that rounding here so
        # the equality is exact rather than approximate.
        assert stat.max_abs_diff == _round(ref_max), (
            stat.layer,
            stat.max_abs_diff,
            _round(ref_max),
        )
        assert stat.mean_abs_diff == _round(ref_mean), (
            stat.layer,
            stat.mean_abs_diff,
            _round(ref_mean),
        )


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(layers=_layer_set(), tol_log=st.floats(min_value=-9.0, max_value=2.0))
def test_exceeds_tol_is_strictly_greater(
    layers: tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]],
    tol_log: float,
) -> None:
    pt, ort, order = layers
    tolerance = 10**tol_log
    rpt = compare_python(pt, ort, model="m", tolerance=tolerance, layer_order=order)
    for stat in rpt.layers:
        # exceeds = (rounded max_abs_diff) > tolerance, strictly.
        assert stat.exceeds_tol == bool(stat.max_abs_diff > tolerance), (
            stat.layer,
            stat.max_abs_diff,
            tolerance,
        )


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(layers=_layer_set(), tol_log=st.floats(min_value=-9.0, max_value=2.0))
def test_drift_origin_is_first_violator(
    layers: tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]],
    tol_log: float,
) -> None:
    pt, ort, order = layers
    tolerance = 10**tol_log
    rpt = compare_python(pt, ort, model="m", tolerance=tolerance, layer_order=order)
    violators = [s.layer for s in rpt.layers if s.exceeds_tol]
    if violators:
        assert rpt.drift_origin == violators[0]
    else:
        assert rpt.drift_origin is None


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(layers=_layer_set(), tol_log=st.floats(min_value=-9.0, max_value=2.0))
def test_counts_are_consistent(
    layers: tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]],
    tol_log: float,
) -> None:
    pt, ort, order = layers
    tolerance = 10**tol_log
    rpt = compare_python(pt, ort, model="m", tolerance=tolerance, layer_order=order)
    assert rpt.layers_total == len(rpt.layers)
    assert rpt.layers_exceeding == sum(1 for s in rpt.layers if s.exceeds_tol)
    # If nothing exceeds, drift_origin is None; otherwise it points to a
    # layer that actually exceeded.
    if rpt.layers_exceeding == 0:
        assert rpt.drift_origin is None
    else:
        names = {s.layer for s in rpt.layers if s.exceeds_tol}
        assert rpt.drift_origin in names


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(layers=_layer_set())
def test_identical_arrays_produce_zero_diff(
    layers: tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]],
) -> None:
    pt, _ort, order = layers
    rpt = compare_python(pt, pt, model="m", tolerance=1e-9, layer_order=order)
    assert rpt.layers_exceeding == 0
    assert rpt.drift_origin is None
    for s in rpt.layers:
        assert s.max_abs_diff == 0.0
        assert s.mean_abs_diff == 0.0
