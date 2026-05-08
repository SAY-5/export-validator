"""Detect NCHW ↔ NHWC layout mismatches between PyTorch and ONNX activations.

PyTorch defaults to NCHW (channels-first); some ONNX consumers (TFLite,
older mobile runtimes, hand-written CUDA kernels) want NHWC (channels-last).
A mismatch produces *silent* drift: every value is still present, just in
the wrong axis order, so a naive ``np.abs(pt - ort).max()`` reports a large
diff with no obvious root cause. From the per-layer report, that looks
like an op-implementation mismatch — but the fix is one transpose, not a
deep dive into kernel numerics.

The detector here takes the same per-layer activation pair the differ
already consumes and asks: **for every layer that exceeds tolerance, does
permuting one of the two tensors restore agreement?** If yes, we record
the inferred permutation. Concretely the heuristic walks the four common
4-D permutations against NCHW (identity, NHWC, CHWN, HWCN) plus a few
3-D variants for transformer-style ``(B, T, C)`` tensors and reports the
first permutation under which max-abs drops back below the original
tolerance.

This is intentionally a structural detector, not a layout solver: it does
not attempt to *fix* the mismatch by adding transposes, only to flag the
exact layer where the layout flip is observed. Cross-link: see
``attribution.py`` for cause classification of *non-layout* drift.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.float32]

# 4-D permutations: NCHW reordered four common ways. (0,1,2,3) is identity.
_PERMS_4D: tuple[tuple[int, int, int, int], ...] = (
    (0, 2, 3, 1),  # NCHW -> NHWC
    (0, 3, 1, 2),  # NHWC -> NCHW
    (0, 2, 1, 3),  # transpose H/C
    (0, 1, 3, 2),  # transpose H/W
    (0, 3, 2, 1),  # full reverse-non-batch
    (3, 1, 2, 0),  # batch <-> last
)

# 3-D permutations: ``(B, C, T)`` ↔ ``(B, T, C)`` etc.
_PERMS_3D: tuple[tuple[int, int, int], ...] = (
    (0, 2, 1),
    (1, 0, 2),
    (2, 1, 0),
)


@dataclass(frozen=True)
class FormatMismatch:
    """One detected layout mismatch, anchored to a layer."""

    layer: str
    inferred_permutation: tuple[int, ...]
    pre_max_abs_diff: float
    post_max_abs_diff: float


def _max_abs_diff(a: NDArray, b: NDArray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _shape_compatible(
    shape_a: tuple[int, ...], shape_b: tuple[int, ...], perm: tuple[int, ...]
) -> bool:
    if len(shape_a) != len(perm) or len(shape_b) != len(perm):
        return False
    return tuple(shape_a[i] for i in perm) == shape_b


def _permutations_for(ndim: int) -> Iterable[tuple[int, ...]]:
    if ndim == 4:
        return _PERMS_4D
    if ndim == 3:
        return _PERMS_3D
    return ()


def detect_layer(
    layer: str,
    pt: NDArray,
    ort: NDArray,
    *,
    tolerance: float,
) -> FormatMismatch | None:
    """Return a :class:`FormatMismatch` if a permutation restores agreement.

    The two tensors must have the same total element count. If the shapes
    are already equal, we still try every permutation that maps shape A
    onto shape B; some hosts can lay activations out NHWC under the same
    shape tuple as NCHW when channel == 1 (a degenerate but real case).
    """
    if pt.size == 0 or pt.size != ort.size:
        return None
    pre = _max_abs_diff(pt, ort) if pt.shape == ort.shape else float("inf")
    if pre <= tolerance:
        return None
    best: FormatMismatch | None = None
    for perm in _permutations_for(pt.ndim):
        if not _shape_compatible(pt.shape, ort.shape, perm) and pt.shape != ort.shape:
            continue
        try:
            post = _max_abs_diff(np.transpose(pt, perm), ort)
        except (ValueError, TypeError):
            continue
        if post <= tolerance and (best is None or post < best.post_max_abs_diff):
            best = FormatMismatch(
                layer=layer,
                inferred_permutation=perm,
                pre_max_abs_diff=pre,
                post_max_abs_diff=post,
            )
    return best


def detect(
    pt: dict[str, NDArray],
    ort: dict[str, NDArray],
    *,
    tolerance: float,
    layer_order: list[str] | None = None,
) -> list[FormatMismatch]:
    """Run :func:`detect_layer` over every layer present in both dicts.

    The output is a list of :class:`FormatMismatch` records, one per layer
    where a permutation restored agreement under ``tolerance``. Layers
    whose drift is *not* explained by a layout flip are omitted.
    """
    layers = layer_order if layer_order is not None else list(pt.keys() & ort.keys())
    out: list[FormatMismatch] = []
    for name in layers:
        if name not in pt or name not in ort:
            continue
        hit = detect_layer(name, pt[name], ort[name], tolerance=tolerance)
        if hit is not None:
            out.append(hit)
    return out
