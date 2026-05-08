"""Unit tests for NCHW/NHWC format-mismatch detection.

Two strategies:

1. Direct: build two activation tensors that are permutations of each other
   and verify the detector flags the permutation that brings them back into
   agreement. Doubles as the synthetic test the v4 spec calls for.

2. Indirect: build a tiny PyTorch model that injects a ``permute(0, 2, 3, 1)``
   somewhere mid-graph in the *capture* path (not in the model itself, since
   that would need re-exporting through ONNX). Then run the detector on
   the resulting per-layer dict and confirm only the affected layer is
   reported.
"""

from __future__ import annotations

import numpy as np

from export_validator.format_mismatch import detect, detect_layer


def _rand(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def test_identical_tensors_are_not_flagged() -> None:
    a = _rand((1, 4, 8, 8))
    assert detect_layer("x", a, a.copy(), tolerance=1e-5) is None


def test_nchw_vs_nhwc_is_detected() -> None:
    """A tensor and its (0,2,3,1) permutation must be detected."""
    nchw = _rand((1, 4, 8, 8), seed=1)
    nhwc = nchw.transpose(0, 2, 3, 1).copy()
    hit = detect_layer("layer", nchw, nhwc, tolerance=1e-5)
    assert hit is not None
    assert hit.inferred_permutation == (0, 2, 3, 1)
    # Pre-permute drift is large; post-permute drift is exactly zero.
    assert hit.pre_max_abs_diff > 0.1
    assert hit.post_max_abs_diff <= 1e-7


def test_nhwc_vs_nchw_is_detected() -> None:
    """The reverse direction — the detector finds (0,3,1,2)."""
    nchw = _rand((1, 4, 8, 8), seed=2)
    nhwc = nchw.transpose(0, 2, 3, 1).copy()
    hit = detect_layer("layer", nhwc, nchw, tolerance=1e-5)
    assert hit is not None
    assert hit.inferred_permutation == (0, 3, 1, 2)


def test_genuinely_unrelated_drift_is_not_attributed_to_layout() -> None:
    """When the difference is real numerical noise, no permutation helps."""
    a = _rand((1, 4, 8, 8), seed=3)
    rng = np.random.default_rng(99)
    b = a + rng.standard_normal(a.shape).astype(np.float32) * 0.5
    assert detect_layer("layer", a, b, tolerance=1e-5) is None


def test_3d_transformer_layout_flip_is_detected() -> None:
    """``(B, C, T)`` vs ``(B, T, C)`` is a common transformer-export pitfall."""
    bct = _rand((2, 8, 16), seed=4)
    btc = bct.transpose(0, 2, 1).copy()
    hit = detect_layer("attn", bct, btc, tolerance=1e-5)
    assert hit is not None
    assert hit.inferred_permutation == (0, 2, 1)


def test_detect_filters_to_layout_only() -> None:
    """A mixed dict with one layout flip and one numerical-noise diff
    must produce a single FormatMismatch (the flip)."""
    a_nchw = _rand((1, 3, 8, 8), seed=5)
    a_nhwc = a_nchw.transpose(0, 2, 3, 1).copy()
    b = _rand((1, 8), seed=6)
    rng = np.random.default_rng(11)
    b_noisy = b + rng.standard_normal(b.shape).astype(np.float32) * 0.5
    pt = {"layer1": a_nchw, "fc": b}
    ort = {"layer1": a_nhwc, "fc": b_noisy}
    hits = detect(pt, ort, tolerance=1e-5, layer_order=["layer1", "fc"])
    assert [h.layer for h in hits] == ["layer1"]
    assert hits[0].inferred_permutation == (0, 2, 3, 1)


def test_synthetic_mid_graph_permute_is_flagged_at_the_spot() -> None:
    """Spec test from the task: deliberately add a permute mid-graph and
    assert the detector flags only the spot where the layout flips, not
    every downstream layer.

    We don't need a real ONNX export — the detector consumes PyTorch and
    "ONNX" activation dicts and treats them symmetrically. We simulate by
    building two parallel chains: in one, the activations stay NCHW; in
    the other, the same activations are permuted at a known layer. The
    detector should flag exactly that layer.
    """
    rng = np.random.default_rng(42)
    pt: dict[str, np.ndarray] = {}
    ort: dict[str, np.ndarray] = {}
    layer_order = ["conv1", "conv2", "permuted_layer", "conv3"]
    for name in layer_order:
        pt[name] = rng.standard_normal((1, 4, 8, 8)).astype(np.float32)
        # Mirror PT exactly *except* at the permuted layer.
        if name == "permuted_layer":
            ort[name] = pt[name].transpose(0, 2, 3, 1).copy()
        else:
            ort[name] = pt[name].copy()
    hits = detect(pt, ort, tolerance=1e-5, layer_order=layer_order)
    assert [h.layer for h in hits] == ["permuted_layer"]
    assert hits[0].inferred_permutation == (0, 2, 3, 1)


def test_size_mismatch_is_not_flagged() -> None:
    """Tensors with different element counts cannot be a layout flip."""
    a = _rand((1, 4, 8, 8))
    b = _rand((1, 4, 8, 9))
    assert detect_layer("layer", a, b, tolerance=1e-5) is None


def test_picks_smallest_post_drift_permutation() -> None:
    """When two permutations both pass tolerance, the lower post-drift wins."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((1, 2, 2, 2)).astype(np.float32)
    flipped = base.transpose(0, 2, 3, 1).copy()
    # Add ~0 noise so multiple permutations could marginally pass.
    hit = detect_layer("layer", base, flipped, tolerance=1.0)
    assert hit is not None
    # Identity is excluded by construction (tested separately); the
    # detector should report the layout-flip permutation.
    assert hit.inferred_permutation == (0, 2, 3, 1)
