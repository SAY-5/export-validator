# Tolerance

## Why 1e-4 by default

CPU-only PyTorch and CPU-only ONNX Runtime should be mathematically
identical for FP32 — both implement the same operators on IEEE-754
doubles internally. In practice:

- Different fused-multiply-add ordering between BLAS implementations.
- Different SIMD widths (PyTorch's MKL/OpenBLAS vs ORT's MLAS).
- ORT's graph-level constant folding shifts the order of additions.

Empirically on ResNet-18 fp32 with this code on macOS arm64:

| metric | layer | value |
|---|---|---|
| max max_abs_diff (any layer) | layer4.1.relu | 9.537e-06 |
| min max_abs_diff (any layer) | layer4.0.downsample.0 | 5.811e-07 |

So `1e-4` gives ~10x headroom over what FP32 ResNet-18 actually
exhibits. That headroom matters because:

- FP16 export typically introduces 1e-3 to 1e-2 per layer.
- ORT's CUDA EP applies kernels that can drift further than CPU.
- A library upgrade that introduces a small regression (e.g. a new
  ORT kernel rewrite) tends to shift the per-layer numbers by 1e-5
  to 1e-4. A tighter tolerance would flag that as drift; `1e-4`
  treats it as noise.

If you tighten to `1e-5`, this repo's committed report would still
show `layers_exceeding: 0`. At `1e-6`, every layer flips to
`exceeds_tol: yes` and the tool becomes useless on FP32. Pick a
tolerance that is ~10x your expected noise floor.

## What the comparison metric is

`max_abs_diff = max(abs(pt - ort))` over every element of the layer
output, computed in float64.

`mean_abs_diff = mean(abs(pt - ort))`, also float64.

Relative diff (`abs(pt - ort) / (abs(pt) + eps)`) is *not* reported;
it is unstable when `pt ≈ 0` and disproportionately punishes BN/ReLU
layers that pin many pixels to zero. Stick with absolute diff.

## What `exceeds_tol` means

`max_abs_diff > tolerance`. Note: strictly greater. A layer whose
max diff equals the tolerance exactly is *not* flagged. The C++ and
Python implementations both follow this rule (regression-tested in
`Differ.ToleranceIsStrictlyGreater`).
