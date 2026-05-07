# Architecture

## Goal

Walk a PyTorch model leaf-by-leaf and report, for every leaf, the absolute
difference between PyTorch's activation and ONNX Runtime's activation when
both are fed the same float32 input bytes. The first leaf whose max-abs
diff exceeds tolerance is the `drift_origin`.

## The named-output trick

`torch.onnx.export` accepts an `output_names` argument, but the model must
*return* tensors with those names from `forward`. In a stock
`torchvision.models.resnet18`, only `output` (the final classifier logits)
is returned, so naming the 60 intermediate activations needs a wrapper.

`NamedOutputWrapper` registers a forward hook on every leaf module that
appends the module's output tensor to an internal buffer. `forward` then
returns `(final, *buffer)`. ONNX export sees one tuple element per named
graph output, and the order in which leaves run during `forward` becomes
the order of the ONNX outputs.

There are two non-obvious failure modes:

1. **In-place ReLU after BN.** ResNet uses `nn.ReLU(inplace=True)` after
   every BatchNorm. The wrapper's hook captures the BN tensor *before*
   ReLU runs, but the captured Python reference and the BN's own output
   share storage. When the in-place ReLU mutates that storage, the
   ONNX exporter (which has already added the buffered tensor to the
   graph output list) ends up emitting the *post-ReLU* tensor. Solution:
   `output.clone()` inside the stash hook.

2. **Module-order vs execution-order.** A naive implementation of
   `output_names` enumerates `model.named_modules()` to derive the name
   list, but Bottleneck/BasicBlock blocks share submodules across forward
   paths and the ReLU module gets called twice per block. The wrapper
   does one dry forward pass to populate `execution_order`, then uses that
   list (which can have duplicates — that is correct) for `output_names`.

## The C++ comparator

`cpp/src/main.cpp` reads two simple binary files in the `EVL1` format
(documented in `cpp/include/export_validator/npz_loader.h`), compares them
layer-by-layer with the same logic as the Python fallback, and prints
JSON to stdout. The Python side writes the binary files via
`compare.save_evl1`.

### Why a custom binary format

`.npz` is a ZIP file. Reading ZIP from C++ without a third-party library
is a lot of code. The binary `EVL1` format takes 30 lines of C++ and
includes only what we need: `(name, shape, float32 data)` records.

### Byte-identical JSON parity contract

The Python report goes through `json.dumps(..., indent=2, sort_keys=True)`.
The C++ writer in `json_writer.cpp` emits keys in the same alphabetical
order at every level and uses `%.13g` for doubles, which round-trips
through Python's `float()` to the same IEEE bits. To keep the floating-
point text representation identical, both sides round their per-layer
metrics to 12 significant digits (`%.12e` -> parse) before serialization.
`tests/integration/test_python_cpp_parity.py` asserts byte-equality.

### libtorch + onnxruntime smoke binary

A second C++ target, `export_validator_runtimes`, links libtorch (from
the Python torch wheel's bundled `share/cmake/Torch/`) and onnxruntime
(from a passed-in `ONNXRUNTIME_DIR`). It only prints version strings and
runs `torch::ones({2,2}).sum()`. The point is to prove the linker line
works on this host without making the comparator path depend on a C++
ResNet implementation.

## Tolerance

- FP32 export to ONNX is mathematically lossless on paper but each
  operator's CPU implementation differs in fma, ordering, and SIMD width
  between PyTorch and ORT. Empirically, ResNet-18 lands at ~1e-5 max abs
  diff per layer.
- The default `1e-4` tolerance accommodates that headroom while still
  catching FP16 (which typically introduces 1e-3 to 1e-2 per layer) or
  ops that ORT replaces with non-equivalent kernels.
- Tolerance comparison is `max_abs_diff > tolerance` — strictly greater,
  so the boundary value is *not* flagged.

## Drift-origin heuristic and its limits

`drift_origin = first layer in execution order with max_abs_diff > tol`.

This points at *where* divergence emerges, not *why*. If layer N drifts,
the root cause may be:
- a single operator in N (most common for FP16 BN, fused Conv+BN)
- N inheriting a small numerical perturbation from layer N-1 that crosses
  a non-linear threshold (ReLU, MaxPool index argmax)
- a graph-rewriting pass in ORT that fuses/replaces N's implementation

The report does not attempt to disambiguate these. See
`docs/drift-debugging.md` for what to do once you have an origin.
