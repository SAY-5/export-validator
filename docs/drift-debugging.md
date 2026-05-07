# Drift debugging

Once `drift_origin` names a layer, here is the workflow for figuring out
why.

## Step 1 — confirm it is real, not a hook artifact

The most common false positive is captured below. If your `drift_origin`
is a `BatchNorm2d` immediately followed by `ReLU(inplace=True)`, and the
divergence is "the negative half is zeroed in PyTorch but not ONNX" (or
vice versa), the bug is in your *capture* code, not the model.

`NamedOutputWrapper` clones every buffered tensor for exactly this reason
— without the clone, the in-place ReLU mutates the same storage that the
ONNX exporter believes contains the BN output. The result is a "BN
output" that is actually the post-ReLU tensor in the ONNX graph but the
pre-ReLU tensor in the PyTorch hook capture.

Symptom: BN layers show ~1-5 unit max abs diff while every adjacent
Conv shows ~1e-6.

## Step 2 — narrow with a single-input bisect

Run the pipeline on a constant-zero input:

```python
import torch
sample = torch.zeros(1, 3, 224, 224)
```

Many layers will produce zero outputs on this input, so any non-zero
`max_abs_diff` is suspicious. Compare to the random-input run.

## Step 3 — inspect the offending layer's ONNX node

```python
import onnx
m = onnx.load("examples/resnet18.onnx")
for node in m.graph.node:
    if "<drift_origin>" in node.output:
        print(node)
```

Look at `node.op_type`. If ORT replaces the op (e.g. fuses `Conv` +
`BatchNorm` into a single `Conv`), the named output points at a tensor
that no longer exists in the original sense, and the comparison becomes
apples-to-oranges. Disable optimisations:

```python
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
```

We do this by default in `ort_capture.run`.

## Step 4 — bisect by tolerance

Re-run with `--tolerance 1e-5`. If `drift_origin` moves earlier in the
network, the original origin was a layer where small upstream drift
became visible (a non-linearity). Whatever layer is now the origin at
tighter tolerance is the better lead.

## Step 5 — check determinism

Set `torch.use_deterministic_algorithms(True)` on the PyTorch side, and
set `sess_options.enable_cpu_mem_arena = False`,
`sess_options.intra_op_num_threads = 1` on the ORT side. Re-run. If the
divergence vanishes, the issue is parallel-reduction order, not a
mathematical disagreement.

## Step 6 — when the model is your code

If the drift is in a custom module, the most common causes are:

- a parameter not registered as `nn.Parameter` (so `state_dict` doesn't
  carry it through to the export),
- an op torch traces differently in eager vs script mode (e.g. boolean
  masks, `tensor.item()` calls),
- a constant folded to the wrong dtype during export.

Check the ONNX graph's initializers for the offending layer:

```python
for init in m.graph.initializer:
    if init.name.startswith("<drift_origin>."):
        print(init.name, init.dims, init.data_type)
```
