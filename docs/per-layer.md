# Per-layer comparison

The validator walks a model's leaf modules. A "leaf" is any
`nn.Module` whose `children()` iterator is empty — `nn.Conv2d`,
`nn.BatchNorm2d`, `nn.ReLU`, `nn.Linear`, `nn.AdaptiveAvgPool2d`, etc.
Composite modules (Bottleneck, Sequential, the model itself) are walked
through, not measured.

A layer's "output" is whatever its `forward(...)` returns. If a leaf
returns a tuple (uncommon but not unheard-of in custom blocks), it is
silently filtered out by `select_exportable_layers`. The remaining leaves
form the per-layer report.

## ResNet-18 layer count

The committed run has 60 layers. ResNet-18 has 21 leaf module *instances*
in `named_modules` but several are reused across the forward pass:

- The two `nn.ReLU` instances in each `BasicBlock` are actually the same
  shared `relu` attribute, called twice per block — once after the first
  BN, once after the residual add.
- The block-level shared ReLUs explain why the report contains lines like
  `layer1.0.relu` twice with identical numbers — the hook fires on each
  execution.

## Reading the JSON report

```json
{
  "model": "resnet18",
  "tolerance": 0.0001,
  "drift_origin": null,
  "layers_total": 60,
  "layers_exceeding": 0,
  "layers": [
    { "layer": "conv1", "shape": [1,64,112,112],
      "max_abs_diff": 3.815e-06, "mean_abs_diff": 9.565e-08,
      "exceeds_tol": false },
    ...
  ]
}
```

Order of `layers` is execution order, which is what you want for the
`drift_origin` heuristic to make sense.

## When `drift_origin` is misleading

A layer's small drift can rocket through the next non-linearity if it
crosses a decision boundary. Specifically:

- `MaxPool2d`: a tiny diff can flip the argmax; downstream values jump.
- `ReLU`: a value crossing zero flips between 0 and v. The diff at the
  ReLU layer looks small (just the affected pixels), but every layer
  after it inherits a clamped vs unclamped pixel.
- `argmax` / `topk`: rank-sensitive, completely discontinuous.

If the reported `drift_origin` is the first such layer after a tiny
upstream drift, the *root cause* is the upstream layer; the report just
catches the layer where the drift becomes visible.
