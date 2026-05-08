"""Command-line interface for the export validator."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch

from . import compare as cmp
from . import export as exp
from . import ort_capture, pt_capture, report
from .models import (
    build_mobilenet_v3_small,
    build_resnet18,
    build_resnet50,
    build_vit_b_16,
)

_MODELS = {
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "mobilenet_v3_small": build_mobilenet_v3_small,
    "vit_b_16": build_vit_b_16,
}


@click.group()
def main() -> None:
    """Export PyTorch models to ONNX and validate per-layer parity."""


@main.command()
@click.option("--model", "model_name", type=click.Choice(list(_MODELS)), default="resnet18")
@click.option("--out", type=click.Path(path_type=Path), default=Path("examples/resnet18.onnx"))
@click.option(
    "--layer-map",
    type=click.Path(path_type=Path),
    default=Path("examples/resnet18_layer_map.json"),
)
def export(model_name: str, out: Path, layer_map: Path) -> None:
    """Export a model to ONNX with per-layer named outputs."""
    model, sample = _MODELS[model_name]()
    info = exp.export_with_named_layers(model, sample, out, layer_map)
    click.echo(f"exported {model_name} to {out} ({len(info['layers'])} named layers)")


@main.command()
@click.option("--model", "model_name", type=click.Choice(list(_MODELS)), default="resnet18")
@click.option("--onnx", "onnx_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--layer-map",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--report-base",
    type=click.Path(path_type=Path),
    default=Path("examples/reports/resnet18_fp32"),
    help="Base path for .json and .md report files (extension stripped).",
)
@click.option("--tolerance", type=float, default=1e-4)
@click.option(
    "--backend",
    type=click.Choice(["python", "native", "auto"]),
    default="auto",
    help="Comparator backend. 'auto' prefers C++ if available.",
)
@click.option("--seed", type=int, default=42)
def compare(
    model_name: str,
    onnx_path: Path,
    layer_map: Path,
    report_base: Path,
    tolerance: float,
    backend: str,
    seed: int,
) -> None:
    """Run PyTorch + ONNX Runtime, compare per layer, write reports."""
    model, _ = _MODELS[model_name]()
    layer_info = json.loads(layer_map.read_text())
    layer_order: list[str] = layer_info["layers"]

    generator = torch.Generator().manual_seed(seed)
    sample = torch.randn(1, 3, 224, 224, generator=generator)

    pt_acts = pt_capture.run(model, sample)
    ort_acts = ort_capture.run(onnx_path, sample.numpy())
    # ORT also returns the final classifier output under 'output'; drop it
    # from the per-layer view because it is not in the model's leaves.
    ort_layer_acts = {k: v for k, v in ort_acts.items() if k in pt_acts}

    use_native = backend == "native" or (backend == "auto" and cmp.native_binary())
    if use_native:
        pt_evl = report_base.with_name(report_base.name + "_pt.evl1")
        ort_evl = report_base.with_name(report_base.name + "_ort.evl1")
        cmp.save_evl1(pt_evl, pt_acts)
        cmp.save_evl1(ort_evl, ort_layer_acts)
        rpt = cmp.compare_native(
            pt_evl,
            ort_evl,
            model=model_name,
            tolerance=tolerance,
            layer_order=layer_order,
        )
        backend_name = "native"
    else:
        rpt = cmp.compare_python(
            pt_acts,
            ort_layer_acts,
            model=model_name,
            tolerance=tolerance,
            layer_order=layer_order,
        )
        backend_name = "python"

    json_path, md_path = report.write_outputs(rpt, report_base)
    click.echo(
        f"compared via {backend_name}: "
        f"{rpt.layers_exceeding}/{rpt.layers_total} layers exceed tol={tolerance:g}; "
        f"drift_origin={rpt.drift_origin or 'none'}"
    )
    click.echo(f"json: {json_path}")
    click.echo(f"md:   {md_path}")


@main.command()
@click.option("--model", "model_name", type=click.Choice(list(_MODELS)), default="resnet18")
@click.option("--out-dir", type=click.Path(path_type=Path), default=Path("examples/inputs"))
@click.option("--n", "count", type=int, default=8)
@click.option("--seed", type=int, default=42)
def inputs(model_name: str, out_dir: Path, count: int, seed: int) -> None:
    """Materialise N deterministic input tensors as a single .npz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_name}_inputs.npz"
    rng = np.random.default_rng(seed)
    arrays = {
        f"x{i:02d}": rng.standard_normal((1, 3, 224, 224)).astype(np.float32) for i in range(count)
    }
    np.savez(out, **arrays)
    click.echo(f"wrote {count} deterministic inputs to {out}")


if __name__ == "__main__":
    main()
