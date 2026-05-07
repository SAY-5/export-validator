"""Render comparison results to JSON and Markdown."""

from __future__ import annotations

import json
from pathlib import Path

from .compare import CompareReport


def _fmt_shape(shape: list[int]) -> str:
    return "(" + ",".join(str(d) for d in shape) + ")"


def _fmt_float(value: float) -> str:
    if value == 0.0:
        return "0.0"
    return f"{value:.3e}"


def render_markdown(report: CompareReport) -> str:
    """Emit a Markdown report. Sorted by layer order from the comparator."""
    lines: list[str] = []
    lines.append(f"# Per-layer divergence: {report.model}")
    lines.append("")
    lines.append(
        f"Tolerance: {report.tolerance:g}  ·  "
        f"layers checked: {report.layers_total}  ·  "
        f"layers exceeding: {report.layers_exceeding}"
    )
    drift = report.drift_origin if report.drift_origin else "none"
    lines.append(f"Drift origin: {drift}")
    lines.append("")
    lines.append("| layer | shape | max_abs_diff | mean_abs_diff | exceeds_tol |")
    lines.append("|---|---|---:|---:|:-:|")
    for s in report.layers:
        lines.append(
            f"| `{s.layer}` | {_fmt_shape(s.shape)} | {_fmt_float(s.max_abs_diff)} "
            f"| {_fmt_float(s.mean_abs_diff)} | {'yes' if s.exceeds_tol else '—'} |"
        )
    if report.layers_exceeding == 0:
        lines.append("")
        lines.append(
            f"No drift detected at tolerance {report.tolerance:g} "
            f"across {report.layers_total} layers."
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: CompareReport, base: Path) -> tuple[Path, Path]:
    """Write ``base.json`` and ``base.md``. Returns the two paths."""
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
