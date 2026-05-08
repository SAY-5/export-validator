"""Render comparison results to JSON and Markdown.

The primary JSON report (``<base>.json``) is the byte-for-byte contract
shared between the Python and C++ comparators — no attribution data leaks
into it. Per-layer cause attributions, when produced, are written to a
separate sidecar (``<base>_causes.json``) and included in the Markdown
report. This keeps the Python ↔ C++ parity invariant intact.
"""

from __future__ import annotations

import json
from pathlib import Path

from .attribution import CauseAssignment
from .compare import CompareReport


def _fmt_shape(shape: list[int]) -> str:
    return "(" + ",".join(str(d) for d in shape) + ")"


def _fmt_float(value: float) -> str:
    if value == 0.0:
        return "0.0"
    return f"{value:.3e}"


def render_markdown(
    report: CompareReport,
    causes: list[CauseAssignment] | None = None,
) -> str:
    """Emit a Markdown report. Sorted by layer order from the comparator.

    If ``causes`` is provided, each violating layer's row gains a
    ``cause`` column and a follow-up section lists every cause and its
    detail string.
    """
    cause_by_layer: dict[str, CauseAssignment] = {}
    if causes:
        for c in causes:
            cause_by_layer[c.layer] = c
    show_cause = bool(cause_by_layer)

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
    if show_cause:
        lines.append("| layer | shape | max_abs_diff | mean_abs_diff | exceeds_tol | cause |")
        lines.append("|---|---|---:|---:|:-:|---|")
    else:
        lines.append("| layer | shape | max_abs_diff | mean_abs_diff | exceeds_tol |")
        lines.append("|---|---|---:|---:|:-:|")
    for s in report.layers:
        row = (
            f"| `{s.layer}` | {_fmt_shape(s.shape)} | {_fmt_float(s.max_abs_diff)} "
            f"| {_fmt_float(s.mean_abs_diff)} | {'yes' if s.exceeds_tol else '—'}"
        )
        if show_cause:
            cell = cause_by_layer.get(s.layer)
            row += f" | `{cell.cause}`" if cell else " | —"
        row += " |"
        lines.append(row)
    if report.layers_exceeding == 0:
        lines.append("")
        lines.append(
            f"No drift detected at tolerance {report.tolerance:g} "
            f"across {report.layers_total} layers."
        )
    if show_cause and cause_by_layer:
        lines.append("")
        lines.append("## Cause attribution")
        lines.append("")
        for s in report.layers:
            entry = cause_by_layer.get(s.layer)
            if entry is None:
                continue
            lines.append(f"- `{entry.layer}` → **{entry.cause}** ({entry.detail})")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: CompareReport,
    base: Path,
    causes: list[CauseAssignment] | None = None,
) -> tuple[Path, Path]:
    """Write ``base.json`` and ``base.md``. Returns the two paths.

    When ``causes`` is provided, also writes ``<base>_causes.json`` so
    consumers (CI, downstream tooling) can read the attribution without
    having to re-parse Markdown. The primary JSON file remains the
    Python-↔-C++ byte-equal contract and never carries cause data.
    """
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report, causes))
    if causes is not None:
        sidecar = base.with_name(base.name + "_causes.json")
        payload = {
            "model": report.model,
            "tolerance": report.tolerance,
            "causes": [{"layer": c.layer, "cause": c.cause, "detail": c.detail} for c in causes],
        }
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return json_path, md_path
