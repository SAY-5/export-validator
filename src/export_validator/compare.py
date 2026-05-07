"""Per-layer divergence comparison.

Two paths exist:
- ``compare_python``: pure-Python NumPy implementation. Always available.
- ``compare_native``: shells out to the ``export_validator_compare`` C++
  binary (if found on ``PATH`` or pointed to by ``EXPORT_VALIDATOR_BINARY``).
  The C++ implementation is byte-identical to the Python fallback by design;
  ``tests/integration/test_parity.py`` enforces this.

Both produce the same dict shape so the report layer never needs to branch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class LayerStat:
    """Per-layer divergence record."""

    layer: str
    shape: list[int]
    max_abs_diff: float
    mean_abs_diff: float
    exceeds_tol: bool


@dataclass(frozen=True)
class CompareReport:
    """Top-level comparison output."""

    model: str
    tolerance: float
    layers: list[LayerStat]
    drift_origin: str | None
    layers_total: int
    layers_exceeding: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _round(value: float, digits: int = 12) -> float:
    """Round to a deterministic precision so JSON output is reproducible."""
    if not np.isfinite(value):
        return float(value)
    return float(np.format_float_scientific(value, precision=digits, unique=False))


def compare_python(
    pt: dict[str, NDArrayF32],
    ort: dict[str, NDArrayF32],
    *,
    model: str,
    tolerance: float,
    layer_order: list[str],
) -> CompareReport:
    """Compare two name-keyed activation dicts and produce a CompareReport.

    Layers in ``layer_order`` that are missing from either side are skipped
    silently — those would have been filtered out at export time.
    """
    stats: list[LayerStat] = []
    drift_origin: str | None = None
    exceeding = 0
    for name in layer_order:
        if name not in pt or name not in ort:
            continue
        a = pt[name].astype(np.float64, copy=False)
        b = ort[name].astype(np.float64, copy=False)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {name}: {a.shape} vs {b.shape}")
        diff = np.abs(a - b)
        max_abs = _round(float(diff.max())) if diff.size else 0.0
        mean_abs = _round(float(diff.mean())) if diff.size else 0.0
        exceeds = bool(max_abs > tolerance)
        if exceeds and drift_origin is None:
            drift_origin = name
        if exceeds:
            exceeding += 1
        stats.append(
            LayerStat(
                layer=name,
                shape=list(a.shape),
                max_abs_diff=max_abs,
                mean_abs_diff=mean_abs,
                exceeds_tol=exceeds,
            )
        )
    return CompareReport(
        model=model,
        tolerance=tolerance,
        layers=stats,
        drift_origin=drift_origin,
        layers_total=len(stats),
        layers_exceeding=exceeding,
    )


def native_binary() -> Path | None:
    """Resolve the optional C++ comparator binary, if available."""
    explicit = os.environ.get("EXPORT_VALIDATOR_BINARY")
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
    found = shutil.which("export_validator_compare")
    if found:
        return Path(found)
    # Common build location.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "build" / "cpp" / "export_validator_compare"
        if candidate.is_file():
            return candidate
        if (parent / "CMakeLists.txt").is_file():
            break
    return None


def compare_native(
    pt_path: Path,
    ort_path: Path,
    *,
    model: str,
    tolerance: float,
    layer_order: list[str],
) -> CompareReport:
    """Run the C++ comparator over the two ``.evl1`` files and parse its JSON."""
    binary = native_binary()
    if binary is None:
        raise RuntimeError("C++ comparator binary not found")
    layer_list_path = pt_path.with_suffix(".layers.txt")
    layer_list_path.write_text("\n".join(layer_order) + "\n")
    cmd = [
        str(binary),
        "--pt",
        str(pt_path),
        "--ort",
        str(ort_path),
        "--model",
        model,
        "--tolerance",
        repr(tolerance),
        "--layers",
        str(layer_list_path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    return CompareReport(
        model=payload["model"],
        tolerance=payload["tolerance"],
        layers=[LayerStat(**row) for row in payload["layers"]],
        drift_origin=payload["drift_origin"],
        layers_total=payload["layers_total"],
        layers_exceeding=payload["layers_exceeding"],
    )


def save_npz(path: Path, arrays: dict[str, NDArrayF32]) -> None:
    """Save a name-keyed dict of float32 arrays to .npz (Python consumers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: v.astype(np.float32, copy=False) for k, v in arrays.items()})


def save_evl1(path: Path, arrays: dict[str, NDArrayF32]) -> None:
    """Write a name-keyed dict in the simple binary format read by the C++ runner.

    Format (little-endian, fixed-width):

      magic = "EVL1"; version = 1; n_layers u32
      per layer: name_len u32 | name | ndim u32 | dims i64[ndim] |
                 dtype u32 (0 = float32) | count u64 | data float32[count]

    Keeps the C++ runtime free of any third-party dependency.
    """
    import struct

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(b"EVL1")
        fh.write(struct.pack("<II", 1, len(arrays)))
        for name, arr in arrays.items():
            arr32 = np.ascontiguousarray(arr.astype(np.float32, copy=False))
            name_bytes = name.encode("utf-8")
            fh.write(struct.pack("<I", len(name_bytes)))
            fh.write(name_bytes)
            fh.write(struct.pack("<I", arr32.ndim))
            for d in arr32.shape:
                fh.write(struct.pack("<q", int(d)))
            fh.write(struct.pack("<I", 0))  # dtype: float32
            fh.write(struct.pack("<Q", int(arr32.size)))
            fh.write(arr32.tobytes(order="C"))
