"""Versioned persistence for complete NTN channel realizations."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.dataset import load_cir, save_cir
from channel_dataset.channels.wireless.tr38811.types import (
    NTNChannelRealization,
    NTNLargeScaleComponents,
    NTNLinkGeometry,
    NTNRealizationDiagnostics,
)
from channel_dataset.core import ExecutionContext

_GEOMETRY_FIELDS = (
    "slant_range_m", "elevation_rad", "azimuth_rad", "central_angle_rad",
    "propagation_delay_s", "radial_velocity_mps", "common_doppler_hz",
    "visibility", "direction_rx_to_tx",
)
_LARGE_SCALE_FIELDS = (
    "free_space_db", "clutter_db", "shadow_fading_db", "building_entry_db",
    "atmospheric_gas_db", "rain_db", "cloud_db", "scintillation_db", "total_db",
)


def _numpy(value: Any, backend_name: str) -> np.ndarray:
    return np.asarray(value) if backend_name == "numpy" else value.detach().cpu().numpy()


def _write_array(root: Path, name: str, value: Any, backend_name: str) -> dict[str, Any]:
    filename = f"{name}.npy"
    path = root / filename
    array = _numpy(value, backend_name)
    np.save(path, array, allow_pickle=False)
    return {
        "file": filename,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _read_array(root: Path, descriptor: Any, execution: ExecutionContext) -> Any:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "file", "sha256", "shape", "dtype",
    }:
        raise ValueError("invalid NTN array descriptor")
    path = root / descriptor["file"]
    if sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"NTN array checksum mismatch: {descriptor['file']}")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != descriptor["shape"] or array.dtype.str != descriptor["dtype"]:
        raise ValueError("NTN array shape or dtype does not match its manifest")
    return execution.array_backend.array(array, dtype=array.dtype.name, device=execution.device)


def save_ntn_realization(path: str | Path, realization: NTNChannelRealization) -> Path:
    """Persist an NTN realization in a checksummed directory."""

    root = Path(path).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    save_cir(root / "cir", realization.cir)
    backend_name = realization.cir.metadata.backend
    arrays: dict[str, Any] = {}
    for name in _GEOMETRY_FIELDS:
        arrays[f"geometry.{name}"] = _write_array(
            root, f"geometry.{name}", getattr(realization.geometry, name), backend_name
        )
    for name in _LARGE_SCALE_FIELDS:
        arrays[f"large_scale.{name}"] = _write_array(
            root, f"large_scale.{name}", getattr(realization.large_scale, name), backend_name
        )
    arrays["los_mask"] = _write_array(root, "los_mask", realization.los_mask, backend_name)
    diagnostics: dict[str, Any] = {}
    backend = get_backend(backend_name)
    for name, value in realization.diagnostics.values.items():
        if backend.is_array(value):
            diagnostics[name] = {"array": _write_array(root, f"diagnostic.{name}", value, backend_name)}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            diagnostics[name] = {"value": value}
        else:
            raise TypeError(f"unsupported diagnostic value for {name!r}")
    manifest = {
        "schema_version": 1,
        "kind": "NTNChannelRealization",
        "arrays": arrays,
        "diagnostics": diagnostics,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def load_ntn_realization(
    path: str | Path, execution: ExecutionContext
) -> NTNChannelRealization:
    """Load a checksummed NTN realization on the requested backend/device."""

    root = Path(path).resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "kind", "arrays", "diagnostics",
    }:
        raise ValueError("invalid NTN realization manifest")
    if document["schema_version"] != 1 or document["kind"] != "NTNChannelRealization":
        raise ValueError("unsupported NTN realization schema or kind")
    cir = load_cir(root / "cir", execution)
    arrays = document["arrays"]
    expected = (
        {f"geometry.{name}" for name in _GEOMETRY_FIELDS}
        | {f"large_scale.{name}" for name in _LARGE_SCALE_FIELDS}
        | {"los_mask"}
    )
    if not isinstance(arrays, dict) or set(arrays) != expected:
        raise ValueError("NTN realization arrays are incomplete or unknown")
    geometry_values = {
        name: _read_array(root, arrays[f"geometry.{name}"], execution)
        for name in _GEOMETRY_FIELDS
    }
    geometry = NTNLinkGeometry(**geometry_values, metadata=cir.metadata)
    large_scale_values = {
        name: _read_array(root, arrays[f"large_scale.{name}"], execution)
        for name in _LARGE_SCALE_FIELDS
    }
    large_scale = NTNLargeScaleComponents(**large_scale_values, metadata=cir.metadata)
    diagnostics = {}
    for name, descriptor in document["diagnostics"].items():
        if not isinstance(descriptor, dict) or len(descriptor) != 1:
            raise ValueError("invalid NTN diagnostic descriptor")
        diagnostics[name] = (
            _read_array(root, descriptor["array"], execution)
            if "array" in descriptor
            else descriptor["value"]
        )
    return NTNChannelRealization(
        cir,
        geometry,
        large_scale,
        _read_array(root, arrays["los_mask"], execution),
        NTNRealizationDiagnostics(diagnostics),
    )


__all__ = ["load_ntn_realization", "save_ntn_realization"]
