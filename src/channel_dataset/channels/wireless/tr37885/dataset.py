"""Checksummed persistence for complete TR 37.885 V2X realizations."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.dataset import load_cir, save_cir
from channel_dataset.channels.wireless.tr37885.types import (
    V2XChannelRealization,
    V2XChannelState,
    V2XLargeScaleComponents,
    V2XLinkConditionBatch,
    V2XRealizationDiagnostics,
    V2XSceneTrajectory,
)
from channel_dataset.channels.wireless.types import TopologyBatch
from channel_dataset.core import ExecutionContext, RealizationMetadata

_TOPOLOGY = (
    "tx_positions_m", "rx_positions_m", "tx_velocities_mps", "rx_velocities_mps",
    "tx_orientations_rad", "rx_orientations_rad", "rx_indoor", "tx_site_ids",
    "los_mask", "link_mask",
)
_CONDITIONS = (
    "condition_codes", "same_street", "building_obstructed", "los_probability",
    "blocker_type_codes", "blocker_heights_m", "episode_indices",
)
_LARGE_SCALE = ("path_loss_db", "shadow_fading_db", "vehicle_blockage_db", "total_loss_db")
_STATE = ("condition_codes", "shadow_fading_db", "episode_indices")
_SCENE = (
    "positions_m", "velocities_mps", "orientations_rad", "active_mask",
    "lane_ids", "street_ids", "vehicle_ids", "vehicle_type_codes",
    "dimensions_m", "antenna_heights_m", "mobility_times_s",
)


def _numpy(value: Any, backend_name: str) -> np.ndarray:
    return np.asarray(value) if backend_name == "numpy" else value.detach().cpu().numpy()


def _write(root: Path, name: str, value: Any, backend_name: str) -> dict[str, Any]:
    path = root / f"{name}.npy"
    array = _numpy(value, backend_name)
    np.save(path, array, allow_pickle=False)
    return {
        "file": path.name,
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _read(root: Path, descriptor: Any, execution: ExecutionContext) -> Any:
    expected = {"file", "sha256", "shape", "dtype"}
    if not isinstance(descriptor, dict) or set(descriptor) != expected:
        raise ValueError("invalid V2X array descriptor")
    path = root / descriptor["file"]
    if sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"V2X array checksum mismatch: {descriptor['file']}")
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != descriptor["shape"] or array.dtype.str != descriptor["dtype"]:
        raise ValueError("V2X array shape or dtype does not match manifest")
    return execution.array_backend.array(array, dtype=array.dtype.name, device=execution.device)


def save_v2x_realization(path: str | Path, realization: V2XChannelRealization) -> Path:
    """Persist a complete V2X channel realization in a checksummed directory."""

    root = Path(path).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    save_cir(root / "cir", realization.cir)
    backend_name = realization.cir.metadata.backend
    arrays: dict[str, Any] = {}
    for prefix, instance, names in (
        ("topology", realization.topology, _TOPOLOGY),
        ("conditions", realization.link_conditions, _CONDITIONS),
        ("large_scale", realization.large_scale, _LARGE_SCALE),
        ("state", realization.state, _STATE),
        ("scene", realization.scene, _SCENE),
    ):
        for name in names:
            value = getattr(instance, name)
            if value is not None:
                arrays[f"{prefix}.{name}"] = _write(
                    root, f"{prefix}.{name}", value, backend_name
                )
    diagnostics: dict[str, Any] = {}
    array_backend = get_backend(realization.cir.metadata.backend)
    for name, value in realization.diagnostics.values.items():
        if array_backend.is_array(value):
            diagnostics[name] = {"array": _write(root, f"diagnostic.{name}", value, backend_name)}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            diagnostics[name] = {"value": value}
        else:
            raise TypeError(f"unsupported V2X diagnostic value for {name!r}")
    manifest = {
        "schema_version": 1,
        "kind": "V2XChannelRealization",
        "state_snapshot_index": realization.state.snapshot_index,
        "scene_metadata": realization.scene.metadata.to_document(),
        "arrays": arrays,
        "diagnostics": diagnostics,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def load_v2x_realization(
    path: str | Path, execution: ExecutionContext
) -> V2XChannelRealization:
    """Load a checksummed V2X realization on a requested backend/device."""

    root = Path(path).resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "kind", "state_snapshot_index", "scene_metadata",
        "arrays", "diagnostics",
    }:
        raise ValueError("invalid V2X realization manifest")
    if document["schema_version"] != 1 or document["kind"] != "V2XChannelRealization":
        raise ValueError("unsupported V2X realization schema or kind")
    cir = load_cir(root / "cir", execution)
    arrays = document["arrays"]
    if not isinstance(arrays, dict):
        raise TypeError("V2X realization arrays must be a mapping")
    expected_arrays = {
        *(f"topology.{name}" for name in _TOPOLOGY),
        *(f"conditions.{name}" for name in _CONDITIONS),
        *(f"large_scale.{name}" for name in _LARGE_SCALE),
        *(f"state.{name}" for name in _STATE),
        *(f"scene.{name}" for name in _SCENE),
    }
    if set(arrays) != expected_arrays:
        raise ValueError("V2X realization array set is incomplete or contains unknown fields")

    def values(prefix: str, names: tuple[str, ...]) -> dict[str, Any]:
        return {
            name: _read(root, arrays[f"{prefix}.{name}"], execution)
            for name in names if f"{prefix}.{name}" in arrays
        }

    topology_values = values("topology", _TOPOLOGY)
    topology_values.setdefault("los_mask", None)
    topology_values.setdefault("link_mask", None)
    topology = TopologyBatch(**topology_values, metadata=cir.metadata)
    conditions = V2XLinkConditionBatch(
        **values("conditions", _CONDITIONS), metadata=cir.metadata
    )
    large_scale = V2XLargeScaleComponents(
        **values("large_scale", _LARGE_SCALE), metadata=cir.metadata
    )
    state = V2XChannelState(
        **values("state", _STATE),
        snapshot_index=document["state_snapshot_index"], metadata=cir.metadata,
    )
    scene_metadata_document = document["scene_metadata"]
    if not isinstance(scene_metadata_document, dict):
        raise TypeError("V2X scene metadata must be a mapping")
    scene_metadata_values = dict(scene_metadata_document)
    scene_metadata_values.update(
        backend=execution.backend,
        precision=execution.precision,
        device=execution.device,
    )
    scene_metadata = RealizationMetadata(**scene_metadata_values)
    scene = V2XSceneTrajectory(
        **values("scene", _SCENE), metadata=scene_metadata
    )
    diagnostics = {}
    diagnostic_document = document["diagnostics"]
    if not isinstance(diagnostic_document, dict):
        raise TypeError("V2X diagnostics must be a mapping")
    for name, descriptor in diagnostic_document.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(descriptor, dict)
            or set(descriptor) not in ({"array"}, {"value"})
        ):
            raise ValueError("invalid V2X diagnostic descriptor")
        diagnostics[name] = (
            _read(root, descriptor["array"], execution)
            if "array" in descriptor else descriptor["value"]
        )
    return V2XChannelRealization(
        cir, topology, conditions, large_scale, state, scene,
        V2XRealizationDiagnostics(diagnostics),
    )


__all__ = ["load_v2x_realization", "save_v2x_realization"]
