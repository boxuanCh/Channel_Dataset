"""Checksummed persistence for geometric CDL realizations and resume state."""

from __future__ import annotations

import json
from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from channel_dataset.core import ExecutionContext, RealizationMetadata

from .types import (
    CarrierClusterState,
    CarrierGeometricRealization,
    DynamicCIRBatch,
    GeometricCDLRealization,
    GeometricCDLState,
    LargeScaleTrajectoryBatch,
    RayTrajectoryBatch,
)

_ARRAY_TYPES = (RayTrajectoryBatch, LargeScaleTrajectoryBatch, CarrierClusterState)


def _numpy(value: Any, backend: str) -> np.ndarray:
    return np.asarray(value) if backend == "numpy" else value.detach().cpu().numpy()


def _write_array(root: Path, name: str, value: Any, backend: str) -> dict[str, Any]:
    filename = f"{name}.npy"
    target = root / filename
    array = _numpy(value, backend)
    np.save(target, array, allow_pickle=False)
    return {
        "file": filename,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": sha256(target.read_bytes()).hexdigest(),
    }


def _read_array(root: Path, description: Any, execution: ExecutionContext) -> Any:
    if not isinstance(description, dict) or set(description) != {"file", "shape", "dtype", "sha256"}:
        raise ValueError("invalid geometric array descriptor")
    source = root / description["file"]
    content = source.read_bytes()
    if sha256(content).hexdigest() != description["sha256"]:
        raise ValueError(f"array checksum mismatch: {description['file']}")
    array = np.load(source, allow_pickle=False)
    if list(array.shape) != description["shape"] or array.dtype.str != description["dtype"]:
        raise ValueError("array shape or dtype does not match the manifest")
    return execution.array_backend.array(array, dtype=array.dtype.name, device=execution.device)


def _array_field_names(instance: Any) -> tuple[str, ...]:
    return tuple(
        item.name for item in fields(instance)
        if item.name not in {"metadata", "carrier_frequency_hz"}
        and not isinstance(getattr(instance, item.name), (str, int, float, bool, tuple, dict))
    )


def save_geometric_cdl_realization(
    path: str | Path, realization: GeometricCDLRealization
) -> Path:
    """Save a complete dynamic realization and resumable geometric state."""

    root = Path(path).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    backend = realization.metadata.backend
    arrays: dict[str, dict[str, Any]] = {}
    carriers = []
    for index, carrier in enumerate(realization.carriers):
        prefix = f"carrier_{index}"
        arrays[f"{prefix}.cir.coefficients"] = _write_array(
            root, f"{prefix}.cir.coefficients", carrier.cir.coefficients, backend
        )
        arrays[f"{prefix}.cir.delays_s"] = _write_array(
            root, f"{prefix}.cir.delays_s", carrier.cir.delays_s, backend
        )
        if carrier.cir.path_mask is not None:
            arrays[f"{prefix}.cir.path_mask"] = _write_array(
                root, f"{prefix}.cir.path_mask", carrier.cir.path_mask, backend
            )
        ray_names = _array_field_names(carrier.rays)
        lsp_names = _array_field_names(carrier.large_scale)
        for name in ray_names:
            arrays[f"{prefix}.rays.{name}"] = _write_array(
                root, f"{prefix}.rays.{name}", getattr(carrier.rays, name), backend
            )
        for name in lsp_names:
            arrays[f"{prefix}.large_scale.{name}"] = _write_array(
                root, f"{prefix}.large_scale.{name}", getattr(carrier.large_scale, name), backend
            )
        carriers.append({
            "carrier_frequency_hz": carrier.carrier_frequency_hz,
            "ray_fields": list(ray_names),
            "large_scale_fields": list(lsp_names),
            "has_path_mask": carrier.cir.path_mask is not None,
        })
    state = realization.state
    for name in (
        "field_directions_per_m", "field_phases_rad", "segment_anchor_tx_positions_m",
        "segment_anchor_rx_positions_m", "tx_endpoint_ids", "rx_endpoint_ids", "segment_ids",
        "last_scenario_codes", "last_los_mask",
    ):
        arrays[f"state.{name}"] = _write_array(root, f"state.{name}", getattr(state, name), backend)
    state_carriers = []
    for index, carrier in enumerate(state.carrier_states):
        names = _array_field_names(carrier)
        for name in names:
            arrays[f"state.carrier_{index}.{name}"] = _write_array(
                root, f"state.carrier_{index}.{name}", getattr(carrier, name), backend
            )
        arrays[f"state.carrier_{index}.final_path_lengths_m"] = _write_array(
            root, f"state.carrier_{index}.final_path_lengths_m",
            state.final_path_lengths_m[index], backend,
        )
        state_carriers.append({
            "carrier_frequency_hz": carrier.carrier_frequency_hz,
            "fields": list(names),
        })
    manifest = {
        "schema_version": 1,
        "kind": "GeometricCDLRealization",
        "metadata": realization.metadata.to_document(),
        "arrays": arrays,
        "carriers": carriers,
        "state_carriers": state_carriers,
        "state": {
            "last_time_s": state.last_time_s,
            "last_snapshot_index": state.last_snapshot_index,
            "model_digest": state.model_digest,
            "stream_id": state.stream_id,
            "root_seed": state.root_seed,
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def load_geometric_cdl_realization(
    path: str | Path, execution: ExecutionContext
) -> GeometricCDLRealization:
    """Load a saved realization onto NumPy or PyTorch execution."""

    root = Path(path).resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {"schema_version", "kind", "metadata", "arrays", "carriers", "state_carriers", "state"}
    if not isinstance(document, dict) or set(document) != expected:
        raise ValueError("invalid geometric CDL manifest")
    if document["schema_version"] != 1 or document["kind"] != "GeometricCDLRealization":
        raise ValueError("unsupported geometric CDL persistence schema")
    metadata_values = dict(document["metadata"])
    metadata_values.update(
        backend=execution.backend, precision=execution.precision, device=execution.device
    )
    metadata = RealizationMetadata(**metadata_values)
    arrays = document["arrays"]
    def value(name: str) -> Any:
        try:
            return _read_array(root, arrays[name], execution)
        except KeyError as error:
            raise ValueError(f"manifest is missing array {name}") from error
    carrier_results = []
    for index, description in enumerate(document["carriers"]):
        prefix = f"carrier_{index}"
        mask = value(f"{prefix}.cir.path_mask") if description["has_path_mask"] else None
        cir = DynamicCIRBatch(
            value(f"{prefix}.cir.coefficients"), value(f"{prefix}.cir.delays_s"), metadata, mask
        )
        ray_values = {name: value(f"{prefix}.rays.{name}") for name in description["ray_fields"]}
        lsp_values = {
            name: value(f"{prefix}.large_scale.{name}")
            for name in description["large_scale_fields"]
        }
        carrier_results.append(CarrierGeometricRealization(
            float(description["carrier_frequency_hz"]), cir,
            RayTrajectoryBatch(**ray_values, metadata=metadata),
            LargeScaleTrajectoryBatch(**lsp_values, metadata=metadata),
        ))
    state_carriers = []
    final_paths = []
    for index, description in enumerate(document["state_carriers"]):
        prefix = f"state.carrier_{index}"
        values = {name: value(f"{prefix}.{name}") for name in description["fields"]}
        state_carriers.append(CarrierClusterState(
            float(description["carrier_frequency_hz"]), **values
        ))
        final_paths.append(value(f"{prefix}.final_path_lengths_m"))
    state_values = document["state"]
    state = GeometricCDLState(
        value("state.field_directions_per_m"), value("state.field_phases_rad"),
        value("state.segment_anchor_tx_positions_m"), value("state.segment_anchor_rx_positions_m"),
        tuple(state_carriers), value("state.tx_endpoint_ids"), value("state.rx_endpoint_ids"),
        value("state.segment_ids"), value("state.last_scenario_codes"),
        value("state.last_los_mask"), tuple(final_paths),
        float(state_values["last_time_s"]),
        int(state_values["last_snapshot_index"]), str(state_values["model_digest"]),
        str(state_values["stream_id"]), int(state_values["root_seed"]), metadata,
    )
    return GeometricCDLRealization(tuple(carrier_results), state, metadata)


__all__ = ["load_geometric_cdl_realization", "save_geometric_cdl_realization"]
