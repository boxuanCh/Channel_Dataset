"""Adapters from V2X trajectories to common wireless topology snapshots."""

from __future__ import annotations

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr37885.types import V2XSceneTrajectory
from channel_dataset.channels.wireless.types import TopologyBatch
from channel_dataset.core import RealizationMetadata


def topology_at_snapshot(
    scene: V2XSceneTrajectory,
    snapshot_index: int,
    tx_vehicle_indices: tuple[int, ...],
    rx_vehicle_indices: tuple[int, ...],
    metadata: RealizationMetadata | None = None,
) -> TopologyBatch:
    """Return a canonical `TopologyBatch` view for selected vehicles."""

    if not 0 <= snapshot_index < scene.shape[2]:
        raise IndexError("snapshot_index is outside the scene trajectory")
    backend = scene.metadata.backend
    if not tx_vehicle_indices or not rx_vehicle_indices:
        raise ValueError("transmitter and receiver selections must be nonempty")
    tx, rx = list(tx_vehicle_indices), list(rx_vehicle_indices)
    tx_active = scene.active_mask[:, tx, snapshot_index]
    rx_active = scene.active_mask[:, rx, snapshot_index]
    link_mask = rx_active[:, :, None] * tx_active[:, None, :]
    for rx_axis, rx_vehicle in enumerate(rx_vehicle_indices):
        for tx_axis, tx_vehicle in enumerate(tx_vehicle_indices):
            if rx_vehicle == tx_vehicle:
                link_mask[:, rx_axis, tx_axis] = False
    array_backend = get_backend(backend)
    return TopologyBatch(
        scene.positions_m[:, tx, snapshot_index, :],
        scene.positions_m[:, rx, snapshot_index, :],
        scene.velocities_mps[:, tx, snapshot_index, :],
        scene.velocities_mps[:, rx, snapshot_index, :],
        scene.orientations_rad[:, tx, snapshot_index, :],
        scene.orientations_rad[:, rx, snapshot_index, :],
        array_backend.full((scene.shape[0], len(rx)), False, dtype="bool", device=scene.metadata.device),
        scene.vehicle_ids[:, tx],
        metadata or scene.metadata,
        None,
        link_mask,
    )


__all__ = ["topology_at_snapshot"]
