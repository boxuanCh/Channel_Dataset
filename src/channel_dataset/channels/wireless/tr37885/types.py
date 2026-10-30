"""Immutable array contracts for TR 37.885 urban V2X simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from channel_dataset.backends import RandomStream, get_backend
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import CIRBatch, TopologyBatch
from channel_dataset.core import ExecutionContext, RealizationMetadata

LOS = 0
NLOSV = 1
NLOS = 2
INVALID = 3


def _array(value: Any, metadata: RealizationMetadata, name: str, rank: int, kind: str) -> Any:
    backend = get_backend(metadata.backend)
    if not backend.is_array(value):
        raise TypeError(f"{name} must be a {metadata.backend} array")
    if backend.device(value) != metadata.device:
        raise ValueError(f"{name} device does not match metadata")
    if backend.ndim(value) != rank:
        raise ValueError(f"{name} must have rank {rank}")
    checks = {"float": backend.is_floating, "integer": backend.is_integer, "bool": backend.is_bool}
    if not checks[kind](value):
        raise TypeError(f"{name} must have a {kind} dtype")
    if kind == "float" and not backend.all_finite(value):
        raise ValueError(f"{name} must contain finite values")
    return backend


@dataclass(frozen=True, slots=True)
class V2XSceneSamplingRequest:
    """Controls for generating a time-indexed urban vehicle scene."""

    batch_size: int
    num_mobility_steps: int
    execution: ExecutionContext
    random_stream: RandomStream[Any]
    request_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("batch_size", "num_mobility_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.random_stream.backend_name != self.execution.backend:
            raise ValueError("random stream backend does not match execution")
        if self.random_stream.device != self.execution.device:
            raise ValueError("random stream device does not match execution")
        if self.request_id is not None and not self.request_id:
            raise ValueError("request_id must be non-empty")


@dataclass(frozen=True, slots=True)
class V2XSceneTrajectory:
    """Vehicle state on ``[batch, vehicle, mobility_time, ...]`` axes."""

    positions_m: Any
    velocities_mps: Any
    orientations_rad: Any
    active_mask: Any
    lane_ids: Any
    street_ids: Any
    vehicle_ids: Any
    vehicle_type_codes: Any
    dimensions_m: Any
    antenna_heights_m: Any
    mobility_times_s: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = None
        for name in ("positions_m", "velocities_mps", "orientations_rad"):
            backend = _array(getattr(self, name), self.metadata, name, 4, "float")
            if backend.shape(getattr(self, name))[-1] != 3:
                raise ValueError(f"{name} final axis must have length three")
        assert backend is not None
        shape = backend.shape(self.positions_m)
        for name in ("velocities_mps", "orientations_rad"):
            if backend.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} must match positions_m")
        for name, kind in (("active_mask", "bool"), ("lane_ids", "integer"), ("street_ids", "integer")):
            _array(getattr(self, name), self.metadata, name, 3, kind)
            if backend.shape(getattr(self, name)) != shape[:3]:
                raise ValueError(f"{name} must use [batch, vehicle, mobility_time]")
        for name in ("vehicle_ids", "vehicle_type_codes"):
            _array(getattr(self, name), self.metadata, name, 2, "integer")
            if backend.shape(getattr(self, name)) != shape[:2]:
                raise ValueError(f"{name} must use [batch, vehicle]")
        _array(self.dimensions_m, self.metadata, "dimensions_m", 3, "float")
        if backend.shape(self.dimensions_m) != (*shape[:2], 3):
            raise ValueError("dimensions_m must use [batch, vehicle, length_width_height]")
        _array(self.antenna_heights_m, self.metadata, "antenna_heights_m", 2, "float")
        if backend.shape(self.antenna_heights_m) != shape[:2]:
            raise ValueError("antenna_heights_m must use [batch, vehicle]")
        _array(self.mobility_times_s, self.metadata, "mobility_times_s", 1, "float")
        if backend.shape(self.mobility_times_s) != (shape[2],):
            raise ValueError("mobility_times_s length must match mobility_time")
        if not backend.all_nonnegative(self.mobility_times_s):
            raise ValueError("mobility_times_s must be nonnegative")
        if shape[2] > 1:
            increments = self.mobility_times_s[1:] - self.mobility_times_s[:-1]
            if backend.to_scalar(backend.min(increments)) <= 0.0:
                raise ValueError("mobility_times_s must be strictly increasing")
        if backend.to_scalar(backend.min(self.vehicle_ids)) < 0:
            raise ValueError("vehicle_ids must be nonnegative")
        for batch in range(shape[0]):
            identifiers = [
                int(backend.to_scalar(self.vehicle_ids[batch, vehicle]))
                for vehicle in range(shape[1])
            ]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("vehicle_ids must be unique within each batch")
        type_min = backend.to_scalar(backend.min(self.vehicle_type_codes))
        type_max = backend.to_scalar(backend.max(self.vehicle_type_codes))
        if type_min < 1 or type_max > 3:
            raise ValueError("vehicle_type_codes must contain only 1, 2, or 3")
        if backend.to_scalar(backend.min(self.dimensions_m)) <= 0.0:
            raise ValueError("dimensions_m must be positive")
        if backend.to_scalar(backend.min(self.antenna_heights_m)) <= 0.0:
            raise ValueError("antenna_heights_m must be positive")
        active_lanes = self.lane_ids[self.active_mask]
        active_streets = self.street_ids[self.active_mask]
        if (
            backend.shape(active_lanes)[0] > 0
            and backend.to_scalar(backend.min(active_lanes)) < 0
        ):
            raise ValueError("active vehicles must have nonnegative lane IDs")
        if (
            backend.shape(active_streets)[0] > 0
            and backend.to_scalar(backend.min(active_streets)) < 0
        ):
            raise ValueError("active vehicles must have nonnegative street IDs")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(batch, vehicle, mobility_time)``."""

        return get_backend(self.metadata.backend).shape(self.positions_m)[:3]


@dataclass(frozen=True, slots=True)
class V2XLinkConditionBatch:
    """Per-link condition codes on ``[batch, rx, tx, mobility_time]``."""

    condition_codes: Any
    same_street: Any
    building_obstructed: Any
    los_probability: Any
    blocker_type_codes: Any
    blocker_heights_m: Any
    episode_indices: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = _array(self.condition_codes, self.metadata, "condition_codes", 4, "integer")
        shape = backend.shape(self.condition_codes)
        for name in ("same_street", "building_obstructed"):
            _array(getattr(self, name), self.metadata, name, 4, "bool")
        for name in ("los_probability", "blocker_heights_m"):
            _array(getattr(self, name), self.metadata, name, 4, "float")
        for name in ("blocker_type_codes", "episode_indices"):
            _array(getattr(self, name), self.metadata, name, 4, "integer")
        for name in (
            "same_street", "building_obstructed", "los_probability",
            "blocker_type_codes", "blocker_heights_m", "episode_indices",
        ):
            if backend.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} must match condition_codes")
        minimum = backend.to_scalar(backend.min(self.condition_codes))
        maximum = backend.to_scalar(backend.max(self.condition_codes))
        if minimum < LOS or maximum > INVALID:
            raise ValueError("condition_codes contains an unknown state")
        probability_min = backend.to_scalar(backend.min(self.los_probability))
        probability_max = backend.to_scalar(backend.max(self.los_probability))
        if probability_min < 0.0 or probability_max > 1.0:
            raise ValueError("los_probability must be between zero and one")
        if not backend.all_nonnegative(self.blocker_heights_m):
            raise ValueError("blocker_heights_m must be nonnegative")
        if backend.to_scalar(backend.min(self.episode_indices)) < 0:
            raise ValueError("episode_indices must be nonnegative")


@dataclass(frozen=True, slots=True)
class V2XChannelState:
    """Immutable state carried between consecutive mobility snapshots."""

    condition_codes: Any
    shadow_fading_db: Any
    episode_indices: Any
    snapshot_index: int
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = _array(self.condition_codes, self.metadata, "condition_codes", 3, "integer")
        shape = backend.shape(self.condition_codes)
        _array(self.shadow_fading_db, self.metadata, "shadow_fading_db", 3, "float")
        _array(self.episode_indices, self.metadata, "episode_indices", 3, "integer")
        if backend.shape(self.shadow_fading_db) != shape or backend.shape(self.episode_indices) != shape:
            raise ValueError("channel-state arrays must have equal shape")
        if isinstance(self.snapshot_index, bool) or not isinstance(self.snapshot_index, int):
            raise TypeError("snapshot_index must be an integer")
        if self.snapshot_index < 0:
            raise ValueError("snapshot_index must be nonnegative")


@dataclass(frozen=True, slots=True)
class V2XChannelSamplingRequest:
    """Waveform sampling controls paired with a mobility snapshot."""

    sampling: ChannelSamplingRequest
    scene: V2XSceneTrajectory
    snapshot_index: int
    tx_vehicle_indices: tuple[int, ...]
    rx_vehicle_indices: tuple[int, ...]
    state: V2XChannelState | None = None

    def __post_init__(self) -> None:
        if self.scene.shape[0] != self.sampling.batch_size:
            raise ValueError("scene and sampling batch sizes differ")
        if not 0 <= self.snapshot_index < self.scene.shape[2]:
            raise IndexError("snapshot_index is outside the scene trajectory")
        execution, metadata = self.sampling.execution, self.scene.metadata
        if (execution.backend, execution.precision, execution.device) != (
            metadata.backend, metadata.precision, metadata.device,
        ):
            raise ValueError("scene execution metadata does not match sampling")
        for name in ("tx_vehicle_indices", "rx_vehicle_indices"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} must be nonempty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise TypeError(f"{name} must contain integers")
            if any(value < 0 or value >= self.scene.shape[1] for value in values):
                raise IndexError(f"{name} contains an out-of-range vehicle")
        if set(self.tx_vehicle_indices) & set(self.rx_vehicle_indices):
            raise ValueError("transmit and receive vehicle selections must not contain self-links")
        if self.state is not None:
            expected_shape = (
                self.sampling.batch_size,
                len(self.rx_vehicle_indices),
                len(self.tx_vehicle_indices),
            )
            backend = self.sampling.execution.array_backend
            if backend.shape(self.state.condition_codes) != expected_shape:
                raise ValueError("state link dimensions do not match the selected vehicles")
            if (
                self.state.metadata.backend,
                self.state.metadata.precision,
                self.state.metadata.device,
            ) != (execution.backend, execution.precision, execution.device):
                raise ValueError("state execution metadata does not match sampling")
            if self.state.snapshot_index >= self.snapshot_index:
                raise ValueError("state must precede the requested mobility snapshot")


@dataclass(frozen=True, slots=True)
class V2XLargeScaleComponents:
    """Decomposed link losses in dB on ``[batch, rx, tx]`` axes."""

    path_loss_db: Any
    shadow_fading_db: Any
    vehicle_blockage_db: Any
    total_loss_db: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = None
        shape = None
        for name in ("path_loss_db", "shadow_fading_db", "vehicle_blockage_db", "total_loss_db"):
            backend = _array(getattr(self, name), self.metadata, name, 3, "float")
            if shape is None:
                shape = backend.shape(getattr(self, name))
            elif backend.shape(getattr(self, name)) != shape:
                raise ValueError("large-scale component shapes must match")


@dataclass(frozen=True, slots=True)
class V2XRealizationDiagnostics:
    """Optional immutable intermediate values."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class V2XChannelRealization:
    """Reusable V2X CIR and the scene-derived state that produced it."""

    cir: CIRBatch
    topology: TopologyBatch
    link_conditions: V2XLinkConditionBatch
    large_scale: V2XLargeScaleComponents
    state: V2XChannelState
    scene: V2XSceneTrajectory
    diagnostics: V2XRealizationDiagnostics = field(default_factory=V2XRealizationDiagnostics)

    def __post_init__(self) -> None:
        metadata = self.cir.metadata
        for candidate in (self.topology.metadata, self.link_conditions.metadata, self.large_scale.metadata, self.state.metadata):
            if candidate != metadata:
                raise ValueError("all V2X realization components must share metadata")
        if self.cir.shape[0] != self.topology.shape[0]:
            raise ValueError("CIR and topology batch sizes differ")
        if self.cir.shape[1] != self.topology.shape[1] or self.cir.shape[3] != self.topology.shape[2]:
            raise ValueError("CIR and topology link dimensions differ")
        if self.scene.shape[0] != self.cir.shape[0]:
            raise ValueError("scene and CIR batch dimensions differ")
        if (
            self.scene.metadata.backend,
            self.scene.metadata.precision,
            self.scene.metadata.device,
        ) != (metadata.backend, metadata.precision, metadata.device):
            raise ValueError("scene and CIR execution metadata differ")


__all__ = [
    "INVALID", "LOS", "NLOS", "NLOSV", "V2XChannelRealization",
    "V2XChannelSamplingRequest", "V2XChannelState", "V2XLargeScaleComponents",
    "V2XLinkConditionBatch", "V2XRealizationDiagnostics", "V2XSceneSamplingRequest",
    "V2XSceneTrajectory",
]
