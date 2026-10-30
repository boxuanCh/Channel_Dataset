"""Array contracts for 3GPP TR 38.811 non-terrestrial channels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.core import RealizationMetadata

_ENDPOINT_KINDS = {"ground", "satellite", "haps"}


def _validate_array(
    value: Any,
    metadata: RealizationMetadata,
    name: str,
    *,
    rank: int,
    kind: str,
) -> Any:
    backend = get_backend(metadata.backend)
    if not backend.is_array(value):
        raise TypeError(f"{name} must be a {metadata.backend} array")
    if backend.device(value) != metadata.device:
        raise ValueError(f"{name} device does not match metadata")
    if backend.ndim(value) != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if kind == "float" and not backend.is_floating(value):
        raise TypeError(f"{name} must have a real floating dtype")
    if kind == "bool" and not backend.is_bool(value):
        raise TypeError(f"{name} must have a Boolean dtype")
    if kind == "float" and not backend.all_finite(value):
        raise ValueError(f"{name} must contain only finite values")
    return backend


@dataclass(frozen=True, slots=True)
class NTNTopologyBatch:
    """Time-indexed ECEF topology for ground, satellite, and HAPS endpoints.

    Position, velocity, and orientation arrays use axes
    ``[batch, endpoint, trajectory_time, xyz]``. A trajectory time dimension
    of one is broadcast across a sampling request.
    """

    tx_positions_ecef_m: Any
    rx_positions_ecef_m: Any
    tx_velocities_ecef_mps: Any
    rx_velocities_ecef_mps: Any
    tx_orientations_rad: Any
    rx_orientations_rad: Any
    tx_endpoint_kind: str
    rx_endpoint_kind: str
    rx_indoor: Any
    metadata: RealizationMetadata
    los_mask: Any | None = None
    link_mask: Any | None = None

    def __post_init__(self) -> None:
        for name in ("tx_endpoint_kind", "rx_endpoint_kind"):
            if getattr(self, name) not in _ENDPOINT_KINDS:
                raise ValueError(f"{name} must be ground, satellite, or haps")
        if self.tx_endpoint_kind == "ground" and self.rx_endpoint_kind == "ground":
            raise ValueError("an NTN service link requires a satellite or HAPS endpoint")
        if self.tx_endpoint_kind != "ground" and self.rx_endpoint_kind != "ground":
            raise ValueError("an NTN service link requires exactly one ground endpoint side")

        names = (
            "tx_positions_ecef_m", "rx_positions_ecef_m",
            "tx_velocities_ecef_mps", "rx_velocities_ecef_mps",
            "tx_orientations_rad", "rx_orientations_rad",
        )
        backend = None
        for name in names:
            value = getattr(self, name)
            backend = _validate_array(value, self.metadata, name, rank=4, kind="float")
            if backend.shape(value)[-1] != 3:
                raise ValueError(f"{name} final axis must contain three values")
        assert backend is not None
        tx_shape = backend.shape(self.tx_positions_ecef_m)
        rx_shape = backend.shape(self.rx_positions_ecef_m)
        if any(dimension <= 0 for dimension in tx_shape + rx_shape):
            raise ValueError("topology dimensions must be positive")
        for name in ("tx_velocities_ecef_mps", "tx_orientations_rad"):
            if backend.shape(getattr(self, name)) != tx_shape:
                raise ValueError(f"{name} must match transmitter position shape")
        for name in ("rx_velocities_ecef_mps", "rx_orientations_rad"):
            if backend.shape(getattr(self, name)) != rx_shape:
                raise ValueError(f"{name} must match receiver position shape")
        if tx_shape[0] != rx_shape[0]:
            raise ValueError("transmitter and receiver batch dimensions must match")
        if tx_shape[2] != rx_shape[2] and tx_shape[2] != 1 and rx_shape[2] != 1:
            raise ValueError("trajectory lengths must match unless one side is static")

        _validate_array(self.rx_indoor, self.metadata, "rx_indoor", rank=2, kind="bool")
        if backend.shape(self.rx_indoor) != rx_shape[:2]:
            raise ValueError("rx_indoor shape must match receiver endpoints")
        link_shape = (tx_shape[0], rx_shape[1], tx_shape[1])
        for name in ("los_mask", "link_mask"):
            value = getattr(self, name)
            if value is not None:
                _validate_array(value, self.metadata, name, rank=3, kind="bool")
                if backend.shape(value) != link_shape:
                    raise ValueError(f"{name} shape must be {link_shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return ``(batch, rx, tx)`` link dimensions."""

        backend = get_backend(self.metadata.backend)
        tx = backend.shape(self.tx_positions_ecef_m)
        rx = backend.shape(self.rx_positions_ecef_m)
        return tx[0], rx[1], tx[1]

    @property
    def trajectory_length(self) -> int:
        """Return the non-singleton trajectory length, or one for a snapshot."""

        backend = get_backend(self.metadata.backend)
        return max(
            backend.shape(self.tx_positions_ecef_m)[2],
            backend.shape(self.rx_positions_ecef_m)[2],
        )


@dataclass(frozen=True, slots=True)
class NTNChannelSamplingRequest:
    """Sampling controls paired with an Earth-space topology."""

    sampling: ChannelSamplingRequest
    topology: NTNTopologyBatch

    def __post_init__(self) -> None:
        if self.topology.shape[0] != self.sampling.batch_size:
            raise ValueError("topology and request batch sizes differ")
        if self.topology.trajectory_length not in {1, self.sampling.num_time_samples}:
            raise ValueError("topology trajectory length must be one or num_time_samples")
        metadata = self.topology.metadata
        execution = self.sampling.execution
        if (metadata.backend, metadata.device, metadata.precision) != (
            execution.backend, execution.device, execution.precision,
        ):
            raise ValueError("topology execution metadata does not match the request")


@dataclass(frozen=True, slots=True)
class NTNLinkGeometry:
    """Derived Earth-space geometry with link-time scalar axes."""

    slant_range_m: Any
    elevation_rad: Any
    azimuth_rad: Any
    central_angle_rad: Any
    propagation_delay_s: Any
    radial_velocity_mps: Any
    common_doppler_hz: Any
    visibility: Any
    direction_rx_to_tx: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        scalar_names = (
            "slant_range_m", "elevation_rad", "azimuth_rad", "central_angle_rad",
            "propagation_delay_s", "radial_velocity_mps", "common_doppler_hz",
        )
        backend = None
        shape = None
        for name in scalar_names:
            value = getattr(self, name)
            backend = _validate_array(value, self.metadata, name, rank=4, kind="float")
            if shape is None:
                shape = backend.shape(value)
            elif backend.shape(value) != shape:
                raise ValueError("all geometry scalar arrays must have equal shape")
        assert backend is not None and shape is not None
        _validate_array(self.visibility, self.metadata, "visibility", rank=4, kind="bool")
        if backend.shape(self.visibility) != shape:
            raise ValueError("visibility shape must match geometry scalar arrays")
        _validate_array(
            self.direction_rx_to_tx, self.metadata, "direction_rx_to_tx", rank=5, kind="float"
        )
        if backend.shape(self.direction_rx_to_tx) != (*shape, 3):
            raise ValueError("direction_rx_to_tx must append an xyz axis")
        if not backend.all_nonnegative(self.slant_range_m):
            raise ValueError("slant_range_m must be nonnegative")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return ``(batch, rx, tx, time)``."""

        return get_backend(self.metadata.backend).shape(self.slant_range_m)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class NTNLargeScaleComponents:
    """Decomposed attenuation in dB on ``[batch, rx, tx, time]`` axes."""

    free_space_db: Any
    clutter_db: Any
    shadow_fading_db: Any
    building_entry_db: Any
    atmospheric_gas_db: Any
    rain_db: Any
    cloud_db: Any
    scintillation_db: Any
    total_db: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        shape = None
        backend = None
        for name in (
            "free_space_db", "clutter_db", "shadow_fading_db", "building_entry_db",
            "atmospheric_gas_db", "rain_db", "cloud_db", "scintillation_db", "total_db",
        ):
            value = getattr(self, name)
            backend = _validate_array(value, self.metadata, name, rank=4, kind="float")
            if shape is None:
                shape = backend.shape(value)
            elif backend.shape(value) != shape:
                raise ValueError("all large-scale arrays must have equal shape")


@dataclass(frozen=True, slots=True)
class NTNRealizationDiagnostics:
    """Optional immutable intermediate values that do not affect sampling."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class NTNChannelRealization:
    """Reusable NTN CIR plus geometry, loss decomposition, and LOS state."""

    cir: CIRBatch
    geometry: NTNLinkGeometry
    large_scale: NTNLargeScaleComponents
    los_mask: Any
    diagnostics: NTNRealizationDiagnostics = field(default_factory=NTNRealizationDiagnostics)

    def __post_init__(self) -> None:
        metadata = self.cir.metadata
        if self.geometry.metadata != metadata or self.large_scale.metadata != metadata:
            raise ValueError("all NTN realization components must share metadata")
        backend = _validate_array(self.los_mask, metadata, "los_mask", rank=3, kind="bool")
        if backend.shape(self.los_mask) != self.geometry.shape[:3]:
            raise ValueError("los_mask must match geometry link dimensions")
        if self.cir.shape[0] != self.geometry.shape[0]:
            raise ValueError("CIR and geometry batch sizes differ")
        if self.cir.shape[1] != self.geometry.shape[1] or self.cir.shape[3] != self.geometry.shape[2]:
            raise ValueError("CIR and geometry link dimensions differ")
        if self.cir.shape[-1] != self.geometry.shape[-1]:
            raise ValueError("CIR and geometry time dimensions differ")


__all__ = [
    "NTNChannelRealization", "NTNChannelSamplingRequest", "NTNLargeScaleComponents",
    "NTNLinkGeometry", "NTNRealizationDiagnostics", "NTNTopologyBatch",
]
