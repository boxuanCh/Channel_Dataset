"""Array contracts for geometric CDL mobility and dynamic CIRs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import CIRBatch, ClusterBatch, RayBatch, TopologyBatch
from channel_dataset.core import RealizationMetadata


def _array(value: Any, metadata: RealizationMetadata, name: str, rank: int, kind: str) -> Any:
    backend = get_backend(metadata.backend)
    if not backend.is_array(value):
        raise TypeError(f"{name} must be a {metadata.backend} array")
    if backend.device(value) != metadata.device:
        raise ValueError(f"{name} device does not match metadata")
    if backend.ndim(value) != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if kind == "float" and not backend.is_floating(value):
        raise TypeError(f"{name} must have a floating dtype")
    if kind == "complex" and not backend.is_complex(value):
        raise TypeError(f"{name} must have a complex dtype")
    if kind == "bool" and not backend.is_bool(value):
        raise TypeError(f"{name} must have a Boolean dtype")
    if kind == "integer" and not backend.is_integer(value):
        raise TypeError(f"{name} must have an integer dtype")
    if kind in {"float", "complex"}:
        prefix = "complex" if kind == "complex" else "float"
        bits = (
            ("64" if metadata.precision == "single" else "128")
            if kind == "complex"
            else ("32" if metadata.precision == "single" else "64")
        )
        if backend.dtype(value) != backend.resolve_dtype(prefix + bits):
            raise TypeError(f"{name} dtype does not match realization precision")
    if kind in {"float", "complex"} and not backend.all_finite(value):
        raise ValueError(f"{name} must contain only finite values")
    return backend


@dataclass(frozen=True, slots=True)
class TopologyTrajectoryBatch:
    """Time-indexed terrestrial topology on time-last endpoint axes."""

    times_s: Any
    tx_positions_m: Any
    rx_positions_m: Any
    tx_velocities_mps: Any
    rx_velocities_mps: Any
    tx_orientations_rad: Any
    rx_orientations_rad: Any
    tx_endpoint_ids: Any
    rx_endpoint_ids: Any
    rx_indoor: Any
    tx_site_ids: Any
    metadata: RealizationMetadata
    link_mask: Any | None = None
    los_mask: Any | None = None

    def __post_init__(self) -> None:
        backend = _array(self.times_s, self.metadata, "times_s", 1, "float")
        time_count = backend.shape(self.times_s)[0]
        if time_count <= 0 or not backend.all_nonnegative(self.times_s):
            raise ValueError("times_s must be nonempty and nonnegative")
        if time_count > 1 and backend.to_scalar(backend.min(self.times_s[1:] - self.times_s[:-1])) <= 0:
            raise ValueError("times_s must be strictly increasing")
        for name in (
            "tx_positions_m", "rx_positions_m", "tx_velocities_mps", "rx_velocities_mps",
            "tx_orientations_rad", "rx_orientations_rad",
        ):
            _array(getattr(self, name), self.metadata, name, 4, "float")
            if backend.shape(getattr(self, name))[-1] != 3:
                raise ValueError(f"{name} final axis must have length three")
        tx_shape, rx_shape = backend.shape(self.tx_positions_m), backend.shape(self.rx_positions_m)
        if tx_shape[0] != rx_shape[0] or tx_shape[2] != time_count or rx_shape[2] != time_count:
            raise ValueError("trajectory batch and time dimensions must match")
        for name in ("tx_velocities_mps", "tx_orientations_rad"):
            if backend.shape(getattr(self, name)) != tx_shape:
                raise ValueError(f"{name} must match tx_positions_m")
        for name in ("rx_velocities_mps", "rx_orientations_rad"):
            if backend.shape(getattr(self, name)) != rx_shape:
                raise ValueError(f"{name} must match rx_positions_m")
        for name, shape in (("tx_endpoint_ids", tx_shape[:2]), ("rx_endpoint_ids", rx_shape[:2]), ("tx_site_ids", tx_shape[:2])):
            _array(getattr(self, name), self.metadata, name, 2, "integer")
            if backend.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} has incompatible shape")
        _array(self.rx_indoor, self.metadata, "rx_indoor", 3, "bool")
        if backend.shape(self.rx_indoor) != rx_shape[:3]:
            raise ValueError("rx_indoor must use [batch, rx, time]")
        link_shape = (tx_shape[0], rx_shape[1], tx_shape[1], time_count)
        for name in ("link_mask", "los_mask"):
            value = getattr(self, name)
            if value is not None:
                _array(value, self.metadata, name, 4, "bool")
                if backend.shape(value) != link_shape:
                    raise ValueError(f"{name} must use [batch, rx, tx, time]")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        backend = get_backend(self.metadata.backend)
        tx, rx = backend.shape(self.tx_positions_m), backend.shape(self.rx_positions_m)
        return tx[0], rx[1], tx[1], tx[2]

    def snapshot(self, time_index: int, *, metadata: RealizationMetadata | None = None) -> TopologyBatch:
        """Return a common static topology view for one trajectory sample."""

        if isinstance(time_index, bool) or not isinstance(time_index, int):
            raise TypeError("time_index must be an integer")
        if not 0 <= time_index < self.shape[-1]:
            raise IndexError("time_index is outside the trajectory")
        return TopologyBatch(
            self.tx_positions_m[:, :, time_index, :], self.rx_positions_m[:, :, time_index, :],
            self.tx_velocities_mps[:, :, time_index, :], self.rx_velocities_mps[:, :, time_index, :],
            self.tx_orientations_rad[:, :, time_index, :], self.rx_orientations_rad[:, :, time_index, :],
            self.rx_indoor[:, :, time_index], self.tx_site_ids, metadata or self.metadata,
            None if self.los_mask is None else self.los_mask[..., time_index],
            None if self.link_mask is None else self.link_mask[..., time_index],
        )


@dataclass(frozen=True, slots=True)
class ScenarioTimelineBatch:
    """Scenario codes and optional propagation-state overrides per link/time."""

    scenario_codes: Any
    profiles: Mapping[int, str]
    metadata: RealizationMetadata
    propagation_state_codes: Any | None = None

    def __post_init__(self) -> None:
        backend = _array(self.scenario_codes, self.metadata, "scenario_codes", 4, "integer")
        profiles = dict(self.profiles)
        if not profiles or any(not isinstance(key, int) or not isinstance(value, str) or not value for key, value in profiles.items()):
            raise ValueError("profiles must map integer codes to nonempty identifiers")
        used = {
            int(backend.to_scalar(value))
            for value in self.scenario_codes.reshape(-1)
        }
        if not used <= set(profiles):
            raise ValueError("scenario_codes contains a code missing from profiles")
        object.__setattr__(self, "profiles", MappingProxyType(profiles))
        if self.propagation_state_codes is not None:
            _array(self.propagation_state_codes, self.metadata, "propagation_state_codes", 4, "integer")
            if backend.shape(self.propagation_state_codes) != backend.shape(self.scenario_codes):
                raise ValueError("propagation_state_codes must match scenario_codes")

    @property
    def shape(self) -> tuple[int, ...]:
        return get_backend(self.metadata.backend).shape(self.scenario_codes)


@dataclass(frozen=True, slots=True)
class DynamicCIRBatch:
    """CIR whose path delays and validity may vary with coefficient time."""

    coefficients: Any
    delays_s: Any
    metadata: RealizationMetadata
    path_mask: Any | None = None

    def __post_init__(self) -> None:
        backend = _array(self.coefficients, self.metadata, "coefficients", 7, "complex")
        _array(self.delays_s, self.metadata, "delays_s", 5, "float")
        coefficient_shape = backend.shape(self.coefficients)
        expected = (coefficient_shape[0], coefficient_shape[1], coefficient_shape[3], coefficient_shape[5], coefficient_shape[6])
        if backend.shape(self.delays_s) != expected:
            raise ValueError(f"delays_s shape must be {expected}")
        if not backend.all_nonnegative(self.delays_s):
            raise ValueError("delays_s must be nonnegative")
        if self.path_mask is not None:
            _array(self.path_mask, self.metadata, "path_mask", 5, "bool")
            if backend.shape(self.path_mask) != expected:
                raise ValueError("path_mask shape must match delays_s")

    @property
    def shape(self) -> tuple[int, ...]:
        return get_backend(self.metadata.backend).shape(self.coefficients)

    def select_link(self, batch_index: int, rx: int, tx: int) -> DynamicCIRBatch:
        b, r, _, t, *_ = self.shape
        for value, size, name in ((batch_index, b, "batch_index"), (rx, r, "rx"), (tx, t, "tx")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < size:
                raise IndexError(f"{name} is out of range")
        bs, rs, ts = slice(batch_index, batch_index + 1), slice(rx, rx + 1), slice(tx, tx + 1)
        mask = None if self.path_mask is None else self.path_mask[bs, rs, ts, :, :]
        return DynamicCIRBatch(
            self.coefficients[bs, rs, :, ts, :, :, :],
            self.delays_s[bs, rs, ts, :, :], self.metadata, mask,
        )

    def at_time(self, time_index: int) -> CIRBatch:
        if isinstance(time_index, bool) or not isinstance(time_index, int):
            raise TypeError("time_index must be an integer")
        if not 0 <= time_index < self.shape[-1]:
            raise IndexError("time_index is out of range")
        mask = None if self.path_mask is None else self.path_mask[..., time_index]
        return CIRBatch(
            self.coefficients[..., time_index : time_index + 1],
            self.delays_s[..., time_index], self.metadata, mask,
        )


@dataclass(frozen=True, slots=True)
class LargeScaleTrajectoryBatch:
    """Large-scale variables on ``[batch, rx, tx, time]`` axes."""

    delay_spread_s: Any
    azimuth_spread_arrival_rad: Any
    azimuth_spread_departure_rad: Any
    zenith_spread_arrival_rad: Any
    zenith_spread_departure_rad: Any
    shadow_fading_db: Any
    k_factor_db: Any
    xpr_db: Any
    path_loss_db: Any
    los_mask: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = None
        shape = None
        for name in (
            "delay_spread_s", "azimuth_spread_arrival_rad", "azimuth_spread_departure_rad",
            "zenith_spread_arrival_rad", "zenith_spread_departure_rad", "shadow_fading_db",
            "k_factor_db", "xpr_db", "path_loss_db",
        ):
            backend = _array(getattr(self, name), self.metadata, name, 4, "float")
            shape = shape or backend.shape(getattr(self, name))
            if backend.shape(getattr(self, name)) != shape:
                raise ValueError("large-scale trajectory shapes must match")
        assert backend is not None and shape is not None
        _array(self.los_mask, self.metadata, "los_mask", 4, "bool")
        if backend.shape(self.los_mask) != shape:
            raise ValueError("los_mask must match large-scale trajectories")


@dataclass(frozen=True, slots=True)
class RayTrajectoryBatch:
    """Persistent ray identities and time-varying geometric quantities."""

    absolute_path_length_m: Any
    absolute_delay_s: Any
    excess_delay_s: Any
    azimuth_arrival_rad: Any
    zenith_arrival_rad: Any
    azimuth_departure_rad: Any
    zenith_departure_rad: Any
    cluster_power_linear: Any
    ray_power_fraction: Any
    amplitude_weight: Any
    xpr_linear: Any
    blockage_loss_db: Any
    ray_mask: Any
    cluster_ids: Any
    ray_ids: Any
    generation_ids: Any
    component_type_codes: Any
    initial_phase_rad: Any
    polarization_phases_rad: Any
    fbs_positions_m: Any
    lbs_positions_m: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = _array(self.absolute_path_length_m, self.metadata, "absolute_path_length_m", 6, "float")
        shape = backend.shape(self.absolute_path_length_m)
        for name in (
            "absolute_delay_s", "excess_delay_s", "azimuth_arrival_rad", "zenith_arrival_rad",
            "azimuth_departure_rad", "zenith_departure_rad", "ray_power_fraction",
            "amplitude_weight", "xpr_linear", "blockage_loss_db",
        ):
            _array(getattr(self, name), self.metadata, name, 6, "float")
            if backend.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} must match ray trajectory shape")
        _array(self.ray_mask, self.metadata, "ray_mask", 6, "bool")
        if backend.shape(self.ray_mask) != shape:
            raise ValueError("ray_mask must match ray trajectory shape")
        _array(self.cluster_power_linear, self.metadata, "cluster_power_linear", 5, "float")
        if backend.shape(self.cluster_power_linear) != shape[:4] + shape[-1:]:
            raise ValueError("cluster_power_linear must omit the ray axis")
        for name in ("cluster_ids", "generation_ids", "component_type_codes"):
            _array(getattr(self, name), self.metadata, name, 4, "integer")
            if backend.shape(getattr(self, name)) != shape[:4]:
                raise ValueError(f"{name} must use cluster axes")
        _array(self.ray_ids, self.metadata, "ray_ids", 5, "integer")
        if backend.shape(self.ray_ids) != shape[:5]:
            raise ValueError("ray_ids must use ray identity axes")
        _array(self.initial_phase_rad, self.metadata, "initial_phase_rad", 5, "float")
        if backend.shape(self.initial_phase_rad) != shape[:5]:
            raise ValueError("initial_phase_rad must use ray identity axes")
        _array(self.polarization_phases_rad, self.metadata, "polarization_phases_rad", 6, "float")
        if backend.shape(self.polarization_phases_rad) != shape[:5] + (4,):
            raise ValueError("polarization phases must append four components")
        for name in ("fbs_positions_m", "lbs_positions_m"):
            _array(getattr(self, name), self.metadata, name, 6, "float")
            if backend.shape(getattr(self, name)) != shape[:5] + (3,):
                raise ValueError(f"{name} must append xyz to ray identity axes")
        if not backend.all_nonnegative(self.absolute_path_length_m) or not backend.all_nonnegative(self.excess_delay_s):
            raise ValueError("path lengths and delays must be nonnegative")

    @property
    def shape(self) -> tuple[int, ...]:
        return get_backend(self.metadata.backend).shape(self.absolute_path_length_m)

    def snapshot(self, time_index: int) -> RayBatch:
        """Return a static diagnostic ``RayBatch`` at one time index."""

        if not 0 <= time_index < self.shape[-1]:
            raise IndexError("time_index is outside the ray trajectory")
        backend = get_backend(self.metadata.backend)
        mask = self.ray_mask[..., time_index]
        cluster_mask = backend.max(backend.astype(mask, "int64"), axis=-1) > 0
        ray_fraction = self.ray_power_fraction[..., time_index]
        ray_count = ray_fraction.shape[-1]
        clusters = ClusterBatch(
            self.excess_delay_s[..., 0, time_index], self.cluster_power_linear[..., time_index],
            self.azimuth_arrival_rad[..., 0, time_index], self.zenith_arrival_rad[..., 0, time_index],
            self.azimuth_departure_rad[..., 0, time_index], self.zenith_departure_rad[..., 0, time_index],
            self.metadata, cluster_mask,
        )
        # RayBatch has cluster-level power and equal-ray coefficient semantics;
        # the exact nonuniform fractions remain available on this trajectory.
        _ = ray_count
        return RayBatch(
            clusters,
            self.azimuth_arrival_rad[..., time_index], self.zenith_arrival_rad[..., time_index],
            self.azimuth_departure_rad[..., time_index], self.zenith_departure_rad[..., time_index],
            self.xpr_linear[..., time_index], self.initial_phase_rad, ray_mask=mask,
            blockage_loss_db=self.blockage_loss_db[..., time_index],
        )


@dataclass(frozen=True, slots=True)
class CarrierClusterState:
    """Persistent scatterer state for one carrier."""

    carrier_frequency_hz: float
    fbs_positions_m: Any
    lbs_positions_m: Any
    cluster_ids: Any
    ray_ids: Any
    generation_ids: Any
    component_type_codes: Any
    initial_phase_rad: Any
    polarization_phases_rad: Any
    xpr_linear: Any
    cluster_power_linear: Any
    ray_power_fraction: Any
    ray_mask: Any
    solver_residual_m: Any
    fallback_mask: Any


@dataclass(frozen=True, slots=True)
class GeometricCDLState:
    """Immutable resume state for a geometric CDL realization."""

    field_directions_per_m: Any
    field_phases_rad: Any
    segment_anchor_tx_positions_m: Any
    segment_anchor_rx_positions_m: Any
    carrier_states: tuple[CarrierClusterState, ...]
    tx_endpoint_ids: Any
    rx_endpoint_ids: Any
    segment_ids: Any
    last_scenario_codes: Any
    last_los_mask: Any
    final_path_lengths_m: tuple[Any, ...]
    last_time_s: float
    last_snapshot_index: int
    model_digest: str
    stream_id: str
    root_seed: int
    metadata: RealizationMetadata


@dataclass(frozen=True, slots=True)
class GeometricCDLSamplingRequest:
    """Common sampling controls paired with topology and scenario timelines."""

    sampling: ChannelSamplingRequest
    trajectory: TopologyTrajectoryBatch
    scenarios: ScenarioTimelineBatch
    prior_state: GeometricCDLState | None = None
    explicit_segment_boundaries: tuple[int, ...] | None = None
    diagnostics: bool = False

    def __post_init__(self) -> None:
        if self.trajectory.shape[0] != self.sampling.batch_size:
            raise ValueError("trajectory and sampling batch sizes differ")
        if self.trajectory.shape[-1] != self.sampling.num_time_samples:
            raise ValueError("trajectory time length must equal num_time_samples")
        if self.scenarios.shape != self.trajectory.shape:
            raise ValueError("scenario timeline must match trajectory link-time shape")
        execution = self.sampling.execution
        for metadata in (self.trajectory.metadata, self.scenarios.metadata):
            if (metadata.backend, metadata.precision, metadata.device) != (
                execution.backend, execution.precision, execution.device,
            ):
                raise ValueError("request arrays do not match sampling execution")
        if self.explicit_segment_boundaries is not None:
            boundaries = self.explicit_segment_boundaries
            if tuple(sorted(set(boundaries))) != boundaries:
                raise ValueError("explicit segment boundaries must be sorted and unique")
            if any(value <= 0 or value >= self.sampling.num_time_samples for value in boundaries):
                raise ValueError("explicit segment boundary is outside the trajectory interior")
        if self.prior_state is not None:
            backend = self.sampling.execution.array_backend
            first_time = float(backend.to_scalar(self.trajectory.times_s[0]))
            expected_time = (
                self.prior_state.last_time_s
                + 1.0 / self.sampling.sampling_frequency_hz
            )
            tolerance = max(1.0e-12, abs(expected_time) * 1.0e-9)
            if abs(first_time - expected_time) > tolerance:
                raise ValueError("prior state is not contiguous with the trajectory")


@dataclass(frozen=True, slots=True)
class CarrierGeometricRealization:
    carrier_frequency_hz: float
    cir: DynamicCIRBatch
    rays: RayTrajectoryBatch
    large_scale: LargeScaleTrajectoryBatch
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True, slots=True)
class GeometricCDLRealization:
    carriers: tuple[CarrierGeometricRealization, ...]
    state: GeometricCDLState
    metadata: RealizationMetadata
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.carriers:
            raise ValueError("at least one carrier realization is required")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


__all__ = [
    "CarrierClusterState", "CarrierGeometricRealization", "DynamicCIRBatch",
    "GeometricCDLRealization", "GeometricCDLSamplingRequest", "GeometricCDLState",
    "LargeScaleTrajectoryBatch", "RayTrajectoryBatch", "ScenarioTimelineBatch",
    "TopologyTrajectoryBatch",
]
