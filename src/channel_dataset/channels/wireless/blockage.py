"""Snapshot blockage models and spatial-consistency fields."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, pi
from types import MappingProxyType
from typing import Any

from channel_dataset.backends import RandomStream, backend_name_for_array, get_backend
from channel_dataset.channels.wireless.rays import ExponentialSpatialField
from channel_dataset.channels.wireless.types import RayBatch, TopologyBatch
from channel_dataset.config import load_resource_document

_RESOURCE = "standards/wireless/tr_38_901/v16_1/blockage.yaml"


@dataclass(frozen=True, slots=True)
class WirelessCapabilityMetadata:
    """Declared support boundary for system-level wireless features."""

    standard: str = "3GPP TR 38.901 V16.1.0"
    blockage: tuple[str, ...] = (
        "self-blocking portrait/landscape angular regions",
        "snapshot non-self-blocking Model A angular regions",
        "snapshot screen-based Model B",
    )
    spatial_consistency: tuple[str, ...] = (
        "snapshot LoS field",
        "snapshot LSP Gaussian fields",
        "snapshot small-scale Gaussian fields",
    )
    omissions: tuple[str, ...] = (
        "stateful mobility evolution",
        "temporal blocker evolution",
        "dynamic birth/death of blockers",
    )


CAPABILITIES = WirelessCapabilityMetadata()


def load_blockage_parameters() -> Mapping[str, Any]:
    """Load the strict versioned blockage parameter resource."""

    document = load_resource_document(_RESOURCE)
    expected = {
        "schema_version", "standard", "standard_version", "section",
        "self_blocking", "model_a", "model_b", "source_url",
        "transcription_date", "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1:
        raise ValueError("invalid blockage resource")
    return MappingProxyType(dict(document))


def _wrap(value: Any, backend: Any) -> Any:
    return backend.atan2(backend.sin(value), backend.cos(value))


def _angular_region(
    rays: RayBatch,
    azimuth_center: Any,
    azimuth_half_width: Any,
    zenith_center: Any,
    zenith_half_width: Any,
) -> Any:
    backend = get_backend(rays.metadata.backend)
    azimuth = backend.abs(_wrap(rays.azimuth_arrival_rad - azimuth_center, backend))
    zenith = backend.abs(rays.zenith_arrival_rad - zenith_center)
    return (azimuth <= azimuth_half_width + 1.0e-12) * (
        zenith <= zenith_half_width + 1.0e-12
    )


@dataclass(frozen=True, slots=True)
class SelfBlockingModel:
    """Hand/body self-blocking angular region fixed to receiver orientation.

    Parameters
    ----------
    mode : {"portrait", "landscape"}, default="portrait"
        Standard user-device orientation preset.
    """

    mode: str = "portrait"

    def __post_init__(self) -> None:
        if self.mode not in {"portrait", "landscape"}:
            raise ValueError("mode must be 'portrait' or 'landscape'")

    def loss_db(self, rays: RayBatch, topology: TopologyBatch) -> Any:
        """Return nonnegative per-ray attenuation in decibels."""

        if rays.shape[:3] != topology.shape:
            raise ValueError("rays and topology must have matching link axes")
        backend = get_backend(rays.metadata.backend)
        values = load_blockage_parameters()["self_blocking"][self.mode]
        scale = pi / 180.0
        bearing = topology.rx_orientations_rad[..., 0][:, :, None, None, None]
        inside = _angular_region(
            rays,
            bearing + values["azimuth_center_degrees"] * scale,
            values["azimuth_half_width_degrees"] * scale,
            values["zenith_center_degrees"] * scale,
            values["zenith_half_width_degrees"] * scale,
        )
        return backend.astype(inside, rays.metadata.precision.replace("single", "float32").replace("double", "float64")) * values["attenuation_db"]


@dataclass(frozen=True, slots=True)
class NonSelfBlockingModelA:
    """Snapshot Model A with random receiver-local angular blockers."""

    mode: str = "portrait"

    def __post_init__(self) -> None:
        if self.mode not in {"portrait", "landscape"}:
            raise ValueError("mode must be 'portrait' or 'landscape'")

    def loss_db(
        self,
        rays: RayBatch,
        topology: TopologyBatch,
        random_stream: RandomStream[Any],
    ) -> Any:
        """Sample blockers and return their accumulated per-ray loss."""

        backend = get_backend(rays.metadata.backend)
        if random_stream.backend_name != rays.metadata.backend:
            raise ValueError("random stream backend must match rays")
        values = load_blockage_parameters()["model_a"][self.mode]
        dtype = "float32" if rays.metadata.precision == "single" else "float64"
        result = backend.zeros(rays.shape, dtype=dtype, device=rays.metadata.device)
        centers = random_stream.fork("wireless.blockage.model_a.centers").uniform(
            (*topology.shape[:2], values["num_blockers"]), -pi, pi, dtype=dtype
        )
        scale = pi / 180.0
        bearing = topology.rx_orientations_rad[..., 0]
        for index in range(values["num_blockers"]):
            center = (bearing + centers[..., index])[:, :, None, None, None]
            inside = _angular_region(
                rays,
                center,
                values["azimuth_half_width_degrees"] * scale,
                values["zenith_center_degrees"] * scale,
                values["zenith_half_width_degrees"] * scale,
            )
            result = result + backend.astype(inside, dtype) * values["attenuation_db"]
        return result


@dataclass(frozen=True, slots=True)
class ScreenBlocker:
    """Stationary rectangular blocker represented by its centre and extent."""

    center_m: tuple[float, float, float]
    width_m: float
    height_m: float
    attenuation_db: float = 30.0

    def __post_init__(self) -> None:
        if len(self.center_m) != 3 or any(not isfinite(value) for value in self.center_m):
            raise ValueError("center_m must contain three finite coordinates")
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError("screen dimensions must be positive")
        if self.attenuation_db < 0.0 or not isfinite(self.attenuation_db):
            raise ValueError("attenuation_db must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ScreenBlockingModelB:
    """Snapshot Model B using stationary rectangular physical screens."""

    screens: tuple[ScreenBlocker, ...]

    def __post_init__(self) -> None:
        if not self.screens:
            raise ValueError("at least one screen is required")

    def loss_db(self, rays: RayBatch, topology: TopologyBatch) -> Any:
        """Project screens into receiver angular space and attenuate rays."""

        backend = get_backend(rays.metadata.backend)
        dtype = "float32" if rays.metadata.precision == "single" else "float64"
        result = backend.zeros(rays.shape, dtype=dtype, device=rays.metadata.device)
        for screen in self.screens:
            center = backend.array(screen.center_m, dtype=dtype, device=rays.metadata.device)
            delta = center[None, None, :] - topology.rx_positions_m
            horizontal = backend.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2)
            distance = backend.sqrt(horizontal**2 + delta[..., 2] ** 2)
            azimuth = backend.atan2(delta[..., 1], delta[..., 0])[:, :, None, None, None]
            zenith = backend.atan2(horizontal, delta[..., 2])[:, :, None, None, None]
            azimuth_half = backend.atan2(
                horizontal * 0.0 + screen.width_m * 0.5,
                distance,
            )[:, :, None, None, None]
            zenith_half = backend.atan2(
                horizontal * 0.0 + screen.height_m * 0.5,
                distance,
            )[:, :, None, None, None]
            inside = _angular_region(
                rays, azimuth, azimuth_half, zenith, zenith_half
            )
            result = result + backend.astype(inside, dtype) * screen.attenuation_db
        return result


def apply_blockage(rays: RayBatch, blockage_loss_db: Any) -> RayBatch:
    """Return a ray batch carrying attenuation for coefficient generation."""

    return RayBatch(
        rays.clusters,
        rays.azimuth_arrival_rad,
        rays.zenith_arrival_rad,
        rays.azimuth_departure_rad,
        rays.zenith_departure_rad,
        rays.xpr_linear,
        rays.initial_phase_rad,
        ray_mask=rays.ray_mask,
        blockage_loss_db=blockage_loss_db,
        sort_indices=rays.sort_indices,
        strongest_cluster_indices=rays.strongest_cluster_indices,
    )


def apply_blockage_to_coefficients(coefficients: Any, blockage_loss_db: Any) -> Any:
    """Apply per-ray blockage as voltage attenuation.

    Parameters
    ----------
    coefficients : backend array
        Per-ray coefficients whose leading axes are
        ``[batch, rx, tx, cluster, ray]``.
    blockage_loss_db : backend array
        Nonnegative decibel loss with exactly those five leading dimensions.

    Returns
    -------
    backend array
        Coefficients multiplied by ``10**(-loss_db/20)``.
    """

    backend = get_backend(backend_name_for_array(coefficients))
    coefficient_shape = backend.shape(coefficients)
    if backend.shape(blockage_loss_db) != coefficient_shape[:5]:
        raise ValueError("blockage loss shape must match coefficient ray axes")
    if not backend.all_nonnegative(blockage_loss_db):
        raise ValueError("blockage loss must be nonnegative")
    attenuation = backend.exp(-blockage_loss_db * (0.05 * 2.302585092994046))
    for _ in coefficient_shape[5:]:
        attenuation = attenuation[..., None]
    return coefficients * attenuation


@dataclass(frozen=True, slots=True)
class SpatialConsistencySnapshot:
    """Correlated snapshot fields without temporal evolution state."""

    los_uniform: Any
    lsp_gaussian: Any
    small_scale_gaussian: Any


class SnapshotSpatialConsistency:
    """Generate repeatable exponentially correlated snapshot variables."""

    def __init__(self, correlation_distance_m: float) -> None:
        self.field = ExponentialSpatialField(correlation_distance_m)

    def sample(
        self,
        positions_m: Any,
        random_stream: RandomStream[Any],
        *,
        num_lsp_fields: int = 7,
        num_small_scale_fields: int = 4,
    ) -> SpatialConsistencySnapshot:
        """Sample LoS, LSP, and small-scale correlated Gaussian fields."""

        if num_lsp_fields <= 0 or num_small_scale_fields <= 0:
            raise ValueError("field counts must be positive")
        backend = get_backend(random_stream.backend_name)
        los_gaussian = self.field.sample(
            positions_m, random_stream.fork("wireless.consistency.los")
        )
        los_uniform = 1.0 / (1.0 + backend.exp(-1.702 * los_gaussian))
        lsp = tuple(
            self.field.sample(
                positions_m, random_stream.fork(f"wireless.consistency.lsp.{index}")
            )
            for index in range(num_lsp_fields)
        )
        small = tuple(
            self.field.sample(
                positions_m,
                random_stream.fork(f"wireless.consistency.small_scale.{index}"),
            )
            for index in range(num_small_scale_fields)
        )
        return SpatialConsistencySnapshot(
            los_uniform,
            backend.stack(lsp, axis=-1),
            backend.stack(small, axis=-1),
        )


__all__ = [
    "CAPABILITIES",
    "NonSelfBlockingModelA",
    "ScreenBlocker",
    "ScreenBlockingModelB",
    "SelfBlockingModel",
    "SnapshotSpatialConsistency",
    "SpatialConsistencySnapshot",
    "WirelessCapabilityMetadata",
    "apply_blockage",
    "apply_blockage_to_coefficients",
    "load_blockage_parameters",
]
