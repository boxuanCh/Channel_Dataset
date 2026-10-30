"""Backend-neutral 3GPP cluster and ray initialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import log, pi
from types import MappingProxyType
from typing import Any

from channel_dataset.backends import RandomStream, get_backend
from channel_dataset.channels.wireless.math import matrix_sqrt_psd
from channel_dataset.channels.wireless.topology import compute_link_geometry
from channel_dataset.channels.wireless.tr38901 import load_cdl_ray_offsets
from channel_dataset.channels.wireless.types import (
    ClusterBatch,
    LargeScaleParameters,
    RayBatch,
    TopologyBatch,
)
from channel_dataset.config import load_resource_document

_RESOURCE = "standards/wireless/tr_38_901/v16_1/scenario_rays.yaml"
_SCENARIOS = frozenset({"RMa", "UMa", "UMi", "InH", "InF"})


@dataclass(frozen=True, slots=True)
class RayGenerationConfig:
    """Small-scale initialization parameters for one propagation state.

    Parameters
    ----------
    num_clusters : int
        Number of valid clusters.
    delay_scaling : float
        Exponential delay scaling parameter :math:`r_\tau`.
    cluster_shadow_std_db : float
        Per-cluster shadowing standard deviation in decibels.
    xpr_mean_db, xpr_std_db : float
        Lognormal cross-polarization-ratio parameters in decibels.
    rays_per_cluster : int
        Number of rays generated for every valid cluster.
    """

    num_clusters: int
    delay_scaling: float
    cluster_shadow_std_db: float
    xpr_mean_db: float
    xpr_std_db: float
    rays_per_cluster: int = 20

    def __post_init__(self) -> None:
        if self.num_clusters <= 0 or self.rays_per_cluster <= 0:
            raise ValueError("cluster and ray counts must be positive")
        if self.delay_scaling <= 1.0:
            raise ValueError("delay_scaling must be greater than one")
        if self.cluster_shadow_std_db < 0.0 or self.xpr_std_db < 0.0:
            raise ValueError("standard deviations must be nonnegative")


@dataclass(frozen=True, slots=True)
class SubclusterPlan:
    """Ray grouping used to split the two strongest clusters.

    Parameters
    ----------
    strongest_cluster_indices : backend array
        Two cluster indices on axes ``[batch, rx, tx, strongest]``.
    ray_group_indices : backend array
        Integer group identifier for each ray on axis ``[ray]``.
    delay_offsets : tuple of float
        Relative subcluster-delay multipliers ``(0, 1.28, 2.56)``.
    """

    strongest_cluster_indices: Any
    ray_group_indices: Any
    delay_offsets: tuple[float, float, float] = (0.0, 1.28, 2.56)


def load_ray_generation_parameters(
    scenario: str,
) -> Mapping[str, RayGenerationConfig]:
    """Load validated LoS and NLoS ray parameters for a scenario.

    Parameters
    ----------
    scenario : {"RMa", "UMa", "UMi", "InH", "InF"}
        3GPP scenario name.

    Returns
    -------
    mapping
        Read-only mapping with ``"los"`` and ``"nlos"`` configurations.
    """

    if scenario not in _SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario!r}")
    document = load_resource_document(_RESOURCE)
    expected = {
        "schema_version", "standard", "standard_version", "tables", "scenarios",
        "rays_per_cluster", "subcluster_ray_groups", "source_url",
        "transcription_date", "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1:
        raise ValueError("invalid scenario-ray resource")
    if set(document["scenarios"]) != _SCENARIOS:
        raise ValueError("scenario-ray resource has incomplete scenarios")
    rays = int(document["rays_per_cluster"])
    result = {
        state: RayGenerationConfig(rays_per_cluster=rays, **values)
        for state, values in document["scenarios"][scenario].items()
    }
    if set(result) != {"los", "nlos"}:
        raise ValueError("scenario-ray resource must define LoS and NLoS states")
    return MappingProxyType(result)


def _wrap_azimuth(value: Any, backend: Any) -> Any:
    return backend.atan2(backend.sin(value), backend.cos(value))


def _wrap_zenith(value: Any, backend: Any) -> Any:
    return backend.abs(_wrap_azimuth(value, backend))


class ExponentialSpatialField:
    """Sample a Gaussian field with exponential spatial covariance.

    Parameters
    ----------
    correlation_distance_m : float
        Positive distance at which correlation has decayed to ``exp(-1)``.
    """

    def __init__(self, correlation_distance_m: float) -> None:
        if correlation_distance_m <= 0.0:
            raise ValueError("correlation_distance_m must be positive")
        self.correlation_distance_m = float(correlation_distance_m)

    def sample(self, positions_m: Any, random_stream: RandomStream[Any]) -> Any:
        """Draw one spatially correlated value per point.

        Parameters
        ----------
        positions_m : backend array
            Cartesian points on axes ``[batch, point, xyz]``.
        random_stream : RandomStream
            Explicit backend-native source of independent Gaussian samples.

        Returns
        -------
        backend array
            Correlated standard-normal values on ``[batch, point]``.
        """

        backend = get_backend(random_stream.backend_name)
        shape = backend.shape(positions_m)
        if len(shape) != 3 or shape[-1] not in {2, 3}:
            raise ValueError("positions_m must have shape [batch, point, 2 or 3]")
        if not backend.is_floating(positions_m) or not backend.all_finite(positions_m):
            raise TypeError("positions_m must be a finite floating array")
        difference = positions_m[:, :, None, :] - positions_m[:, None, :, :]
        distance = backend.sqrt(backend.sum(difference * difference, axis=-1))
        covariance = backend.exp(-distance / self.correlation_distance_m)
        root = matrix_sqrt_psd(covariance)
        dtype = "float32" if str(backend.dtype(positions_m)).endswith("32") else "float64"
        independent = random_stream.fork("wireless.spatial_field").normal(
            shape[:2], dtype=dtype
        )
        return backend.einsum("bij,bj->bi", root, independent)


class ScenarioRayGenerator:
    """Generate antenna-independent small-scale rays for a 3GPP scenario.

    Parameters
    ----------
    scenario : {"RMa", "UMa", "UMi", "InH", "InF"}
        Supported scenario identifier.

    Notes
    -----
    Every stochastic component uses a named child stream. Optional diagnostics
    therefore cannot perturb another component's random draws.
    """

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.parameters = load_ray_generation_parameters(scenario)

    def generate(
        self,
        topology: TopologyBatch,
        large_scale: LargeScaleParameters,
        random_stream: RandomStream[Any],
        *,
        los_mask: Any | None = None,
    ) -> RayBatch:
        """Generate delays, powers, angles, XPR, phases, and masks.

        Parameters
        ----------
        topology : TopologyBatch
            Endpoint positions and link validity.
        large_scale : LargeScaleParameters
            Delay and angular spreads for the same links.
        random_stream : RandomStream
            Explicit stream matching the arrays' backend and device.
        los_mask : backend array, optional
            Boolean propagation state on ``[batch, rx, tx]``. When omitted,
            ``topology.los_mask`` is used, or a nonzero K-factor is treated as
            LoS if the topology has no fixed state.

        Returns
        -------
        RayBatch
            Validated rays on ``[batch, rx, tx, cluster, ray]``.
        """

        backend = get_backend(topology.metadata.backend)
        if topology.shape != large_scale.shape:
            raise ValueError("topology and large-scale link dimensions must match")
        if large_scale.metadata.backend != topology.metadata.backend:
            raise ValueError("topology and large-scale backends must match")
        if random_stream.backend_name != topology.metadata.backend:
            raise ValueError("random stream backend must match topology")
        if random_stream.device != topology.metadata.device:
            raise ValueError("random stream device must match topology")
        if los_mask is None:
            los_mask = topology.los_mask
        if los_mask is None:
            los_mask = large_scale.k_factor_db != 0.0
        if backend.shape(los_mask) != topology.shape or not backend.is_bool(los_mask):
            raise ValueError("los_mask must be Boolean with topology link shape")

        dtype = "float32" if topology.metadata.precision == "single" else "float64"
        device = topology.metadata.device
        shape = topology.shape
        los_config, nlos_config = self.parameters["los"], self.parameters["nlos"]
        max_clusters = max(los_config.num_clusters, nlos_config.num_clusters)
        rays = los_config.rays_per_cluster
        selection = backend.astype(los_mask, dtype)
        counts = (
            selection * los_config.num_clusters
            + (1.0 - selection) * nlos_config.num_clusters
        )
        cluster_index = backend.reshape(
            backend.arange(max_clusters, dtype="int64", device=device),
            (1, 1, 1, max_clusters),
        )
        cluster_mask = cluster_index < counts[..., None]
        link_mask = topology.link_mask
        if link_mask is not None:
            cluster_mask = cluster_mask * link_mask[..., None]
        valid = backend.astype(cluster_mask, dtype)

        delay_scaling = (
            selection * los_config.delay_scaling
            + (1.0 - selection) * nlos_config.delay_scaling
        )
        uniforms = random_stream.fork("wireless.rays.delays").uniform(
            (*shape, max_clusters), 1.0e-7, 1.0, dtype=dtype
        )
        raw_delays = (
            -delay_scaling[..., None]
            * large_scale.delay_spread_s[..., None]
            * backend.log(uniforms)
        )
        sortable = raw_delays * valid + (1.0 - valid) * 1.0e30
        sort_indices = backend.argsort(sortable, axis=-1)
        delays = backend.take_along_axis(raw_delays, sort_indices, axis=-1)
        delays = (delays - delays[..., :1]) * valid

        shadow_std = (
            selection * los_config.cluster_shadow_std_db
            + (1.0 - selection) * nlos_config.cluster_shadow_std_db
        )
        shadow = random_stream.fork("wireless.rays.cluster_shadowing").normal(
            (*shape, max_clusters), dtype=dtype
        )
        shadow = backend.take_along_axis(shadow, sort_indices, axis=-1)
        powers = backend.exp(
            -delays * (delay_scaling[..., None] - 1.0)
            / (delay_scaling[..., None] * large_scale.delay_spread_s[..., None] + 1.0e-30)
            - shadow * shadow_std[..., None] * log(10.0) / 10.0
        ) * valid
        powers = powers / (backend.sum(powers, axis=-1)[..., None] + 1.0e-30)

        geometry = compute_link_geometry(topology)
        angle_specs = (
            ("arrival_azimuth", geometry.azimuth_arrival_rad, large_scale.azimuth_spread_arrival_rad, False),
            ("arrival_zenith", geometry.zenith_arrival_rad, large_scale.zenith_spread_arrival_rad, True),
            ("departure_azimuth", geometry.azimuth_departure_rad, large_scale.azimuth_spread_departure_rad, False),
            ("departure_zenith", geometry.zenith_departure_rad, large_scale.zenith_spread_departure_rad, True),
        )
        cluster_angles = []
        ray_angles = []
        offsets = backend.array(load_cdl_ray_offsets(), dtype=dtype, device=device)
        offset_grid = backend.broadcast_to(
            backend.reshape(offsets, (1, 1, 1, 1, rays)),
            (*shape, max_clusters, rays),
        )
        for label, center, spread, zenith in angle_specs:
            perturbation = random_stream.fork(f"wireless.rays.cluster_angle.{label}").normal(
                (*shape, max_clusters), dtype=dtype
            )
            cluster_angle = center[..., None] + spread[..., None] * perturbation
            cluster_angle = (
                _wrap_zenith(cluster_angle, backend)
                if zenith else _wrap_azimuth(cluster_angle, backend)
            )
            cluster_angle = cluster_angle * valid
            cluster_angles.append(cluster_angle)
            coupling_keys = random_stream.fork(f"wireless.rays.coupling.{label}").uniform(
                (*shape, max_clusters, rays), dtype=dtype
            )
            permutation = backend.argsort(coupling_keys, axis=-1)
            coupled_offsets = backend.take_along_axis(offset_grid, permutation, axis=-1)
            angle = cluster_angle[..., None] + spread[..., None, None] * coupled_offsets
            angle = _wrap_zenith(angle, backend) if zenith else _wrap_azimuth(angle, backend)
            ray_angles.append(angle)

        ray_mask = backend.broadcast_to(cluster_mask[..., None], (*shape, max_clusters, rays))
        ray_valid = backend.astype(ray_mask, dtype)
        ray_angles = [angle * ray_valid for angle in ray_angles]
        xpr_mean = selection * los_config.xpr_mean_db + (1.0 - selection) * nlos_config.xpr_mean_db
        xpr_std = selection * los_config.xpr_std_db + (1.0 - selection) * nlos_config.xpr_std_db
        xpr_normal = random_stream.fork("wireless.rays.xpr").normal(
            (*shape, max_clusters, rays), dtype=dtype
        )
        xpr = backend.exp(
            (xpr_mean[..., None, None] + xpr_std[..., None, None] * xpr_normal)
            * log(10.0) / 10.0
        )
        xpr = xpr * ray_valid + (1.0 - ray_valid)
        phases = random_stream.fork("wireless.rays.initial_phase").uniform(
            (*shape, max_clusters, rays), -pi, pi, dtype=dtype
        ) * ray_valid
        strongest = backend.argsort(-powers, axis=-1)[..., : min(2, max_clusters)]
        clusters = ClusterBatch(
            delays,
            powers,
            cluster_angles[0],
            cluster_angles[1],
            cluster_angles[2],
            cluster_angles[3],
            large_scale.metadata,
            cluster_mask,
        )
        return RayBatch(
            clusters,
            ray_angles[0],
            ray_angles[1],
            ray_angles[2],
            ray_angles[3],
            xpr,
            phases,
            ray_mask=ray_mask,
            sort_indices=sort_indices,
            strongest_cluster_indices=strongest,
        )

    def subcluster_plan(self, rays: RayBatch) -> SubclusterPlan:
        """Return standard ray groups for the two strongest clusters."""

        if rays.strongest_cluster_indices is None:
            raise ValueError("rays do not identify strongest clusters")
        document = load_resource_document(_RESOURCE)
        groups = document["subcluster_ray_groups"]
        assignments = [0] * rays.shape[-1]
        for group_index, indices in enumerate(groups):
            for ray_index in indices:
                if ray_index < len(assignments):
                    assignments[ray_index] = group_index
        backend = get_backend(rays.metadata.backend)
        values = backend.array(assignments, dtype="int64", device=rays.metadata.device)
        return SubclusterPlan(rays.strongest_cluster_indices, values)


__all__ = [
    "ExponentialSpatialField",
    "RayGenerationConfig",
    "ScenarioRayGenerator",
    "SubclusterPlan",
    "load_ray_generation_parameters",
]
