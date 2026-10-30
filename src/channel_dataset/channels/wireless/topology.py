"""Backend-neutral link geometry, reciprocal views, and topology generation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from math import isfinite, pi
from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import ClusterBatch, RayBatch, TopologyBatch
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import RealizationMetadata


@dataclass(frozen=True, slots=True)
class LinkGeometry:
    """Derived pairwise link geometry in a global Cartesian frame.

    Parameters
    ----------
    distance_2d_m, distance_3d_m : backend array
        Horizontal and Euclidean distances with axes ``[batch, rx, tx]``.
    azimuth_departure_rad, zenith_departure_rad : backend array
        Direction from transmitter to receiver.
    azimuth_arrival_rad, zenith_arrival_rad : backend array
        Direction from receiver to transmitter.
    tx_height_m, rx_height_m : backend array
        Endpoint heights broadcast to link axes.
    metadata : RealizationMetadata
        Backend, device, precision, and provenance.
    """

    distance_2d_m: Any
    distance_3d_m: Any
    azimuth_departure_rad: Any
    zenith_departure_rad: Any
    azimuth_arrival_rad: Any
    zenith_arrival_rad: Any
    tx_height_m: Any
    rx_height_m: Any
    metadata: RealizationMetadata

    def __post_init__(self) -> None:
        backend = get_backend(self.metadata.backend)
        shape = backend.shape(self.distance_2d_m)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("link geometry must have positive [batch, rx, tx] dimensions")
        for name in (
            "distance_3d_m", "azimuth_departure_rad", "zenith_departure_rad",
            "azimuth_arrival_rad", "zenith_arrival_rad",
            "tx_height_m", "rx_height_m",
        ):
            value = getattr(self, name)
            if backend.shape(value) != shape:
                raise ValueError(f"{name} shape must match distance_2d_m")
            if not backend.is_floating(value) or not backend.all_finite(value):
                raise TypeError(f"{name} must be a finite floating array")
        if not backend.all_nonnegative(self.distance_2d_m):
            raise ValueError("distance_2d_m must be nonnegative")
        if backend.to_scalar(backend.min(self.distance_3d_m)) <= 0.0:
            raise ValueError("distance_3d_m must be positive")

    @property
    def shape(self) -> tuple[int, int, int]:
        """tuple of int: ``(batch, rx, tx)`` dimensions."""

        return get_backend(self.metadata.backend).shape(self.distance_2d_m)

    def reverse(self) -> LinkGeometry:
        """Return the reciprocal link view without mutating this object.

        Returns
        -------
        LinkGeometry
            Receiver/transmitter axes and arrival/departure angles exchanged.
        """

        backend = get_backend(self.metadata.backend)
        transpose = lambda value: backend.transpose(value, (0, 2, 1))
        return LinkGeometry(
            distance_2d_m=transpose(self.distance_2d_m),
            distance_3d_m=transpose(self.distance_3d_m),
            azimuth_departure_rad=transpose(self.azimuth_arrival_rad),
            zenith_departure_rad=transpose(self.zenith_arrival_rad),
            azimuth_arrival_rad=transpose(self.azimuth_departure_rad),
            zenith_arrival_rad=transpose(self.zenith_departure_rad),
            tx_height_m=transpose(self.rx_height_m),
            rx_height_m=transpose(self.tx_height_m),
            metadata=self.metadata,
        )


def compute_link_geometry(topology: TopologyBatch) -> LinkGeometry:
    """Compute pairwise distances and global arrival/departure angles.

    Parameters
    ----------
    topology : TopologyBatch
        Validated transmitter and receiver positions.

    Returns
    -------
    LinkGeometry
        Geometry for every ``[batch, rx, tx]`` pair.

    Raises
    ------
    ValueError
        If any transmitter and receiver are collocated.
    """

    backend = get_backend(topology.metadata.backend)
    displacement = topology.rx_positions_m[:, :, None, :] - topology.tx_positions_m[:, None, :, :]
    x, y, z = displacement[..., 0], displacement[..., 1], displacement[..., 2]
    distance_2d = backend.sqrt(x * x + y * y)
    distance_3d = backend.sqrt(distance_2d * distance_2d + z * z)
    if backend.to_scalar(backend.min(distance_3d)) <= 0.0:
        raise ValueError("collocated endpoints do not define link geometry")
    departure_azimuth = backend.atan2(y, x)
    departure_zenith = backend.atan2(distance_2d, z)
    arrival_azimuth = backend.atan2(-y, -x)
    arrival_zenith = backend.atan2(distance_2d, -z)
    shape = (topology.shape[0], topology.shape[1], topology.shape[2])
    tx_height = backend.broadcast_to(topology.tx_positions_m[:, None, :, 2], shape)
    rx_height = backend.broadcast_to(topology.rx_positions_m[:, :, None, 2], shape)
    return LinkGeometry(
        distance_2d, distance_3d, departure_azimuth, departure_zenith,
        arrival_azimuth, arrival_zenith, tx_height, rx_height, topology.metadata,
    )


def reverse_ray_batch(rays: RayBatch) -> RayBatch:
    """Return a reciprocal ray view with rx/tx and angle roles exchanged.

    Parameters
    ----------
    rays : RayBatch
        Downlink or uplink ray diagnostics.

    Returns
    -------
    RayBatch
        Storage-independent immutable container with transposed array views.
    """

    backend = get_backend(rays.metadata.backend)
    cluster_axes = (0, 2, 1, 3)
    ray_axes = (0, 2, 1, 3, 4)
    cluster_mask = (
        None if rays.cluster_mask is None else backend.transpose(rays.cluster_mask, cluster_axes)
    )
    clusters = ClusterBatch(
        delays_s=backend.transpose(rays.delays_s, cluster_axes),
        powers_linear=backend.transpose(rays.powers_linear, cluster_axes),
        azimuth_arrival_rad=backend.transpose(rays.clusters.azimuth_departure_rad, cluster_axes),
        zenith_arrival_rad=backend.transpose(rays.clusters.zenith_departure_rad, cluster_axes),
        azimuth_departure_rad=backend.transpose(rays.clusters.azimuth_arrival_rad, cluster_axes),
        zenith_departure_rad=backend.transpose(rays.clusters.zenith_arrival_rad, cluster_axes),
        metadata=rays.metadata,
        cluster_mask=cluster_mask,
    )
    optional_ray = lambda value: None if value is None else backend.transpose(value, ray_axes)
    optional_cluster = (
        lambda value: None if value is None else backend.transpose(value, cluster_axes)
    )
    strongest = None
    if rays.strongest_cluster_indices is not None:
        strongest = backend.transpose(rays.strongest_cluster_indices, cluster_axes)
    return RayBatch(
        clusters=clusters,
        azimuth_arrival_rad=backend.transpose(rays.azimuth_departure_rad, ray_axes),
        zenith_arrival_rad=backend.transpose(rays.zenith_departure_rad, ray_axes),
        azimuth_departure_rad=backend.transpose(rays.azimuth_arrival_rad, ray_axes),
        zenith_departure_rad=backend.transpose(rays.zenith_arrival_rad, ray_axes),
        xpr_linear=backend.transpose(rays.xpr_linear, ray_axes),
        initial_phase_rad=backend.transpose(rays.initial_phase_rad, ray_axes),
        ray_mask=optional_ray(rays.ray_mask),
        blockage_loss_db=optional_ray(rays.blockage_loss_db),
        sort_indices=optional_cluster(rays.sort_indices),
        strongest_cluster_indices=strongest,
    )


@dataclass(frozen=True, slots=True)
class RectangularTopologyConfig:
    """Configuration for independent endpoints in a rectangular region.

    Parameters
    ----------
    num_tx, num_rx : int
        Positive endpoint counts.
    width_m, length_m : float
        Positive horizontal region dimensions.
    tx_height_m, rx_height_m : float
        Positive fixed endpoint heights.
    rx_speed_mps : float, default=0
        Horizontal receiver speed with independently sampled heading.
    indoor_probability : float, default=0
        Probability that each receiver is marked indoor.
    fixed_los : bool, optional
        If supplied, fixes every link's LoS state; otherwise ``los_mask`` is
        absent for later scenario sampling.
    """

    num_tx: int
    num_rx: int
    width_m: float
    length_m: float
    tx_height_m: float
    rx_height_m: float
    rx_speed_mps: float = 0.0
    indoor_probability: float = 0.0
    fixed_los: bool | None = None

    def __post_init__(self) -> None:
        for name in ("num_tx", "num_rx"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("width_m", "length_m", "tx_height_m", "rx_height_m"):
            value = getattr(self, name)
            if not isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(float(self.rx_speed_mps)) or self.rx_speed_mps < 0.0:
            raise ValueError("rx_speed_mps must be finite and nonnegative")
        if not 0.0 <= self.indoor_probability <= 1.0:
            raise ValueError("indoor_probability must lie in [0, 1]")
        if self.fixed_los is not None and not isinstance(self.fixed_los, bool):
            raise TypeError("fixed_los must be bool or None")


class RectangularTopologyGenerator:
    """Generate reproducible endpoint geometry independently of channel models."""

    def __init__(self, config: RectangularTopologyConfig) -> None:
        self.config = config

    def generate(self, request: ChannelSamplingRequest) -> TopologyBatch:
        """Generate one topology batch using labeled child streams.

        Parameters
        ----------
        request : ChannelSamplingRequest
            Batch size, execution controls, and injected random stream.

        Returns
        -------
        TopologyBatch
            Reproducible positions, velocities, orientations, and states.
        """

        c, execution, backend = self.config, request.execution, request.execution.array_backend
        b = request.batch_size
        def positions(kind: str, count: int, height: float) -> Any:
            xy = request.random_stream.fork(f"wireless.topology.{kind}.positions").uniform(
                (b, count, 2), dtype=execution.float_dtype
            )
            scale = backend.array([c.width_m, c.length_m], dtype=execution.float_dtype, device=execution.device)
            xy = xy * scale
            z = backend.full((b, count, 1), height, dtype=execution.float_dtype, device=execution.device)
            return backend.concatenate((xy, z), axis=-1)
        tx_positions = positions("tx", c.num_tx, c.tx_height_m)
        rx_positions = positions("rx", c.num_rx, c.rx_height_m)
        tx_velocities = backend.zeros((b, c.num_tx, 3), dtype=execution.float_dtype, device=execution.device)
        heading = request.random_stream.fork("wireless.topology.rx.heading").uniform(
            (b, c.num_rx), -pi, pi, dtype=execution.float_dtype
        )
        rx_velocities = backend.stack(
            (c.rx_speed_mps * backend.cos(heading), c.rx_speed_mps * backend.sin(heading), heading * 0.0),
            axis=-1,
        )
        orientations_tx = backend.zeros((b, c.num_tx, 3), dtype=execution.float_dtype, device=execution.device)
        orientations_rx = backend.zeros((b, c.num_rx, 3), dtype=execution.float_dtype, device=execution.device)
        indoor = request.random_stream.fork("wireless.topology.rx.indoor").bernoulli(
            (b, c.num_rx), c.indoor_probability
        )
        site = backend.broadcast_to(
            backend.reshape(backend.arange(c.num_tx, dtype="int64", device=execution.device), (1, c.num_tx)),
            (b, c.num_tx),
        )
        digest = ConfigEnvelope(1, "wireless", "rectangular_topology", self.config.__dict__ if hasattr(self.config, "__dict__") else {
            field: getattr(self.config, field) for field in self.config.__dataclass_fields__
        }).model_digest
        try:
            software = version("channel-dataset")
        except PackageNotFoundError:
            software = "0.1.0"
        metadata = RealizationMetadata(
            "wireless", "rectangular_topology", 1, execution.backend, execution.precision,
            execution.device, request.random_stream.stream_id, digest,
            {field: getattr(c, field) for field in c.__dataclass_fields__}, software,
            request.request_id,
        )
        los = None
        if c.fixed_los is not None:
            los = backend.full((b, c.num_rx, c.num_tx), c.fixed_los, dtype="bool", device=execution.device)
        links = backend.full((b, c.num_rx, c.num_tx), True, dtype="bool", device=execution.device)
        topology = TopologyBatch(
            tx_positions, rx_positions, tx_velocities, rx_velocities,
            orientations_tx, orientations_rx, indoor, site, metadata, los, links,
        )
        compute_link_geometry(topology)
        return topology


__all__ = [
    "LinkGeometry", "RectangularTopologyConfig", "RectangularTopologyGenerator",
    "compute_link_geometry", "reverse_ray_batch",
]
