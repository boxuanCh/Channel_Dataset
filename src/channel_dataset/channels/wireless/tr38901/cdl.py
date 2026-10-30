"""3GPP TR 38.901 clustered-delay-line channel models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from math import isfinite, pi, sqrt
from types import MappingProxyType
from typing import Any

from channel_dataset.channels.wireless.antennas import (
    AntennaLayout,
    Orientation,
    load_antenna_layout,
    spherical_basis,
)
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import CIRBatch, ClusterBatch, RayBatch, TopologyBatch
from channel_dataset.config import ConfigEnvelope, load_resource_document
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry

_PROFILE_RESOURCES = {
    name: f"standards/wireless/tr_38_901/v16_1/cdl_{name[-1].lower()}.yaml"
    for name in ("CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E")
}
_RAY_OFFSET_RESOURCE = "standards/wireless/tr_38_901/v16_1/cdl_ray_offsets.yaml"


def _finite_tuple(value: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if any(not isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True, slots=True)
class CDLProfile:
    """Validated CDL cluster table and source provenance.

    Parameters
    ----------
    name, standard, standard_version, table : str
        Profile and exact source identifiers.
    delays, powers_db : tuple of float
        Normalized cluster delays and diffuse powers in decibels.
    aod_degrees, aoa_degrees, zod_degrees, zoa_degrees : tuple of float
        Mean departure/arrival azimuth and zenith angles.
    asd_degrees, asa_degrees, zsd_degrees, zsa_degrees : float
        Per-cluster ray angular spreads.
    xpr_db : float
        Per-ray cross-polarization power ratio.
    specular_power_db : float, optional
        First-cluster LoS component power for CDL-D/E.
    metadata : mapping
        Applicability, source, transcription, and review metadata.
    """

    name: str
    standard: str
    standard_version: str
    table: str
    delays: tuple[float, ...]
    powers_db: tuple[float, ...]
    aod_degrees: tuple[float, ...]
    aoa_degrees: tuple[float, ...]
    zod_degrees: tuple[float, ...]
    zoa_degrees: tuple[float, ...]
    asd_degrees: float
    asa_degrees: float
    zsd_degrees: float
    zsa_degrees: float
    xpr_db: float
    specular_power_db: float | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.delays)
        if count == 0:
            raise ValueError("CDL profile must contain clusters")
        for values in (
            self.powers_db, self.aod_degrees, self.aoa_degrees,
            self.zod_degrees, self.zoa_degrees,
        ):
            if len(values) != count:
                raise ValueError("CDL profile cluster columns must have equal lengths")
        if self.delays[0] != 0.0 or any(value < 0.0 for value in self.delays):
            raise ValueError("CDL delays must be nonnegative and begin at zero")
        if any(not isfinite(value) for values in (
            self.delays, self.powers_db, self.aod_degrees, self.aoa_degrees,
            self.zod_degrees, self.zoa_degrees,
        ) for value in values):
            raise ValueError("CDL table values must be finite")
        if self.specular_power_db is not None and self.name not in {"CDL-D", "CDL-E"}:
            raise ValueError("only CDL-D/E may define a specular component")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def num_clusters(self) -> int:
        """int: Number of diffuse clusters in the standard table."""

        return len(self.delays)

    @property
    def has_los(self) -> bool:
        """bool: Whether the profile includes a specular LoS component."""

        return self.specular_power_db is not None


@dataclass(frozen=True, slots=True)
class CDLConfig:
    """Typed single-link 3GPP CDL configuration.

    Parameters
    ----------
    profile : {"CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"}
        Standard cluster profile.
    standard_version : str
        Resource version, currently ``"16.1.0"``.
    delay_spread_s, carrier_frequency_hz, bandwidth_hz : float
        Desired RMS delay spread, carrier frequency, and occupied bandwidth.
    link_direction : {"downlink", "uplink"}
        Assigns base-station and terminal arrays to transmitter/receiver roles.
    base_station_array_profile, terminal_array_profile : str
        External antenna-layout resources.
    base_station_orientation_rad, terminal_orientation_rad : tuple of float
        Bearing, downtilt, and slant for each endpoint.
    base_station_velocity_mps, terminal_velocity_mps : tuple of float
        Cartesian endpoint velocities used for Doppler.
    link_distance_m : float
        Deterministic single-link separation used for topology and LoS phase.
    k_factor_db : float, optional
        Desired aggregate K-factor for CDL-D/E; omitted to retain table powers.
    """

    profile: str
    standard_version: str
    delay_spread_s: float
    carrier_frequency_hz: float
    bandwidth_hz: float
    link_direction: str
    base_station_array_profile: str
    terminal_array_profile: str
    base_station_orientation_rad: tuple[float, float, float]
    terminal_orientation_rad: tuple[float, float, float]
    base_station_velocity_mps: tuple[float, float, float]
    terminal_velocity_mps: tuple[float, float, float]
    link_distance_m: float
    k_factor_db: float | None

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_RESOURCES:
            raise ValueError(f"unsupported CDL profile: {self.profile!r}")
        if self.standard_version != "16.1.0":
            raise ValueError("standard_version must be '16.1.0'")
        for name in ("delay_spread_s", "carrier_frequency_hz", "bandwidth_hz", "link_distance_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.link_direction not in {"downlink", "uplink"}:
            raise ValueError("link_direction must be 'downlink' or 'uplink'")
        for name in ("base_station_array_profile", "terminal_array_profile"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty resource reference")
        for name in (
            "base_station_orientation_rad", "terminal_orientation_rad",
            "base_station_velocity_mps", "terminal_velocity_mps",
        ):
            object.__setattr__(self, name, _finite_tuple(getattr(self, name), name, 3))
        if self.k_factor_db is not None:
            if not isfinite(float(self.k_factor_db)):
                raise ValueError("k_factor_db must be finite")
            if self.profile not in {"CDL-D", "CDL-E"}:
                raise ValueError("k_factor_db is supported only for CDL-D/E")


def load_cdl_profile(name: str) -> CDLProfile:
    """Load a validated packaged CDL-A through CDL-E table.

    Parameters
    ----------
    name : str
        Standard profile identifier.

    Returns
    -------
    CDLProfile
        Immutable standard table and provenance.
    """

    try:
        document = load_resource_document(_PROFILE_RESOURCES[name])
    except KeyError as error:
        raise ValueError(f"unsupported CDL profile: {name!r}") from error
    expected = {
        "schema_version", "name", "standard", "standard_version", "table",
        "delays", "powers_db", "aod_degrees", "aoa_degrees", "zod_degrees",
        "zoa_degrees", "cluster_spreads_degrees", "xpr_db", "specular_power_db",
        "minimum_frequency_hz", "maximum_frequency_hz", "maximum_bandwidth_hz",
        "source_url", "transcription_date", "review_status",
    }
    if set(document) != expected:
        raise ValueError(
            "invalid CDL resource fields; "
            f"missing={sorted(expected - set(document))}, "
            f"unknown={sorted(set(document) - expected)}"
        )
    if document["schema_version"] != 1 or document["name"] != name:
        raise ValueError("CDL resource schema version or name is invalid")
    spreads = document["cluster_spreads_degrees"]
    if not isinstance(spreads, dict) or set(spreads) != {"asd", "asa", "zsd", "zsa"}:
        raise ValueError("cluster_spreads_degrees must define asd, asa, zsd, and zsa")
    count = len(document["delays"])
    metadata_keys = (
        "minimum_frequency_hz", "maximum_frequency_hz", "maximum_bandwidth_hz",
        "source_url", "transcription_date", "review_status",
    )
    return CDLProfile(
        name=name,
        standard=document["standard"],
        standard_version=document["standard_version"],
        table=document["table"],
        delays=_finite_tuple(document["delays"], "delays", count),
        powers_db=_finite_tuple(document["powers_db"], "powers_db", count),
        aod_degrees=_finite_tuple(document["aod_degrees"], "aod_degrees", count),
        aoa_degrees=_finite_tuple(document["aoa_degrees"], "aoa_degrees", count),
        zod_degrees=_finite_tuple(document["zod_degrees"], "zod_degrees", count),
        zoa_degrees=_finite_tuple(document["zoa_degrees"], "zoa_degrees", count),
        asd_degrees=float(spreads["asd"]), asa_degrees=float(spreads["asa"]),
        zsd_degrees=float(spreads["zsd"]), zsa_degrees=float(spreads["zsa"]),
        xpr_db=float(document["xpr_db"]),
        specular_power_db=(
            None if document["specular_power_db"] is None
            else float(document["specular_power_db"])
        ),
        metadata={key: document[key] for key in metadata_keys},
    )


def load_cdl_ray_offsets() -> tuple[float, ...]:
    """Load the 20 normalized ray offsets from Table 7.5-3.

    Returns
    -------
    tuple of float
        Signed offsets ordered by standard ray number.
    """

    document = load_resource_document(_RAY_OFFSET_RESOURCE)
    expected = {
        "schema_version", "standard", "standard_version", "table", "offsets",
        "source_url", "transcription_date", "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1:
        raise ValueError("invalid CDL ray-offset resource")
    if document["standard_version"] != "16.1.0" or document["table"] != "7.5-3":
        raise ValueError("CDL ray-offset source identity is invalid")
    return _finite_tuple(document["offsets"], "offsets", 20)


def parse_cdl_config(parameters: Mapping[str, Any]) -> CDLConfig:
    """Parse and strictly validate CDL model parameters.

    Parameters
    ----------
    parameters : mapping
        Mapping containing exactly the :class:`CDLConfig` fields.

    Returns
    -------
    CDLConfig
        Validated immutable model configuration.
    """

    fields = set(CDLConfig.__dataclass_fields__)
    unknown = set(parameters) - fields
    missing = fields - set(parameters)
    if unknown:
        raise ValueError(f"unknown CDL parameters: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing CDL parameters: {sorted(missing)}")
    values = dict(parameters)
    for name in (
        "base_station_orientation_rad", "terminal_orientation_rad",
        "base_station_velocity_mps", "terminal_velocity_mps",
    ):
        values[name] = _finite_tuple(values[name], name, 3)
    return CDLConfig(**values)


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0.1.0"


class ClusteredDelayLine(ChannelModel[ChannelSamplingRequest, CIRBatch]):
    """Generate antenna-aware 3GPP CDL MIMO channel coefficients.

    Parameters
    ----------
    config : CDLConfig
        Typed single-link model configuration.

    Notes
    -----
    Each diffuse cluster contains the 20 Table 7.5-3 rays. Angle coupling,
    four polarization phases, and initial phases use stable child streams.
    ``sample_rays`` exposes diagnostics while ``realize`` aggregates rays by
    cluster into a reusable :class:`CIRBatch`.
    """

    def __init__(self, config: CDLConfig) -> None:
        self.config = config
        self.profile = load_cdl_profile(config.profile)
        self.base_station_layout = load_antenna_layout(config.base_station_array_profile)
        self.terminal_layout = load_antenna_layout(config.terminal_array_profile)
        self._validate_applicability()
        self._resolved_config = asdict(config)
        self._digest = ConfigEnvelope(
            1, self.family, self.name, self._resolved_config
        ).model_digest

    @property
    def family(self) -> str:
        """str: Stable registry family, ``"wireless"``."""

        return "wireless"

    @property
    def name(self) -> str:
        """str: Stable registry name, ``"cdl"``."""

        return "cdl"

    @property
    def transmitter_layout(self) -> AntennaLayout:
        """AntennaLayout: Layout assigned to the transmitter role."""

        return (
            self.base_station_layout
            if self.config.link_direction == "downlink"
            else self.terminal_layout
        )

    @property
    def receiver_layout(self) -> AntennaLayout:
        """AntennaLayout: Layout assigned to the receiver role."""

        return (
            self.terminal_layout
            if self.config.link_direction == "downlink"
            else self.base_station_layout
        )

    def _validate_applicability(self) -> None:
        metadata = self.profile.metadata
        if self.profile.standard_version != self.config.standard_version:
            raise ValueError("configuration and CDL resource versions do not match")
        if not (
            float(metadata["minimum_frequency_hz"])
            <= self.config.carrier_frequency_hz
            <= float(metadata["maximum_frequency_hz"])
        ):
            raise ValueError("carrier_frequency_hz is outside CDL applicability")
        if self.config.bandwidth_hz > float(metadata["maximum_bandwidth_hz"]):
            raise ValueError("bandwidth_hz exceeds CDL applicability")

    def _metadata(self, request: ChannelSamplingRequest) -> RealizationMetadata:
        return RealizationMetadata(
            model_family=self.family, model_name=self.name, schema_version=1,
            backend=request.execution.backend, precision=request.execution.precision,
            device=request.execution.device, stream_id=request.random_stream.stream_id,
            config_digest=self._digest, resolved_configuration=self._resolved_config,
            software_version=_software_version(), request_id=request.request_id,
        )

    def _endpoint_values(self) -> tuple[tuple[float, ...], tuple[float, ...], Orientation, Orientation]:
        c = self.config
        if c.link_direction == "downlink":
            return (
                c.base_station_velocity_mps, c.terminal_velocity_mps,
                Orientation(*c.base_station_orientation_rad),
                Orientation(*c.terminal_orientation_rad),
            )
        return (
            c.terminal_velocity_mps, c.base_station_velocity_mps,
            Orientation(*c.terminal_orientation_rad),
            Orientation(*c.base_station_orientation_rad),
        )

    def _powers(self) -> tuple[tuple[float, ...], float]:
        diffuse = [10.0 ** (value / 10.0) for value in self.profile.powers_db]
        specular = (
            0.0 if self.profile.specular_power_db is None
            else 10.0 ** (self.profile.specular_power_db / 10.0)
        )
        if self.config.k_factor_db is not None:
            total_diffuse = sum(diffuse)
            k = 10.0 ** (self.config.k_factor_db / 10.0)
            diffuse = [value / total_diffuse / (k + 1.0) for value in diffuse]
            specular = k / (k + 1.0)
        else:
            total = sum(diffuse) + specular
            diffuse = [value / total for value in diffuse]
            specular /= total
        return tuple(diffuse), specular

    @property
    def cluster_delays_s(self) -> tuple[float, ...]:
        """tuple of float: Physical cluster delays after RMS scaling."""

        return tuple(value * self.config.delay_spread_s for value in self.profile.delays)

    def topology(self, request: ChannelSamplingRequest) -> TopologyBatch:
        """Create the deterministic single-link topology used by this model.

        Parameters
        ----------
        request : ChannelSamplingRequest
            Supplies batch dimensions and execution controls.

        Returns
        -------
        TopologyBatch
            One transmitter and receiver per batch entry.
        """

        backend = request.execution.array_backend
        b = request.batch_size
        aod = self.profile.aod_degrees[0] * pi / 180.0
        zod = self.profile.zod_degrees[0] * pi / 180.0
        radial, _, _ = spherical_basis(zod, aod, request.execution)
        zero = backend.zeros((b, 1, 3), dtype=request.execution.float_dtype, device=request.execution.device)
        rx = backend.broadcast_to(
            backend.reshape(radial * self.config.link_distance_m, (1, 1, 3)), (b, 1, 3)
        )
        tx_velocity, rx_velocity, tx_orientation, rx_orientation = self._endpoint_values()
        def repeated(values: tuple[float, ...]) -> Any:
            vector = backend.array(values, dtype=request.execution.float_dtype, device=request.execution.device)
            return backend.broadcast_to(backend.reshape(vector, (1, 1, 3)), (b, 1, 3))
        return TopologyBatch(
            tx_positions_m=zero, rx_positions_m=rx,
            tx_velocities_mps=repeated(tx_velocity), rx_velocities_mps=repeated(rx_velocity),
            tx_orientations_rad=repeated((tx_orientation.bearing_rad, tx_orientation.downtilt_rad, tx_orientation.slant_rad)),
            rx_orientations_rad=repeated((rx_orientation.bearing_rad, rx_orientation.downtilt_rad, rx_orientation.slant_rad)),
            rx_indoor=backend.zeros((b, 1), dtype="bool", device=request.execution.device),
            tx_site_ids=backend.zeros((b, 1), dtype="int64", device=request.execution.device),
            metadata=self._metadata(request),
            los_mask=backend.full((b, 1, 1), self.profile.has_los, dtype="bool", device=request.execution.device),
        )

    def sample_rays(self, request: ChannelSamplingRequest) -> RayBatch:
        """Sample coupled ray angles, XPR values, and initial phases.

        Parameters
        ----------
        request : ChannelSamplingRequest
            Batch size, execution context, and injected random stream.

        Returns
        -------
        RayBatch
            Twenty rays per standard cluster with axes
            ``[batch, 1, 1, cluster, ray]``.
        """

        backend = request.execution.array_backend
        execution = request.execution
        ray_offsets = load_cdl_ray_offsets()
        b, c, m = request.batch_size, self.profile.num_clusters, len(ray_offsets)
        offsets = backend.array(ray_offsets, dtype=execution.float_dtype, device=execution.device)
        angle_specs = (
            ("aoa", self.profile.aoa_degrees, self.profile.asa_degrees, False),
            ("zoa", self.profile.zoa_degrees, self.profile.zsa_degrees, True),
            ("aod", self.profile.aod_degrees, self.profile.asd_degrees, False),
            ("zod", self.profile.zod_degrees, self.profile.zsd_degrees, True),
        )
        angle_arrays = []
        for label, means, spread, zenith in angle_specs:
            batches = []
            for batch_index in range(b):
                clusters = []
                for cluster_index, mean in enumerate(means):
                    permutation = request.random_stream.fork(
                        f"wireless.cdl.{label}.{batch_index}.{cluster_index}"
                    ).permutation(m)
                    values = offsets[permutation] * spread + mean
                    if zenith:
                        values = backend.abs(
                            backend.atan2(
                                backend.sin(values * pi / 180.0),
                                backend.cos(values * pi / 180.0),
                            )
                        )
                    else:
                        values = backend.atan2(
                            backend.sin(values * pi / 180.0),
                            backend.cos(values * pi / 180.0),
                        )
                    clusters.append(values)
                batches.append(backend.stack(tuple(clusters)))
            angle_arrays.append(backend.reshape(backend.stack(tuple(batches)), (b, 1, 1, c, m)))
        diffuse, specular = self._powers()
        cluster_power = list(diffuse)
        cluster_power[0] += specular
        def cluster_array(values: tuple[float, ...] | list[float]) -> Any:
            vector = backend.array(values, dtype=execution.float_dtype, device=execution.device)
            return backend.broadcast_to(backend.reshape(vector, (1, 1, 1, c)), (b, 1, 1, c))
        clusters = ClusterBatch(
            delays_s=cluster_array(self.cluster_delays_s),
            powers_linear=cluster_array(cluster_power),
            azimuth_arrival_rad=cluster_array(tuple(v * pi / 180.0 for v in self.profile.aoa_degrees)),
            zenith_arrival_rad=cluster_array(tuple(v * pi / 180.0 for v in self.profile.zoa_degrees)),
            azimuth_departure_rad=cluster_array(tuple(v * pi / 180.0 for v in self.profile.aod_degrees)),
            zenith_departure_rad=cluster_array(tuple(v * pi / 180.0 for v in self.profile.zod_degrees)),
            metadata=self._metadata(request),
        )
        shape = (b, 1, 1, c, m)
        phases = request.random_stream.fork("wireless.cdl.initial_phases").uniform(
            shape, -pi, pi, dtype=execution.float_dtype
        )
        xpr = backend.full(
            shape, 10.0 ** (self.profile.xpr_db / 10.0),
            dtype=execution.float_dtype, device=execution.device,
        )
        return RayBatch(clusters, *angle_arrays, xpr, phases)

    def _array_field(self, layout: AntennaLayout, theta: Any, phi: Any, orientation: Orientation, request: ChannelSamplingRequest) -> tuple[Any, Any]:
        theta_fields, phi_fields = [], []
        for index in range(layout.array.num_elements):
            field = layout.array.field_pattern(
                index, theta, phi, request.execution, orientation=orientation
            )
            theta_fields.append(field.theta)
            phi_fields.append(field.phi)
        backend = request.execution.array_backend
        return backend.stack(tuple(theta_fields), axis=-1), backend.stack(tuple(phi_fields), axis=-1)

    def _coefficients(self, rays: RayBatch, request: ChannelSamplingRequest) -> Any:
        backend = request.execution.array_backend
        execution = request.execution
        b, _, _, c, m = rays.shape
        tx_layout, rx_layout = self.transmitter_layout, self.receiver_layout
        tx_velocity, rx_velocity, tx_orientation, rx_orientation = self._endpoint_values()
        aoa, zoa = rays.azimuth_arrival_rad[:, 0, 0], rays.zenith_arrival_rad[:, 0, 0]
        aod, zod = rays.azimuth_departure_rad[:, 0, 0], rays.zenith_departure_rad[:, 0, 0]
        rx_direction, _, _ = spherical_basis(zoa, aoa, execution)
        tx_direction, _, _ = spherical_basis(zod, aod, execution)
        rx_theta, rx_phi = self._array_field(rx_layout, zoa, aoa, rx_orientation, request)
        tx_theta, tx_phi = self._array_field(tx_layout, zod, aod, tx_orientation, request)
        shape = (b, c, m, 4)
        phases = request.random_stream.fork("wireless.cdl.polarization_phases").uniform(
            shape, -pi, pi, dtype=execution.float_dtype
        )
        phase_values = backend.exp(1j * phases)
        cross = 1.0 / backend.sqrt(rays.xpr_linear[:, 0, 0])
        p00, p01 = phase_values[..., 0], cross * phase_values[..., 1]
        p10, p11 = cross * phase_values[..., 2], phase_values[..., 3]
        polarization = (
            rx_theta[..., :, None]
            * (p00[..., None, None] * tx_theta[..., None, :] + p01[..., None, None] * tx_phi[..., None, :])
            + rx_phi[..., :, None]
            * (p10[..., None, None] * tx_theta[..., None, :] + p11[..., None, None] * tx_phi[..., None, :])
        )
        rx_positions = rx_layout.array.positions_wavelengths(execution)
        tx_positions = tx_layout.array.positions_wavelengths(execution)
        spatial = (
            backend.einsum("bcmj,aj->bcma", rx_direction, rx_positions)[..., :, None]
            + backend.einsum("bcmj,aj->bcma", tx_direction, tx_positions)[..., None, :]
        )
        array_phase = backend.exp(1j * 2.0 * pi * spatial)
        rx_velocity_array = backend.array(rx_velocity, dtype=execution.float_dtype, device=execution.device)
        tx_velocity_array = backend.array(tx_velocity, dtype=execution.float_dtype, device=execution.device)
        doppler = (
            backend.einsum("bcmj,j->bcm", rx_direction, rx_velocity_array)
            + backend.einsum("bcmj,j->bcm", tx_direction, tx_velocity_array)
        ) * self.config.carrier_frequency_hz / SPEED_OF_LIGHT_MPS
        time = backend.arange(
            request.num_time_samples, dtype=execution.float_dtype, device=execution.device
        ) / request.sampling_frequency_hz
        evolution = backend.exp(1j * 2.0 * pi * doppler[..., None] * time)
        initial = backend.exp(1j * rays.initial_phase_rad[:, 0, 0])
        diffuse, specular = self._powers()
        power = backend.array(diffuse, dtype=execution.float_dtype, device=execution.device)
        scale = backend.reshape(backend.sqrt(power / m), (1, c, 1, 1, 1, 1))
        ray_values = (
            scale * polarization[..., None] * array_phase[..., None]
            * initial[..., None, None, None] * evolution[..., None, None, :]
        )
        coefficients = backend.sum(ray_values, axis=2)
        if specular > 0.0:
            # The deterministic LoS ray uses the first cluster mean direction,
            # the standard co-polar sign reversal, and geometric phase.
            rt, rp = rx_theta[:, 0, 0], rx_phi[:, 0, 0]
            tt, tp = tx_theta[:, 0, 0], tx_phi[:, 0, 0]
            los_polarization = rt[..., :, None] * tt[..., None, :] - rp[..., :, None] * tp[..., None, :]
            los_spatial = array_phase[:, 0, 0]
            los_doppler = doppler[:, 0, 0]
            los_phase = backend.exp(
                -1j * 2.0 * pi * self.config.link_distance_m
                * self.config.carrier_frequency_hz / SPEED_OF_LIGHT_MPS
                + 1j * 2.0 * pi * los_doppler[..., None] * time
            )
            los = sqrt(specular) * los_polarization[..., None] * los_spatial[..., None] * los_phase[:, None, None, :]
            first = coefficients[:, 0] + los
            coefficients = backend.concatenate((first[:, None], coefficients[:, 1:]), axis=1)
        # [batch, cluster, rx_ant, tx_ant, time] -> canonical CIR axes.
        coefficients = backend.transpose(coefficients, (0, 2, 3, 1, 4))
        return backend.reshape(
            coefficients,
            (b, 1, rx_layout.array.num_elements, 1, tx_layout.array.num_elements, c, request.num_time_samples),
        )

    def realize_with_rays(self, request: ChannelSamplingRequest) -> tuple[CIRBatch, RayBatch]:
        """Generate a CIR together with its ray-level diagnostics.

        Parameters
        ----------
        request : ChannelSamplingRequest
            Sampling dimensions, execution controls, and injected RNG.

        Returns
        -------
        cir, rays : tuple
            Aggregated MIMO CIR and the rays used to construct it.
        """

        rays = self.sample_rays(request)
        return CIRBatch(self._coefficients(rays, request), rays.delays_s, rays.metadata), rays

    def realize(self, request: ChannelSamplingRequest) -> CIRBatch:
        """Generate a reusable antenna- and polarization-aware CDL CIR.

        Parameters
        ----------
        request : ChannelSamplingRequest
            Sampling dimensions, execution controls, and injected RNG.

        Returns
        -------
        CIRBatch
            Cluster-aggregated coefficients using canonical CIR axes.
        """

        return self.realize_with_rays(request)[0]


def _build_cdl(config: object) -> ClusteredDelayLine:
    if not isinstance(config, CDLConfig):
        raise TypeError("CDL builder requires CDLConfig")
    return ClusteredDelayLine(config)


def register_cdl_model(registry: ModelRegistry) -> None:
    """Register the ``wireless/cdl`` parser and model builder.

    Parameters
    ----------
    registry : ModelRegistry
        Registry receiving the typed model registration.
    """

    registry.register("wireless", "cdl", parse_cdl_config, _build_cdl)


__all__ = [
    "CDLConfig", "CDLProfile", "ClusteredDelayLine", "load_cdl_profile",
    "load_cdl_ray_offsets",
    "parse_cdl_config", "register_cdl_model",
]
