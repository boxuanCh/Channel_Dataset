"""3GPP TR 38.811 NTN clustered-delay-line channel models."""

from __future__ import annotations

from dataclasses import asdict
from math import pi, sqrt

from channel_dataset.channels.wireless.antennas import load_antenna_layout, spherical_basis
from channel_dataset.channels.wireless.tr38811.config import (
    TR38811CDLConfig,
    TR38811DeploymentConfig,
    parse_cdl_config,
)
from channel_dataset.channels.wireless.tr38811.geometry import compute_ntn_geometry, doppler_phase
from channel_dataset.channels.wireless.tr38811.large_scale import sample_large_scale
from channel_dataset.channels.wireless.tr38811.polarization import faraday_rotation_matrix
from channel_dataset.channels.wireless.tr38811.profiles import (
    load_cdl_profile,
    validate_deployment_applicability,
    validate_topology_applicability,
)
from channel_dataset.channels.wireless.tr38811.tdl import _software_version
from channel_dataset.channels.wireless.tr38811.types import (
    NTNChannelRealization,
    NTNChannelSamplingRequest,
)
from channel_dataset.channels.wireless.tr38901.cdl import load_cdl_ray_offsets
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry


class TR38811CDLChannel(ChannelModel[NTNChannelSamplingRequest, NTNChannelRealization]):
    """Sample antenna-aware NTN-CDL-A/B/C/D realizations."""

    def __init__(self, config: TR38811CDLConfig) -> None:
        if not isinstance(config, TR38811CDLConfig):
            raise TypeError("TR38811CDLChannel requires TR38811CDLConfig")
        self.config = config
        validate_deployment_applicability(
            TR38811DeploymentConfig(
                deployment=config.deployment,
                standard_version=config.standard_version,
                environment=config.environment,
                carrier_frequency_hz=config.carrier_frequency_hz,
                bandwidth_hz=config.bandwidth_hz,
            ),
            model_kind="frequency_selective",
        )
        self.profile = load_cdl_profile(config.profile)
        self.tx_layout = load_antenna_layout(config.tx_array_profile)
        self.rx_layout = load_antenna_layout(config.rx_array_profile)
        self._resolved = asdict(config)
        self._digest = ConfigEnvelope(1, self.family, self.name, self._resolved).model_digest

    @property
    def family(self) -> str:
        """Return the stable registry family."""

        return "wireless"

    @property
    def name(self) -> str:
        """Return the stable registry model name."""

        return "tr38811_cdl"

    def _metadata(self, request: NTNChannelSamplingRequest) -> RealizationMetadata:
        sampling = request.sampling
        return RealizationMetadata(
            self.family, self.name, 1, sampling.execution.backend,
            sampling.execution.precision, sampling.execution.device,
            sampling.random_stream.stream_id, self._digest, self._resolved,
            _software_version(), sampling.request_id,
            root_seed=sampling.random_stream.root_seed, model_version="1",
            standard_version="15.4.0",
        )

    def _powers(self) -> tuple[float, ...]:
        raw = [10.0 ** (value / 10.0) for value in self.profile.powers_db]
        if self.config.k_factor_db is None:
            total = sum(raw)
            return tuple(value / total for value in raw)
        k_linear = 10.0 ** (self.config.k_factor_db / 10.0)
        diffuse_sum = sum(raw[1:])
        return (k_linear / (k_linear + 1.0),) + tuple(
            value / diffuse_sum / (k_linear + 1.0) for value in raw[1:]
        )

    @staticmethod
    def _wrap_azimuth(value: float) -> float:
        return ((value + 180.0) % 360.0) - 180.0

    @staticmethod
    def _wrap_zenith(value: float) -> float:
        wrapped = value % 360.0
        return 360.0 - wrapped if wrapped > 180.0 else wrapped

    def _scaled_means(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        aoa = self.profile.aoa_degrees
        zoa = self.profile.zoa_degrees
        aoa_center = aoa[0]
        zoa_center = zoa[0]
        asa_scale = self.config.desired_asa_degrees / max(self.profile.asa_degrees, 1e-12)
        zsa_scale = self.config.desired_zsa_degrees / max(self.profile.zsa_degrees, 1e-12)
        return (
            tuple(self._wrap_azimuth(aoa_center + (value - aoa_center) * asa_scale) for value in aoa),
            tuple(self._wrap_zenith(zoa_center + (value - zoa_center) * zsa_scale) for value in zoa),
        )

    def _one_link(
        self,
        request: NTNChannelSamplingRequest,
        common_phase: object,
        batch: int,
        rx: int,
        tx: int,
    ) -> object:
        sampling = request.sampling
        execution = sampling.execution
        backend = execution.array_backend
        offsets = load_cdl_ray_offsets()
        aoa_means, zoa_means = self._scaled_means()
        powers = self._powers()
        path_values = []
        tx_positions = self.tx_layout.array.positions_wavelengths(execution)
        rx_positions = self.rx_layout.array.positions_wavelengths(execution)
        for path, power in enumerate(powers):
            ray_matrices = []
            ray_count = 1 if self.profile.has_los and path == 0 else len(offsets)
            phases = sampling.random_stream.fork(
                f"wireless.tr38811.cdl.phase.{batch}.{rx}.{tx}.{path}"
            ).uniform((ray_count,), -pi, pi, dtype=execution.float_dtype)
            for ray in range(ray_count):
                offset = (
                    0.0
                    if ray_count == 1
                    else float(offsets[ray])
                )
                aoa_degrees = aoa_means[path] + offset * self.profile.asa_degrees
                zoa_degrees = zoa_means[path] + offset * self.profile.zsa_degrees
                # Reflect a ray crossing either zenith pole and rotate its
                # azimuth by 180 degrees so the represented global direction
                # is continuous and the spherical basis receives theta in
                # [0, pi].
                wrapped_zoa = zoa_degrees % 360.0
                if wrapped_zoa > 180.0:
                    wrapped_zoa = 360.0 - wrapped_zoa
                    aoa_degrees += 180.0
                aoa = self._wrap_azimuth(aoa_degrees) * pi / 180.0
                zoa = wrapped_zoa * pi / 180.0
                aod = self.profile.aod_degrees[path] * pi / 180.0
                zod = self.profile.zod_degrees[path] * pi / 180.0
                aoa_array = backend.array(aoa, dtype=execution.float_dtype, device=execution.device)
                zoa_array = backend.array(zoa, dtype=execution.float_dtype, device=execution.device)
                aod_array = backend.array(aod, dtype=execution.float_dtype, device=execution.device)
                zod_array = backend.array(zod, dtype=execution.float_dtype, device=execution.device)
                rx_direction, _, _ = spherical_basis(zoa_array, aoa_array, execution)
                tx_direction, _, _ = spherical_basis(zod_array, aod_array, execution)
                rx_steering = backend.exp(
                    1j * 2.0 * pi * backend.einsum("aj,j->a", rx_positions, rx_direction)
                )
                tx_steering = backend.exp(
                    1j * 2.0 * pi * backend.einsum("aj,j->a", tx_positions, tx_direction)
                )
                ray_matrices.append(
                    backend.exp(1j * phases[ray])
                    * rx_steering[:, None]
                    * tx_steering[None, :]
                )
            matrix = backend.sum(backend.stack(tuple(ray_matrices)), axis=0) / sqrt(ray_count)
            path_values.append(
                sqrt(power) * matrix[..., None] * common_phase[batch, rx, tx, :]
            )
        return backend.stack(tuple(path_values), axis=2)

    def realize(self, request: NTNChannelSamplingRequest) -> NTNChannelRealization:
        """Sample one fixed-window NTN CDL realization."""

        if not isinstance(request, NTNChannelSamplingRequest):
            raise TypeError("request must be NTNChannelSamplingRequest")
        validate_topology_applicability(request.topology, self.config.deployment)
        sampling = request.sampling
        execution = sampling.execution
        backend = execution.array_backend
        metadata = self._metadata(request)
        geometry = compute_ntn_geometry(
            request.topology, sampling.num_time_samples,
            self.config.carrier_frequency_hz, metadata=metadata,
        )
        common = doppler_phase(
            geometry.common_doppler_hz, sampling.sampling_frequency_hz, execution
        )
        batches = []
        for batch in range(geometry.shape[0]):
            receivers = []
            for rx in range(geometry.shape[1]):
                transmitters = [
                    self._one_link(request, common, batch, rx, tx)
                    for tx in range(geometry.shape[2])
                ]
                receivers.append(backend.stack(tuple(transmitters), axis=1))
            batches.append(backend.stack(tuple(receivers), axis=0))
        # Built axes are [batch, rx, rx_ant, tx, tx_ant, path, time].
        coefficients = backend.stack(tuple(batches), axis=0)
        if (
            self.config.faraday_rotation_rad != 0.0
            and backend.shape(coefficients)[2:5:2] == (2, 2)
        ):
            rotation = faraday_rotation_matrix(self.config.faraday_rotation_rad, execution)
            coefficients = backend.einsum("ij,brjtkps->britkps", rotation, coefficients)
        los_value = self.profile.has_los
        los_mask = backend.full(
            geometry.shape[:3], los_value, dtype="bool", device=execution.device
        )
        deployment = TR38811DeploymentConfig(
            self.config.deployment, self.config.standard_version, self.config.environment,
            self.config.carrier_frequency_hz, self.config.bandwidth_hz,
            apply_clutter_loss=self.config.apply_large_scale,
            apply_shadow_fading=self.config.apply_large_scale,
            apply_o2i_loss=self.config.apply_large_scale and self.config.deployment == "D5",
            apply_atmospheric_loss=False, apply_scintillation=False,
        )
        large_scale = sample_large_scale(
            geometry, request.topology, los_mask, deployment,
            sampling.random_stream, execution, metadata,
        )
        if self.config.apply_large_scale:
            from math import log

            amplitude = backend.exp((-log(10.0) / 20.0) * large_scale.total_db)
            coefficients = coefficients * amplitude[:, :, None, :, None, None, :]
        coefficients = backend.where(
            geometry.visibility[:, :, None, :, None, None, :],
            coefficients,
            coefficients * 0.0,
        )
        delays_vector = backend.array(
            [value * self.config.delay_spread_s for value in self.profile.delays],
            dtype=execution.float_dtype, device=execution.device,
        )
        path_count = len(self.profile.delays)
        delays = backend.broadcast_to(
            backend.reshape(delays_vector, (1, 1, 1, path_count)),
            (*geometry.shape[:3], path_count),
        )
        path_mask = backend.broadcast_to(
            (backend.min(backend.astype(geometry.visibility, "int64"), axis=-1) == 1)[..., None],
            (*geometry.shape[:3], path_count),
        )
        cir = CIRBatch(coefficients, delays, metadata, path_mask)
        return NTNChannelRealization(cir, geometry, large_scale, los_mask)


def _build(config: object) -> TR38811CDLChannel:
    if not isinstance(config, TR38811CDLConfig):
        raise TypeError("TR 38.811 CDL builder requires TR38811CDLConfig")
    return TR38811CDLChannel(config)


def register_cdl_model(registry: ModelRegistry) -> None:
    """Register ``wireless/tr38811_cdl``."""

    registry.register("wireless", "tr38811_cdl", parse_cdl_config, _build)


__all__ = ["TR38811CDLChannel", "register_cdl_model"]
