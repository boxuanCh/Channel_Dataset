"""Drop-based two-state land-mobile-satellite flat fading."""

from __future__ import annotations

from dataclasses import asdict
from math import log, pi, sqrt

from channel_dataset.channels.wireless.tr38811.config import (
    TR38811DeploymentConfig,
    TR38811FlatFadingConfig,
    parse_flat_config,
)
from channel_dataset.channels.wireless.tr38811.geometry import compute_ntn_geometry, doppler_phase
from channel_dataset.channels.wireless.tr38811.large_scale import sample_large_scale
from channel_dataset.channels.wireless.tr38811.profiles import (
    validate_deployment_applicability,
    validate_topology_applicability,
)
from channel_dataset.channels.wireless.tr38811.tdl import _software_version
from channel_dataset.channels.wireless.tr38811.types import (
    NTNChannelRealization,
    NTNChannelSamplingRequest,
    NTNRealizationDiagnostics,
)
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry


class TR38811FlatFadingChannel(
    ChannelModel[NTNChannelSamplingRequest, NTNChannelRealization]
):
    """Sample one GOOD/BAD Loo fading state per link and request window.

    State-specific Loo triplets are explicit configuration inputs because the
    applicable ITU-R P.681 table depends on environment and link conditions.
    The GOOD/BAD state is not interpreted as the clause-6.6 LOS/NLOS state.
    """

    def __init__(self, config: TR38811FlatFadingConfig) -> None:
        if not isinstance(config, TR38811FlatFadingConfig):
            raise TypeError("TR38811FlatFadingChannel requires TR38811FlatFadingConfig")
        self.config = config
        validate_deployment_applicability(
            TR38811DeploymentConfig(
                deployment=config.deployment,
                standard_version=config.standard_version,
                environment=config.environment,
                carrier_frequency_hz=config.carrier_frequency_hz,
                bandwidth_hz=config.bandwidth_hz,
            ),
            model_kind="flat",
        )
        self._resolved = asdict(config)
        self._digest = ConfigEnvelope(1, self.family, self.name, self._resolved).model_digest

    @property
    def family(self) -> str:
        """Return the stable registry family."""

        return "wireless"

    @property
    def name(self) -> str:
        """Return the stable registry model name."""

        return "tr38811_flat"

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

    def realize(self, request: NTNChannelSamplingRequest) -> NTNChannelRealization:
        """Sample a fixed-state Loo fading window."""

        validate_topology_applicability(request.topology, self.config.deployment)
        sampling = request.sampling
        execution = sampling.execution
        backend = execution.array_backend
        metadata = self._metadata(request)
        geometry = compute_ntn_geometry(
            request.topology, sampling.num_time_samples,
            self.config.carrier_frequency_hz, metadata=metadata,
        )
        link_shape = geometry.shape[:3]
        if self.config.state_mode == "sample":
            state_draw = sampling.random_stream.fork("wireless.tr38811.flat.state").uniform(
                link_shape, dtype=execution.float_dtype
            )
            good_state = state_draw < self.config.good_state_probability
        else:
            good_state = backend.full(
                link_shape, self.config.state_mode == "good",
                dtype="bool", device=execution.device,
            )
        direct_mean = backend.where(
            good_state,
            backend.full(link_shape, self.config.good_direct_mean_db, dtype=execution.float_dtype, device=execution.device),
            backend.full(link_shape, self.config.bad_direct_mean_db, dtype=execution.float_dtype, device=execution.device),
        )
        direct_std = backend.where(
            good_state,
            backend.full(link_shape, self.config.good_direct_std_db, dtype=execution.float_dtype, device=execution.device),
            backend.full(link_shape, self.config.bad_direct_std_db, dtype=execution.float_dtype, device=execution.device),
        )
        multipath_db = backend.where(
            good_state,
            backend.full(link_shape, self.config.good_multipath_power_db, dtype=execution.float_dtype, device=execution.device),
            backend.full(link_shape, self.config.bad_multipath_power_db, dtype=execution.float_dtype, device=execution.device),
        )
        normal = sampling.random_stream.fork("wireless.tr38811.flat.direct_shadow").normal(
            link_shape, dtype=execution.float_dtype
        )
        direct_amplitude = backend.exp(
            (log(10.0) / 20.0) * (direct_mean + direct_std * normal)
        )
        direct_phase = sampling.random_stream.fork("wireless.tr38811.flat.direct_phase").uniform(
            link_shape, -pi, pi, dtype=execution.float_dtype
        )
        antenna_shape = (
            geometry.shape[0], geometry.shape[1], self.config.num_rx_antennas,
            geometry.shape[2], self.config.num_tx_antennas,
        )
        oscillator_shape = (*antenna_shape, self.config.num_sinusoids)
        angles = sampling.random_stream.fork("wireless.tr38811.flat.doppler_angles").uniform(
            oscillator_shape, 0.0, 2.0 * pi, dtype=execution.float_dtype
        )
        phases = sampling.random_stream.fork("wireless.tr38811.flat.diffuse_phase").uniform(
            oscillator_shape, -pi, pi, dtype=execution.float_dtype
        )
        time = backend.arange(
            sampling.num_time_samples, dtype=execution.float_dtype, device=execution.device
        ) / sampling.sampling_frequency_hz
        maximum_doppler = (
            self.config.terminal_speed_mps * self.config.carrier_frequency_hz / 299_792_458.0
        )
        diffuse_unit = backend.sum(
            backend.exp(
                1j * (
                    phases[..., None]
                    + 2.0 * pi * maximum_doppler * backend.cos(angles)[..., None] * time
                )
            ),
            axis=-2,
        ) / sqrt(self.config.num_sinusoids)
        multipath_power = backend.exp((log(10.0) / 10.0) * multipath_db)
        diffuse = diffuse_unit * backend.sqrt(multipath_power)[:, :, None, :, None, None]
        direct = (
            direct_amplitude * backend.exp(1j * direct_phase)
        )[:, :, None, :, None, None]
        coefficients = direct + diffuse
        common = doppler_phase(
            geometry.common_doppler_hz, sampling.sampling_frequency_hz, execution
        )
        coefficients = coefficients * common[:, :, None, :, None, :]
        coefficients = coefficients[..., None, :]

        false_los = backend.zeros(link_shape, dtype="bool", device=execution.device)
        deployment = TR38811DeploymentConfig(
            self.config.deployment, self.config.standard_version, self.config.environment,
            self.config.carrier_frequency_hz, self.config.bandwidth_hz,
            apply_clutter_loss=False, apply_shadow_fading=False, apply_o2i_loss=False,
            apply_atmospheric_loss=False, apply_scintillation=False,
        )
        large_scale = sample_large_scale(
            geometry, request.topology, false_los, deployment,
            sampling.random_stream, execution, metadata,
        )
        if self.config.apply_large_scale:
            amplitude = backend.exp((-log(10.0) / 20.0) * large_scale.total_db)
            coefficients = coefficients * amplitude[:, :, None, :, None, None, :]
        coefficients = backend.where(
            geometry.visibility[:, :, None, :, None, None, :],
            coefficients,
            coefficients * 0.0,
        )
        delays = backend.zeros(
            (*link_shape, 1), dtype=execution.float_dtype, device=execution.device
        )
        path_mask = backend.min(backend.astype(geometry.visibility, "int64"), axis=-1) == 1
        cir = CIRBatch(coefficients, delays, metadata, path_mask[..., None])
        diagnostics = NTNRealizationDiagnostics({"good_state": good_state})
        return NTNChannelRealization(cir, geometry, large_scale, false_los, diagnostics)


def _build(config: object) -> TR38811FlatFadingChannel:
    if not isinstance(config, TR38811FlatFadingConfig):
        raise TypeError("TR 38.811 flat builder requires TR38811FlatFadingConfig")
    return TR38811FlatFadingChannel(config)


def register_flat_model(registry: ModelRegistry) -> None:
    """Register ``wireless/tr38811_flat``."""

    registry.register("wireless", "tr38811_flat", parse_flat_config, _build)


__all__ = ["TR38811FlatFadingChannel", "register_flat_model"]
