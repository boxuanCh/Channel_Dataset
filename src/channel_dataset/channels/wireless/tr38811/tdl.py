"""3GPP TR 38.811 NTN tapped-delay-line channel models."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from math import pi, sqrt

from channel_dataset.channels.wireless.tr38811.config import (
    TR38811DeploymentConfig,
    TR38811TDLConfig,
    parse_tdl_config,
)
from channel_dataset.channels.wireless.tr38811.geometry import compute_ntn_geometry, doppler_phase
from channel_dataset.channels.wireless.tr38811.large_scale import sample_large_scale
from channel_dataset.channels.wireless.tr38811.profiles import (
    load_tdl_profile,
    validate_deployment_applicability,
    validate_topology_applicability,
)
from channel_dataset.channels.wireless.tr38811.types import (
    NTNChannelRealization,
    NTNChannelSamplingRequest,
)
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0.1.0"


class TR38811TDLChannel(ChannelModel[NTNChannelSamplingRequest, NTNChannelRealization]):
    """Sample NTN-TDL-A/B/C/D with terminal and platform Doppler."""

    def __init__(self, config: TR38811TDLConfig) -> None:
        if not isinstance(config, TR38811TDLConfig):
            raise TypeError("TR38811TDLChannel requires TR38811TDLConfig")
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
        self.profile = load_tdl_profile(config.profile)
        self._resolved = asdict(config)
        self._digest = ConfigEnvelope(1, self.family, self.name, self._resolved).model_digest

    @property
    def family(self) -> str:
        """Return the stable registry family."""

        return "wireless"

    @property
    def name(self) -> str:
        """Return the stable registry model name."""

        return "tr38811_tdl"

    def _delays_and_powers(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        delays = [value * self.config.delay_spread_s for value in self.profile.delays]
        raw = [10.0 ** (value / 10.0) for value in self.profile.powers_db]
        powers = list(raw)
        k_factor = self.config.k_factor_db
        if k_factor is not None:
            los_index = self.profile.component_types.index("los")
            diffuse_sum = sum(
                power
                for power, kind in zip(raw, self.profile.component_types, strict=True)
                if kind == "rayleigh"
            )
            k_linear = 10.0 ** (k_factor / 10.0)
            for index, kind in enumerate(self.profile.component_types):
                powers[index] = (
                    k_linear / (k_linear + 1.0)
                    if kind == "los"
                    else raw[index] / diffuse_sum / (k_linear + 1.0)
                )
            mean = sum(power * delay for power, delay in zip(powers, delays, strict=True))
            rms = sqrt(
                sum(
                    power * (delay - mean) ** 2
                    for power, delay in zip(powers, delays, strict=True)
                )
            )
            if rms <= 0.0 or powers[los_index] <= 0.0:
                raise ValueError("K-factor adjustment produced an invalid profile")
            delays = [delay * self.config.delay_spread_s / rms for delay in delays]
        else:
            total = sum(powers)
            powers = [value / total for value in powers]
        return tuple(delays), tuple(powers)

    def _metadata(self, request: NTNChannelSamplingRequest) -> RealizationMetadata:
        sampling = request.sampling
        return RealizationMetadata(
            self.family,
            self.name,
            1,
            sampling.execution.backend,
            sampling.execution.precision,
            sampling.execution.device,
            sampling.random_stream.stream_id,
            self._digest,
            self._resolved,
            _software_version(),
            sampling.request_id,
            root_seed=sampling.random_stream.root_seed,
            model_version="1",
            standard_version="15.4.0",
        )

    def realize(self, request: NTNChannelSamplingRequest) -> NTNChannelRealization:
        """Sample one fixed-window NTN channel realization."""

        if not isinstance(request, NTNChannelSamplingRequest):
            raise TypeError("request must be NTNChannelSamplingRequest")
        validate_topology_applicability(request.topology, self.config.deployment)
        sampling = request.sampling
        execution = sampling.execution
        backend = execution.array_backend
        metadata = self._metadata(request)
        geometry = compute_ntn_geometry(
            request.topology,
            sampling.num_time_samples,
            self.config.carrier_frequency_hz,
            metadata=metadata,
        )
        batch, num_rx, num_tx, time_count = geometry.shape
        topology_shape = request.topology.shape
        if topology_shape != (batch, num_rx, num_tx):
            raise ValueError("topology link dimensions are inconsistent")
        delays_s, powers = self._delays_and_powers()
        path_count = len(powers)
        antenna_shape = (
            batch, num_rx, self.config.num_rx_antennas,
            num_tx, self.config.num_tx_antennas,
        )
        oscillator_shape = (*antenna_shape, path_count, self.config.num_sinusoids)
        angles = sampling.random_stream.fork("wireless.tr38811.tdl.doppler_angles").uniform(
            oscillator_shape, 0.0, 2.0 * pi, dtype=execution.float_dtype
        )
        phases = sampling.random_stream.fork("wireless.tr38811.tdl.initial_phases").uniform(
            oscillator_shape, 0.0, 2.0 * pi, dtype=execution.float_dtype
        )
        time = backend.arange(
            time_count, dtype=execution.float_dtype, device=execution.device
        ) / sampling.sampling_frequency_hz
        maximum_doppler = (
            self.config.terminal_speed_mps * self.config.carrier_frequency_hz / 299_792_458.0
        )
        terminal_doppler = maximum_doppler * backend.cos(angles)
        diffuse = backend.sum(
            backend.exp(
                1j * (
                    phases[..., None]
                    + 2.0 * pi * terminal_doppler[..., None] * time
                )
            ),
            axis=-2,
        ) / sqrt(self.config.num_sinusoids)
        los_phase = sampling.random_stream.fork("wireless.tr38811.tdl.los_phase").uniform(
            antenna_shape, 0.0, 2.0 * pi, dtype=execution.float_dtype
        )
        paths = []
        for index, (power, kind) in enumerate(
            zip(powers, self.profile.component_types, strict=True)
        ):
            value = (
                backend.exp(1j * (los_phase[..., None] + 2.0 * pi * 0.7 * maximum_doppler * time))
                if kind == "los"
                else diffuse[..., index, :]
            )
            paths.append(sqrt(power) * value)
        coefficients = backend.stack(tuple(paths), axis=5)
        common_phase = doppler_phase(
            geometry.common_doppler_hz, sampling.sampling_frequency_hz, execution
        )
        coefficients = coefficients * common_phase[:, :, None, :, None, None, :]

        los_value = self.config.profile in {"NTN-TDL-C", "NTN-TDL-D"}
        los_mask = backend.full(
            (batch, num_rx, num_tx), los_value, dtype="bool", device=execution.device
        )
        if request.topology.los_mask is not None:
            mismatch = backend.sum(
                backend.astype(request.topology.los_mask != los_mask, "int64")
            )
            if int(backend.to_scalar(mismatch)) != 0:
                raise ValueError("topology LOS state is incompatible with the selected TDL profile")

        deployment = TR38811DeploymentConfig(
            self.config.deployment,
            self.config.standard_version,
            self.config.environment,
            self.config.carrier_frequency_hz,
            self.config.bandwidth_hz,
            apply_clutter_loss=self.config.apply_large_scale,
            apply_shadow_fading=self.config.apply_large_scale,
            apply_o2i_loss=self.config.apply_large_scale and self.config.deployment == "D5",
            apply_atmospheric_loss=False,
            apply_scintillation=False,
        )
        large_scale = sample_large_scale(
            geometry,
            request.topology,
            los_mask,
            deployment,
            sampling.random_stream,
            execution,
            metadata,
        )
        if self.config.apply_large_scale:
            amplitude = backend.exp((-__import__("math").log(10.0) / 20.0) * large_scale.total_db)
            coefficients = coefficients * amplitude[:, :, None, :, None, None, :]
        visibility = geometry.visibility[:, :, None, :, None, None, :]
        coefficients = backend.where(visibility, coefficients, coefficients * 0.0)

        delay_vector = backend.array(
            delays_s, dtype=execution.float_dtype, device=execution.device
        )
        delays = backend.broadcast_to(
            backend.reshape(delay_vector, (1, 1, 1, path_count)),
            (batch, num_rx, num_tx, path_count),
        )
        visible_link = backend.min(backend.astype(geometry.visibility, "int64"), axis=-1) == 1
        path_mask = backend.broadcast_to(
            visible_link[..., None], (batch, num_rx, num_tx, path_count)
        )
        cir = CIRBatch(coefficients, delays, metadata, path_mask)
        return NTNChannelRealization(cir, geometry, large_scale, los_mask)


def _build(config: object) -> TR38811TDLChannel:
    if not isinstance(config, TR38811TDLConfig):
        raise TypeError("TR 38.811 TDL builder requires TR38811TDLConfig")
    return TR38811TDLChannel(config)


def register_tdl_model(registry: ModelRegistry) -> None:
    """Register ``wireless/tr38811_tdl``."""

    registry.register("wireless", "tr38811_tdl", parse_tdl_config, _build)


__all__ = ["TR38811TDLChannel", "register_tdl_model"]
