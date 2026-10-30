"""IEEE 802.11 TGn channel assembly and shared correlated-tap engine."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from math import cos, log, pi, sin, sqrt
from typing import Any

from channel_dataset.channels.wireless.fading import apply_kronecker_correlation
from channel_dataset.channels.wireless.ieee80211.config import TGnConfig, parse_tgn_config
from channel_dataset.channels.wireless.ieee80211.diagnostics import WLANRealizationDiagnostics
from channel_dataset.channels.wireless.ieee80211.large_scale import (
    sample_link_shadowing,
    tgn_breakpoint_path_loss_db,
)
from channel_dataset.channels.wireless.ieee80211.profiles import load_ieee80211_profile
from channel_dataset.channels.wireless.ieee80211.requests import WLANChannelSamplingRequest
from channel_dataset.channels.wireless.ieee80211.spatial import tap_spatial_covariances
from channel_dataset.channels.wireless.ieee80211.temporal import (
    apply_fluorescent_effect,
    apply_moving_vehicle_effect,
    bell_doppler_process,
)
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0.1.0"


class _WLANCorrelatedTapChannel(ChannelModel[WLANChannelSamplingRequest, CIRBatch]):
    """Internal shared engine for standards layers that retain TGn taps."""

    registry_name = "ieee80211_tgn"
    standard_version = "03/940r4"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.profile = load_ieee80211_profile(config.profile)
        self._resolved_config = asdict(config)
        self._config_digest = ConfigEnvelope(
            1, self.family, self.name, self._resolved_config
        ).model_digest

    @property
    def family(self) -> str:
        """str: Stable model family, ``"wireless"``."""
        return "wireless"

    @property
    def name(self) -> str:
        """str: Stable IEEE 802.11 registry model name."""
        return self.registry_name

    @property
    def maximum_doppler_hz(self) -> float:
        """Return the environmental Bell-spectrum Doppler spread."""
        return (
            self.config.environmental_speed_mps
            * self.config.carrier_frequency_hz
            / SPEED_OF_LIGHT_MPS
        )

    def _delays_s(self) -> tuple[float, ...]:
        return self.profile.delays_s

    def _powers_linear(self) -> tuple[float, ...]:
        raw = self.profile.raw_tap_powers_linear
        if self.config.normalize_profile_power:
            total = sum(raw)
            return tuple(value / total for value in raw)
        return raw

    def _requires_topology(self) -> bool:
        return self.config.large_scale_mode != "none"

    def _validate_request(self, request: WLANChannelSamplingRequest) -> None:
        if not isinstance(request, WLANChannelSamplingRequest):
            raise TypeError("request must be a WLANChannelSamplingRequest")
        topology = request.topology
        if self._requires_topology() and topology is None:
            raise ValueError("this WLAN configuration requires topology")
        if topology is not None:
            expected = (request.sampling.batch_size, self.config.num_rx, self.config.num_tx)
            if topology.shape != expected:
                raise ValueError(f"topology shape must be {expected}")

    def _covariances(self, request: WLANChannelSamplingRequest) -> tuple[Any, Any]:
        execution = request.sampling.execution
        receive = tap_spatial_covariances(
            self.profile,
            self.config.num_rx_antennas,
            self.config.rx_antenna_spacing_wavelengths,
            "rx",
            execution,
            quadrature_order=self.config.spatial_quadrature_order,
        )
        transmit = tap_spatial_covariances(
            self.profile,
            self.config.num_tx_antennas,
            self.config.tx_antenna_spacing_wavelengths,
            "tx",
            execution,
            quadrature_order=self.config.spatial_quadrature_order,
        )
        return receive, transmit

    def _los_matrix(self, backend: Any, execution: Any) -> Any:
        values = []
        angle = pi / 4.0
        for rx in range(self.config.num_rx_antennas):
            row = []
            for tx in range(self.config.num_tx_antennas):
                phase = 2.0 * pi * (
                    rx * self.config.rx_antenna_spacing_wavelengths * cos(angle)
                    + tx * self.config.tx_antenna_spacing_wavelengths * cos(angle)
                )
                row.append(complex(cos(phase), sin(phase)))
            values.append(row)
        return backend.array(values, dtype=execution.complex_dtype, device=execution.device)

    def _link_distances(self, request: WLANChannelSamplingRequest) -> Any:
        assert request.topology is not None
        backend = request.sampling.execution.array_backend
        tx = request.topology.tx_positions_m[:, None, :, :]
        rx = request.topology.rx_positions_m[:, :, None, :]
        return backend.sqrt(backend.sum((rx - tx) * (rx - tx), axis=-1))

    def _path_loss_db(self, request: WLANChannelSamplingRequest) -> Any:
        backend = request.sampling.execution.array_backend
        distance = self._link_distances(request)
        return tgn_breakpoint_path_loss_db(
            distance,
            self.config.carrier_frequency_hz,
            self.profile.breakpoint_distance_m,
            backend,
        )

    def _penetration_loss_db(self, request: WLANChannelSamplingRequest) -> float:
        return 0.0

    def _shadow_standard_deviation_db(self) -> float:
        return (
            self.profile.shadow_std_los_db
            if self.config.line_of_sight
            else self.profile.shadow_std_nlos_db
        )

    def _link_stream_key(
        self, request: WLANChannelSamplingRequest, batch: int, rx: int, tx: int
    ) -> str:
        """Return the stable component-stream key for one logical link."""
        return f"{batch}.{rx}.{tx}"

    def _tap_environmental_speed_mps(self, tap: int) -> float:
        return self.config.environmental_speed_mps

    def _large_scale_amplitude(self, request: WLANChannelSamplingRequest) -> Any | None:
        if self.config.large_scale_mode == "none":
            return None
        sampling = request.sampling
        backend = sampling.execution.array_backend
        path_loss = self._path_loss_db(request)
        if self.config.large_scale_mode == "path_loss_shadowing":
            standard_deviation = self._shadow_standard_deviation_db()
            shadow_parts = []
            for batch in range(sampling.batch_size):
                rx_parts = []
                for rx in range(self.config.num_rx):
                    tx_parts = []
                    for tx in range(self.config.num_tx):
                        key = self._link_stream_key(request, batch, rx, tx)
                        label = f"wireless.ieee80211.{self.standard_version}.link.{key}.shadow_fading"
                        value = sample_link_shadowing(
                            sampling.random_stream.fork(label), (1,), standard_deviation,
                            dtype=sampling.execution.float_dtype,
                        )
                        tx_parts.append(value[0])
                    rx_parts.append(backend.stack(tuple(tx_parts)))
                shadow_parts.append(backend.stack(tuple(rx_parts)))
            path_loss = path_loss + backend.stack(tuple(shadow_parts))
        path_loss = path_loss + self._penetration_loss_db(request)
        return backend.exp(-path_loss * (log(10.0) / 20.0))

    def diagnostics(self, request: WLANChannelSamplingRequest) -> WLANRealizationDiagnostics:
        """Return deterministic diagnostics without consuming random samples."""

        self._validate_request(request)
        receive, transmit = self._covariances(request)
        return WLANRealizationDiagnostics(
            self._delays_s(), self._powers_linear(), receive, transmit,
            self.config.line_of_sight,
            self.profile.los_k_factor_db if self.config.line_of_sight else None,
            self.maximum_doppler_hz,
            self.profile.review_status,
            {
                "diffuse": f"wireless.ieee80211.{self.standard_version}.link.<b>.<rx>.<tx>.tap.<tap>.diffuse",
                "los": f"wireless.ieee80211.{self.standard_version}.link.<b>.<rx>.<tx>.tap.0.los_phase",
                "shadowing": f"wireless.ieee80211.{self.standard_version}.link.<b>.<rx>.<tx>.shadow_fading",
            },
        )

    def _one_link_tap(
        self,
        request: WLANChannelSamplingRequest,
        receive_covariance: Any,
        transmit_covariance: Any,
        tap: int,
        batch: int,
        rx: int,
        tx: int,
    ) -> Any:
        sampling = request.sampling
        execution = sampling.execution
        backend = execution.array_backend
        key = self._link_stream_key(request, batch, rx, tx)
        prefix = f"wireless.ieee80211.{self.standard_version}.link.{key}.tap.{tap}"
        values = bell_doppler_process(
            sampling.random_stream.fork(prefix + ".diffuse"),
            backend,
            (self.config.num_rx_antennas, self.config.num_tx_antennas),
            sampling.num_time_samples,
            sampling.sampling_frequency_hz,
            self.config.carrier_frequency_hz,
            self._tap_environmental_speed_mps(tap),
            num_sinusoids=self.config.num_sinusoids,
            float_dtype=execution.float_dtype,
            complex_dtype=execution.complex_dtype,
            device=execution.device,
        )
        ordered = backend.transpose(values, (2, 0, 1))
        ordered = apply_kronecker_correlation(
            ordered, receive_covariance, transmit_covariance
        )
        values = backend.transpose(ordered, (1, 2, 0))
        if tap == 0 and self.config.line_of_sight:
            k_linear = 10.0 ** (self.profile.los_k_factor_db / 10.0)
            phase = sampling.random_stream.fork(prefix + ".los_phase").uniform(
                (1,), 0.0, 2.0 * pi, dtype=execution.float_dtype
            )
            fixed = self._los_matrix(backend, execution) * backend.exp(1j * phase[0])
            values = values / sqrt(k_linear + 1.0) + (
                sqrt(k_linear / (k_linear + 1.0)) * fixed[..., None]
            )
        fluorescent_taps = {"D": {11, 13, 15}, "E": {2, 4, 6}}
        if self.config.fluorescent_effect and tap in fluorescent_taps[self.config.profile]:
            values = apply_fluorescent_effect(
                values, sampling.random_stream.fork(prefix + ".fluorescent"), backend,
                sampling.sampling_frequency_hz,
                float_dtype=execution.float_dtype, device=execution.device,
            )
        if self.config.moving_vehicle_effect and tap == 2:
            values = apply_moving_vehicle_effect(
                values, sampling.random_stream.fork(prefix + ".moving_vehicle"), backend,
                sampling.sampling_frequency_hz, self.config.carrier_frequency_hz,
                float_dtype=execution.float_dtype, device=execution.device,
            )
        return values

    def realize(self, request: WLANChannelSamplingRequest) -> CIRBatch:
        """Sample a replayable, temporally consistent WLAN CIR window."""

        self._validate_request(request)
        sampling = request.sampling
        execution = sampling.execution
        backend = execution.array_backend
        receive_covariances, transmit_covariances = self._covariances(request)
        powers = self._powers_linear()
        path_values = []
        for tap, power in enumerate(powers):
            batches = []
            for batch in range(sampling.batch_size):
                receivers = []
                for rx in range(self.config.num_rx):
                    transmitters = []
                    for tx in range(self.config.num_tx):
                        value = self._one_link_tap(
                            request, receive_covariances[tap], transmit_covariances[tap],
                            tap, batch, rx, tx,
                        )
                        transmitters.append(sqrt(power) * value)
                    receivers.append(backend.stack(tuple(transmitters), axis=1))
                batches.append(backend.stack(tuple(receivers), axis=0))
            path_values.append(backend.stack(tuple(batches), axis=0))
        coefficients = backend.stack(tuple(path_values), axis=5)
        amplitude = self._large_scale_amplitude(request)
        if amplitude is not None:
            coefficients = coefficients * backend.reshape(
                amplitude, (sampling.batch_size, self.config.num_rx, 1, self.config.num_tx, 1, 1, 1)
            )
        delays_vector = backend.array(
            self._delays_s(), dtype=execution.float_dtype, device=execution.device
        )
        delays = backend.broadcast_to(
            backend.reshape(delays_vector, (1, 1, 1, len(powers))),
            (sampling.batch_size, self.config.num_rx, self.config.num_tx, len(powers)),
        )
        path_mask = None
        if request.topology is not None and request.topology.link_mask is not None:
            path_mask = backend.broadcast_to(
                request.topology.link_mask[..., None],
                (sampling.batch_size, self.config.num_rx, self.config.num_tx, len(powers)),
            )
        metadata = RealizationMetadata(
            self.family, self.name, 1, execution.backend, execution.precision,
            execution.device, sampling.random_stream.stream_id, self._config_digest,
            self._resolved_config, _software_version(), sampling.request_id,
            root_seed=sampling.random_stream.root_seed,
            model_version="1", standard_version=self.standard_version,
        )
        return CIRBatch(coefficients, delays, metadata, path_mask)


class TGnChannel(_WLANCorrelatedTapChannel):
    """IEEE 802.11-03/940r4 TGn A-F SISO/MIMO channel model."""

    def __init__(self, config: TGnConfig) -> None:
        if not isinstance(config, TGnConfig):
            raise TypeError("TGnChannel requires TGnConfig")
        super().__init__(config)


def _build_tgn(config: object) -> TGnChannel:
    if not isinstance(config, TGnConfig):
        raise TypeError("TGn builder requires TGnConfig")
    return TGnChannel(config)


def register_tgn_model(registry: ModelRegistry) -> None:
    """Register ``wireless/ieee80211_tgn`` with a model registry."""
    registry.register("wireless", "ieee80211_tgn", parse_tgn_config, _build_tgn)


__all__ = ["TGnChannel", "register_tgn_model"]
