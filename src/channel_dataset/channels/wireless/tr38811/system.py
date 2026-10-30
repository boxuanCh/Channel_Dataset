"""Frequency-selective system-level TR 38.811 channel assembly."""

from __future__ import annotations

from dataclasses import asdict, replace
from math import log

from channel_dataset.channels.wireless.tr38811.cdl import TR38811CDLChannel
from channel_dataset.channels.wireless.tr38811.config import (
    TR38811CDLConfig,
    TR38811DeploymentConfig,
    TR38811SystemConfig,
    parse_system_config,
)
from channel_dataset.channels.wireless.tr38811.geometry import compute_ntn_geometry
from channel_dataset.channels.wireless.tr38811.large_scale import sample_large_scale, sample_los
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


class TR38811SystemChannel(
    ChannelModel[NTNChannelSamplingRequest, NTNChannelRealization]
):
    """Compose LOS/NLOS NTN CDL kernels with system-level link conditions.

    The model samples clause-6.6 LOS conditions independently per link and
    selects NTN-CDL-C for LOS or NTN-CDL-A for NLOS. The construction retains
    the standard NTN tables and stable per-component streams while exposing a
    multi-link system-level interface.
    """

    def __init__(self, config: TR38811SystemConfig) -> None:
        if not isinstance(config, TR38811SystemConfig):
            raise TypeError("TR38811SystemChannel requires TR38811SystemConfig")
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
        self._resolved = asdict(config)
        self._digest = ConfigEnvelope(1, self.family, self.name, self._resolved).model_digest

    @property
    def family(self) -> str:
        """Return the stable registry family."""

        return "wireless"

    @property
    def name(self) -> str:
        """Return the stable registry model name."""

        return "tr38811_system"

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

    def _cdl_config(self, profile: str) -> TR38811CDLConfig:
        return TR38811CDLConfig(
            profile=profile,
            standard_version=self.config.standard_version,
            deployment=self.config.deployment,
            environment=self.config.environment,
            carrier_frequency_hz=self.config.carrier_frequency_hz,
            bandwidth_hz=self.config.bandwidth_hz,
            delay_spread_s=self.config.delay_spread_s,
            desired_asa_degrees=self.config.desired_asa_degrees,
            desired_zsa_degrees=self.config.desired_zsa_degrees,
            tx_array_profile=self.config.tx_array_profile,
            rx_array_profile=self.config.rx_array_profile,
            apply_large_scale=False,
        )

    def _child_request(
        self, request: NTNChannelSamplingRequest, label: str
    ) -> NTNChannelSamplingRequest:
        sampling = request.sampling
        child = replace(sampling, random_stream=sampling.random_stream.fork(label))
        return NTNChannelSamplingRequest(child, request.topology)

    def realize(self, request: NTNChannelSamplingRequest) -> NTNChannelRealization:
        """Sample independent link conditions and a frequency-selective CIR."""

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
        los_mask = sample_los(
            geometry, request.topology, self.config.environment,
            sampling.random_stream, execution,
        )
        nlos = TR38811CDLChannel(self._cdl_config("NTN-CDL-A")).realize(
            self._child_request(request, "wireless.tr38811.system.nlos")
        )
        los = TR38811CDLChannel(self._cdl_config("NTN-CDL-C")).realize(
            self._child_request(request, "wireless.tr38811.system.los")
        )
        select_coefficients = los_mask[:, :, None, :, None, None, None]
        coefficients = backend.where(
            select_coefficients, los.cir.coefficients, nlos.cir.coefficients
        )
        delays = backend.where(los_mask[..., None], los.cir.delays_s, nlos.cir.delays_s)
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
            amplitude = backend.exp((-log(10.0) / 20.0) * large_scale.total_db)
            coefficients = coefficients * amplitude[:, :, None, :, None, None, :]
        visible_link = backend.min(backend.astype(geometry.visibility, "int64"), axis=-1) == 1
        path_mask = backend.broadcast_to(visible_link[..., None], backend.shape(delays))
        cir = CIRBatch(coefficients, delays, metadata, path_mask)
        diagnostics = NTNRealizationDiagnostics(
            {"kernel_los": "NTN-CDL-C", "kernel_nlos": "NTN-CDL-A"}
        )
        return NTNChannelRealization(cir, geometry, large_scale, los_mask, diagnostics)


def _build(config: object) -> TR38811SystemChannel:
    if not isinstance(config, TR38811SystemConfig):
        raise TypeError("TR 38.811 system builder requires TR38811SystemConfig")
    return TR38811SystemChannel(config)


def register_system_model(registry: ModelRegistry) -> None:
    """Register ``wireless/tr38811_system``."""

    registry.register("wireless", "tr38811_system", parse_system_config, _build)


__all__ = ["TR38811SystemChannel", "register_system_model"]
