"""Configurable flat MIMO fading generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from typing import Any

from channel_dataset.channels.wireless.fading.correlation import (
    apply_kronecker_correlation,
    apply_per_column_receive_correlation,
)
from channel_dataset.channels.wireless.math import (
    complex_normal,
    exponential_correlation,
)
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import FlatChannelResponse
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry

_FIELDS = {
    "num_rx",
    "num_rx_antennas",
    "num_tx",
    "num_tx_antennas",
    "average_power_linear",
    "power_normalization",
    "correlation_mode",
    "rx_correlation_coefficient",
    "tx_correlation_coefficient",
    "per_column_rx_correlation_coefficients",
}


@dataclass(frozen=True, slots=True)
class FlatMIMOFadingConfig:
    """Configuration for flat Rayleigh MIMO fading.

    Parameters
    ----------
    num_rx, num_tx : int
        Positive receiver and transmitter counts.
    num_rx_antennas, num_tx_antennas : int
        Positive antenna counts per endpoint.
    average_power_linear : float
        Positive target average power.
    power_normalization : {"per_coefficient", "per_link"}
        ``"per_coefficient"`` gives every matrix entry the configured average
        power. ``"per_link"`` gives each link matrix that expected total
        Frobenius power.
    correlation_mode : {"none", "kronecker", "per_column_receive"}
        Spatial-correlation transform.
    rx_correlation_coefficient, tx_correlation_coefficient : float, optional
        Real exponential coefficients required only for ``"kronecker"``.
    per_column_rx_correlation_coefficients : tuple of float, optional
        One real exponential receive coefficient per transmit antenna, required
        only for ``"per_column_receive"``.
    """

    num_rx: int
    num_rx_antennas: int
    num_tx: int
    num_tx_antennas: int
    average_power_linear: float
    power_normalization: str
    correlation_mode: str
    rx_correlation_coefficient: float | None
    tx_correlation_coefficient: float | None
    per_column_rx_correlation_coefficients: tuple[float, ...] | None

    def __post_init__(self) -> None:
        for field_name in ("num_rx", "num_rx_antennas", "num_tx", "num_tx_antennas"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if isinstance(self.average_power_linear, bool) or not isinstance(
            self.average_power_linear, (int, float)
        ):
            raise TypeError("average_power_linear must be numeric")
        if not isfinite(float(self.average_power_linear)) or self.average_power_linear <= 0:
            raise ValueError("average_power_linear must be finite and positive")
        if self.power_normalization not in {"per_coefficient", "per_link"}:
            raise ValueError("power_normalization must be 'per_coefficient' or 'per_link'")
        if self.correlation_mode not in {"none", "kronecker", "per_column_receive"}:
            raise ValueError("unsupported correlation_mode")
        self._validate_correlation_fields()

    @staticmethod
    def _validate_coefficient(value: Any, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a real number")
        if not isfinite(float(value)) or abs(float(value)) > 1.0:
            raise ValueError(f"{field_name} must be finite with magnitude at most one")

    def _validate_correlation_fields(self) -> None:
        if self.correlation_mode == "none":
            if any(
                value is not None
                for value in (
                    self.rx_correlation_coefficient,
                    self.tx_correlation_coefficient,
                    self.per_column_rx_correlation_coefficients,
                )
            ):
                raise ValueError("uncorrelated mode requires all correlation fields to be null")
        elif self.correlation_mode == "kronecker":
            if self.rx_correlation_coefficient is None or self.tx_correlation_coefficient is None:
                raise ValueError("kronecker mode requires receive and transmit coefficients")
            if self.per_column_rx_correlation_coefficients is not None:
                raise ValueError("kronecker mode does not accept per-column coefficients")
            self._validate_coefficient(
                self.rx_correlation_coefficient, "rx_correlation_coefficient"
            )
            self._validate_coefficient(
                self.tx_correlation_coefficient, "tx_correlation_coefficient"
            )
        else:
            if self.rx_correlation_coefficient is not None or self.tx_correlation_coefficient is not None:
                raise ValueError("per-column mode does not accept shared correlation coefficients")
            values = self.per_column_rx_correlation_coefficients
            if values is None or len(values) != self.num_tx_antennas:
                raise ValueError("per-column coefficients must match num_tx_antennas")
            for value in values:
                self._validate_coefficient(value, "per_column_rx_correlation_coefficients")


def parse_flat_mimo_config(parameters: Mapping[str, Any]) -> FlatMIMOFadingConfig:
    """Parse strict flat-MIMO model parameters into a typed configuration."""

    unknown = set(parameters) - _FIELDS
    if unknown:
        raise ValueError(f"unknown flat-MIMO parameters: {sorted(unknown)}")
    missing = _FIELDS - set(parameters)
    if missing:
        raise ValueError(f"missing flat-MIMO parameters: {sorted(missing)}")
    column_values = parameters["per_column_rx_correlation_coefficients"]
    if column_values is not None:
        if not isinstance(column_values, (list, tuple)):
            raise TypeError("per-column correlation coefficients must be a sequence or null")
        column_values = tuple(column_values)
    values = {field_name: parameters[field_name] for field_name in _FIELDS}
    values["per_column_rx_correlation_coefficients"] = column_values
    return FlatMIMOFadingConfig(**values)


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0.1.0"


class FlatMIMOFading(ChannelModel[ChannelSamplingRequest, FlatChannelResponse]):
    """Sample a replayable flat Rayleigh MIMO response."""

    def __init__(self, config: FlatMIMOFadingConfig) -> None:
        self.config = config
        self._resolved_config = asdict(config)
        self._config_digest = ConfigEnvelope(
            schema_version=1,
            family=self.family,
            name=self.name,
            parameters=self._resolved_config,
        ).model_digest

    @property
    def family(self) -> str:
        """str: Stable registry family, ``"wireless"``."""

        return "wireless"

    @property
    def name(self) -> str:
        """str: Stable registry name, ``"flat_mimo_fading"``."""

        return "flat_mimo_fading"

    def realize(self, request: ChannelSamplingRequest) -> FlatChannelResponse:
        """Sample one flat matrix for every batch entry and link."""

        config = self.config
        shape = (
            request.batch_size,
            config.num_rx,
            config.num_rx_antennas,
            config.num_tx,
            config.num_tx_antennas,
        )
        variance = float(config.average_power_linear)
        if config.power_normalization == "per_link":
            variance /= config.num_rx_antennas * config.num_tx_antennas
        stream = request.random_stream.fork("wireless.flat_mimo.coefficients")
        coefficients = complex_normal(
            stream, shape, variance=variance, dtype=request.execution.complex_dtype
        )
        coefficients = self._apply_correlation(coefficients, request)
        metadata = RealizationMetadata(
            model_family=self.family,
            model_name=self.name,
            schema_version=1,
            backend=request.execution.backend,
            precision=request.execution.precision,
            device=request.execution.device,
            stream_id=request.random_stream.stream_id,
            config_digest=self._config_digest,
            resolved_configuration=self._resolved_config,
            software_version=_software_version(),
            request_id=request.request_id,
        )
        return FlatChannelResponse(coefficients, metadata)

    def _apply_correlation(self, coefficients: Any, request: ChannelSamplingRequest) -> Any:
        config = self.config
        if config.correlation_mode == "none":
            return coefficients
        backend = request.execution.array_backend
        # Canonical flat-response axes place ``tx`` between the antenna axes.
        # Correlation operates link-wise on final matrix axes [rx_ant, tx_ant].
        link_matrices = backend.transpose(coefficients, (0, 1, 3, 2, 4))
        if config.correlation_mode == "kronecker":
            receive = exponential_correlation(
                config.num_rx_antennas,
                config.rx_correlation_coefficient,
                request.execution,
            )
            transmit = exponential_correlation(
                config.num_tx_antennas,
                config.tx_correlation_coefficient,
                request.execution,
            )
            correlated = apply_kronecker_correlation(link_matrices, receive, transmit)
            return backend.transpose(correlated, (0, 1, 3, 2, 4))
        matrices = tuple(
            exponential_correlation(config.num_rx_antennas, value, request.execution)
            for value in config.per_column_rx_correlation_coefficients or ()
        )
        stacked = backend.stack(matrices, axis=0)
        correlated = apply_per_column_receive_correlation(link_matrices, stacked)
        return backend.transpose(correlated, (0, 1, 3, 2, 4))


def _build_flat_mimo(config: object) -> FlatMIMOFading:
    if not isinstance(config, FlatMIMOFadingConfig):
        raise TypeError("flat-MIMO builder requires FlatMIMOFadingConfig")
    return FlatMIMOFading(config)


def register_flat_mimo_model(registry: ModelRegistry) -> None:
    """Register the ``wireless/flat_mimo_fading`` model."""

    registry.register("wireless", "flat_mimo_fading", parse_flat_mimo_config, _build_flat_mimo)
