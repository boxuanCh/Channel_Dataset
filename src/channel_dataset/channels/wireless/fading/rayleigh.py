"""Rayleigh block-fading CIR generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from channel_dataset.channels.wireless.math import complex_normal
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.config import ConfigEnvelope
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry


@dataclass(frozen=True, slots=True)
class RayleighBlockFadingConfig:
    """Configuration for a zero-delay Rayleigh block-fading model.

    Parameters
    ----------
    num_rx, num_tx : int
        Positive receiver and transmitter counts.
    num_rx_antennas, num_tx_antennas : int
        Positive antenna counts per receiver and transmitter.
    average_power_linear : float
        Positive finite value of :math:`E[|h|^2]` for each MIMO coefficient.
    """

    num_rx: int
    num_rx_antennas: int
    num_tx: int
    num_tx_antennas: int
    average_power_linear: float

    def __post_init__(self) -> None:
        for field_name in ("num_rx", "num_rx_antennas", "num_tx", "num_tx_antennas"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if isinstance(self.average_power_linear, bool) or not isinstance(
            self.average_power_linear, (int, float)
        ):
            raise TypeError("average_power_linear must be numeric")
        if not 0.0 < float(self.average_power_linear) < float("inf"):
            raise ValueError("average_power_linear must be finite and positive")


def parse_rayleigh_config(parameters: Mapping[str, Any]) -> RayleighBlockFadingConfig:
    """Parse strict Rayleigh model parameters.

    Parameters
    ----------
    parameters : mapping
        Mapping containing exactly ``num_rx``, ``num_rx_antennas``, ``num_tx``,
        ``num_tx_antennas``, and ``average_power_linear``.

    Returns
    -------
    RayleighBlockFadingConfig
        Validated immutable configuration.

    Raises
    ------
    ValueError
        If fields are missing/unknown or values are outside their valid ranges.
    TypeError
        If a field has an incompatible type.
    """

    fields = {
        "num_rx", "num_rx_antennas", "num_tx", "num_tx_antennas",
        "average_power_linear",
    }
    unknown = set(parameters) - fields
    if unknown:
        raise ValueError(f"unknown Rayleigh parameters: {sorted(unknown)}")
    missing = fields - set(parameters)
    if missing:
        raise ValueError(f"missing Rayleigh parameters: {sorted(missing)}")
    return RayleighBlockFadingConfig(**{name: parameters[name] for name in fields})


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0.1.0"


class RayleighBlockFading(ChannelModel[ChannelSamplingRequest, CIRBatch]):
    """Sample one zero-delay Rayleigh path held constant over a request block.

    Parameters
    ----------
    config : RayleighBlockFadingConfig
        Typed physical model configuration.

    Notes
    -----
    Each batch/link/antenna coefficient is an independent circular Gaussian
    draw. The single path is broadcast across the time axis, so repeated signal
    application uses the same block realization.
    """

    def __init__(self, config: RayleighBlockFadingConfig) -> None:
        self.config = config
        normalized = asdict(config)
        self._config_digest = ConfigEnvelope(
            schema_version=1,
            family=self.family,
            name=self.name,
            parameters=normalized,
        ).model_digest

    @property
    def family(self) -> str:
        """str: Stable registry family, ``"wireless"``."""

        return "wireless"

    @property
    def name(self) -> str:
        """str: Stable registry name, ``"rayleigh_block_fading"``."""

        return "rayleigh_block_fading"

    def realize(self, request: ChannelSamplingRequest) -> CIRBatch:
        """Sample a replayable Rayleigh block-fading CIR.

        Parameters
        ----------
        request : ChannelSamplingRequest
            Batch/time dimensions, execution context, and root random stream.

        Returns
        -------
        CIRBatch
            One zero-delay path with canonical coefficient axes
            ``[batch, rx, rx_ant, tx, tx_ant, 1, time]``.
        """
        backend = request.execution.array_backend
        config = self.config
        base_shape = (
            request.batch_size,
            config.num_rx,
            config.num_rx_antennas,
            config.num_tx,
            config.num_tx_antennas,
            1,
            1,
        )
        coefficient_stream = request.random_stream.fork("wireless.rayleigh.coefficients")
        base = complex_normal(
            coefficient_stream,
            base_shape,
            variance=float(config.average_power_linear),
            dtype=request.execution.complex_dtype,
        )
        coefficients = backend.broadcast_to(base, base_shape[:-1] + (request.num_time_samples,))
        delays = backend.zeros(
            (request.batch_size, config.num_rx, config.num_tx, 1),
            dtype=request.execution.float_dtype,
            device=request.execution.device,
        )
        metadata = RealizationMetadata(
            model_family=self.family,
            model_name=self.name,
            schema_version=1,
            backend=request.execution.backend,
            precision=request.execution.precision,
            device=request.execution.device,
            stream_id=request.random_stream.stream_id,
            config_digest=self._config_digest,
            resolved_configuration=asdict(config),
            software_version=_software_version(),
            request_id=request.request_id,
        )
        return CIRBatch(coefficients=coefficients, delays_s=delays, metadata=metadata)


def _build_rayleigh(config: object) -> RayleighBlockFading:
    if not isinstance(config, RayleighBlockFadingConfig):
        raise TypeError("Rayleigh builder requires RayleighBlockFadingConfig")
    return RayleighBlockFading(config)


def register_rayleigh_model(registry: ModelRegistry) -> None:
    """Register the Rayleigh parser and builder.

    Parameters
    ----------
    registry : ModelRegistry
        Registry receiving the ``wireless/rayleigh_block_fading`` key.

    Raises
    ------
    ValueError
        If that key is already registered.
    """

    registry.register("wireless", "rayleigh_block_fading", parse_rayleigh_config, _build_rayleigh)
