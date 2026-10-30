"""3GPP tapped-delay-line channel realization generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from math import pi, sqrt
from typing import Any

from channel_dataset.channels.wireless.fading import apply_kronecker_correlation
from channel_dataset.channels.wireless.math import (
    SPEED_OF_LIGHT_MPS,
    exponential_correlation,
    matrix_sqrt_psd,
)
from channel_dataset.channels.wireless.requests import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901.config import (
    FIXED_PROFILES,
    SCALABLE_PROFILES,
    TDLConfig,
    TDLProfile,
)
from channel_dataset.channels.wireless.types import CIRBatch
from channel_dataset.config import ConfigEnvelope, load_resource_document
from channel_dataset.core import ChannelModel, RealizationMetadata
from channel_dataset.registry import ModelRegistry

_PROFILE_RESOURCES = {
    "TDL-A": "standards/wireless/tr_38_901/v16_1/tdl_a.yaml",
    "TDL-B": "standards/wireless/tr_38_901/v16_1/tdl_b.yaml",
    "TDL-C": "standards/wireless/tr_38_901/v16_1/tdl_c.yaml",
    "TDL-D": "standards/wireless/tr_38_901/v16_1/tdl_d.yaml",
    "TDL-E": "standards/wireless/tr_38_901/v16_1/tdl_e.yaml",
    "TDLA30": "standards/wireless/ts_38_101_4/v16_1/tdla30.yaml",
    "TDLB100": "standards/wireless/ts_38_101_4/v16_1/tdlb100.yaml",
    "TDLC300": "standards/wireless/ts_38_101_4/v16_1/tdlc300.yaml",
    "TDLC60": "standards/wireless/ts_38_101_4/v16_1/tdlc60.yaml",
}


def load_tdl_profile(name: str) -> TDLProfile:
    """Load and validate a packaged TDL standard resource.

    Parameters
    ----------
    name : str
        Standard profile name: ``TDL-A`` through ``TDL-E``, ``TDLA30``,
        ``TDLB100``, ``TDLC300``, or ``TDLC60``.

    Returns
    -------
    TDLProfile
        Immutable resource with source metadata and tap parameters.

    Raises
    ------
    ValueError
        If the name or resource fields are invalid.
    """

    try:
        reference = _PROFILE_RESOURCES[name]
    except KeyError as error:
        raise ValueError(f"unsupported TDL profile: {name!r}") from error
    document = load_resource_document(reference)
    expected = {
        "schema_version", "name", "standard", "standard_version", "table",
        "scalable", "delays", "powers_db", "component_types",
        "minimum_frequency_hz", "maximum_frequency_hz", "maximum_bandwidth_hz",
        "source_url", "transcription_date", "review_status",
    }
    unknown = set(document) - expected
    missing = expected - set(document)
    if unknown or missing:
        raise ValueError(
            f"invalid TDL resource fields; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if document["schema_version"] != 1 or document["name"] != name:
        raise ValueError("TDL resource schema version or name is invalid")
    return TDLProfile(
        name=document["name"],
        standard=document["standard"],
        standard_version=document["standard_version"],
        table=document["table"],
        scalable=document["scalable"],
        delays=tuple(float(value) for value in document["delays"]),
        powers_db=tuple(float(value) for value in document["powers_db"]),
        component_types=tuple(document["component_types"]),
        minimum_frequency_hz=float(document["minimum_frequency_hz"]),
        maximum_frequency_hz=float(document["maximum_frequency_hz"]),
        maximum_bandwidth_hz=(
            None
            if document["maximum_bandwidth_hz"] is None
            else float(document["maximum_bandwidth_hz"])
        ),
        source_url=document["source_url"],
        transcription_date=document["transcription_date"],
        review_status=document["review_status"],
    )


def parse_tdl_config(parameters: Mapping[str, Any]) -> TDLConfig:
    """Parse a strict typed TDL model configuration.

    Parameters
    ----------
    parameters : mapping
        Mapping containing exactly the public :class:`TDLConfig` fields.

    Returns
    -------
    TDLConfig
        Validated immutable model configuration.
    """

    fields = set(TDLConfig.__dataclass_fields__)
    unknown = set(parameters) - fields
    missing = fields - set(parameters)
    if unknown:
        raise ValueError(f"unknown TDL parameters: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing TDL parameters: {sorted(missing)}")
    values = dict(parameters)
    joint = values["joint_correlation_matrix"]
    if joint is not None:
        if not isinstance(joint, (list, tuple)):
            raise TypeError("joint_correlation_matrix must be a sequence of rows")
        values["joint_correlation_matrix"] = tuple(tuple(row) for row in joint)
    return TDLConfig(**values)


def _software_version() -> str:
    try:
        return version("channel-dataset")
    except PackageNotFoundError:
        return "0.1.0"


class TappedDelayLine(ChannelModel[ChannelSamplingRequest, CIRBatch]):
    """Sample a 3GPP TDL channel with classical Doppler evolution.

    Parameters
    ----------
    config : TDLConfig
        Typed model parameters. Standard tap values are loaded from the
        versioned resource selected by ``config.profile``.

    Notes
    -----
    Diffuse components use a finite sum-of-sinusoids approximation to the
    classical Jakes Doppler spectrum. TDL-D/E include a specular component at
    ``0.7 * maximum_doppler_hz``. Every stochastic component uses a stable
    child stream and no global random state.
    """

    def __init__(self, config: TDLConfig) -> None:
        self.config = config
        self.profile = load_tdl_profile(config.profile)
        self._validate_applicability()
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
        """str: Stable registry name, ``"tdl"``."""

        return "tdl"

    @property
    def maximum_doppler_hz(self) -> float:
        """Return speed-derived maximum Doppler shift in hertz."""

        return self.config.speed_mps * self.config.carrier_frequency_hz / SPEED_OF_LIGHT_MPS

    @property
    def path_delays_s(self) -> tuple[float, ...]:
        """Return resolved physical component delays in seconds."""

        return self._delays_and_powers()[0]

    @property
    def path_powers_linear(self) -> tuple[float, ...]:
        """Return resolved component powers normalized to unit total power."""

        return self._delays_and_powers()[1]

    def _validate_applicability(self) -> None:
        config = self.config
        profile = self.profile
        if config.standard_version != profile.standard_version:
            raise ValueError("configuration and TDL resource versions do not match")
        if not profile.minimum_frequency_hz <= config.carrier_frequency_hz <= profile.maximum_frequency_hz:
            raise ValueError("carrier_frequency_hz is outside the profile applicability range")
        if profile.maximum_bandwidth_hz is not None and config.bandwidth_hz > profile.maximum_bandwidth_hz:
            raise ValueError("bandwidth_hz exceeds the profile applicability limit")
        if config.profile in {"TDLB100", "TDLC300"} and config.carrier_frequency_hz >= 6e9:
            raise ValueError(f"{config.profile} is an FR1 profile and requires carrier below 6 GHz")
        if config.profile == "TDLC60" and config.carrier_frequency_hz < 6e9:
            raise ValueError("TDLC60 is an FR2 profile and requires carrier at or above 6 GHz")
        if profile.scalable != (config.profile in SCALABLE_PROFILES):
            raise ValueError("TDL resource scalability does not match its profile family")
        if config.profile in FIXED_PROFILES and profile.has_los:
            raise ValueError("fixed conformance profiles must contain only Rayleigh taps")

    def _delays_and_powers(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        raw = tuple(10.0 ** (value / 10.0) for value in self.profile.powers_db)
        powers = list(raw)
        delays = list(self.profile.delays)
        if self.config.k_factor_db is not None:
            los_index = self.profile.component_types.index("los")
            diffuse_sum = sum(
                value
                for value, kind in zip(raw, self.profile.component_types, strict=True)
                if kind == "rayleigh"
            )
            k_linear = 10.0 ** (self.config.k_factor_db / 10.0)
            for index, kind in enumerate(self.profile.component_types):
                if kind == "los":
                    powers[index] = k_linear / (k_linear + 1.0)
                else:
                    powers[index] = raw[index] / diffuse_sum / (k_linear + 1.0)
            mean = sum(power * delay for power, delay in zip(powers, delays, strict=True))
            rms = sqrt(
                sum(
                    power * (delay - mean) ** 2
                    for power, delay in zip(powers, delays, strict=True)
                )
            )
            if rms <= 0.0 or powers[los_index] <= 0.0:
                raise ValueError("K-factor adjustment produced an invalid delay profile")
            delays = [delay / rms for delay in delays]
        else:
            total = sum(powers)
            powers = [value / total for value in powers]
        if self.profile.scalable:
            scale = float(self.config.delay_spread_s)
            delays = [value * scale for value in delays]
        return tuple(delays), tuple(powers)

    def _apply_spatial_correlation(
        self, values: Any, backend: Any, execution: Any
    ) -> Any:
        config = self.config
        if config.correlation_mode == "none":
            return values
        b, r, ra, t, ta, p, time = backend.shape(values)
        ordered = backend.transpose(values, (0, 1, 3, 5, 6, 2, 4))
        if config.correlation_mode == "kronecker":
            receive = exponential_correlation(
                ra, float(config.rx_correlation_coefficient), execution
            )
            transmit = exponential_correlation(
                ta, float(config.tx_correlation_coefficient), execution
            )
            correlated = apply_kronecker_correlation(ordered, receive, transmit)
        else:
            matrix = backend.array(
                config.joint_correlation_matrix,
                dtype=execution.float_dtype,
                device=execution.device,
            )
            root = backend.astype(matrix_sqrt_psd(matrix), execution.complex_dtype)
            flattened = backend.reshape(ordered, (b, r, t, p, time, ra * ta))
            correlated = backend.einsum("ij,...j->...i", root, flattened)
            correlated = backend.reshape(correlated, (b, r, t, p, time, ra, ta))
        return backend.transpose(correlated, (0, 1, 5, 2, 6, 3, 4))

    def realize(self, request: ChannelSamplingRequest) -> CIRBatch:
        """Sample a replayable batched TDL realization.

        Parameters
        ----------
        request : ChannelSamplingRequest
            Batch/time dimensions, coefficient sampling frequency, execution
            context, and injected random stream.

        Returns
        -------
        CIRBatch
            Coefficients with canonical axes and standard-defined path delays.
        """

        backend = request.execution.array_backend
        config = self.config
        delays_s, powers = self._delays_and_powers()
        paths = len(delays_s)
        antenna_shape = (
            request.batch_size,
            config.num_rx,
            config.num_rx_antennas,
            config.num_tx,
            config.num_tx_antennas,
        )
        oscillator_shape = (*antenna_shape, paths, config.num_sinusoids)
        angles = request.random_stream.fork("wireless.tdl.doppler_angles").uniform(
            oscillator_shape, 0.0, 2.0 * pi, dtype=request.execution.float_dtype
        )
        phases = request.random_stream.fork("wireless.tdl.initial_phases").uniform(
            oscillator_shape, 0.0, 2.0 * pi, dtype=request.execution.float_dtype
        )
        time = backend.arange(
            request.num_time_samples,
            dtype=request.execution.float_dtype,
            device=request.execution.device,
        ) / request.sampling_frequency_hz
        time = backend.reshape(time, (1,) * len(oscillator_shape) + (request.num_time_samples,))
        doppler = self.maximum_doppler_hz * backend.cos(angles)
        diffuse = backend.sum(
            backend.exp(1j * (phases[..., None] + 2.0 * pi * doppler[..., None] * time)),
            axis=-2,
        ) / sqrt(config.num_sinusoids)

        los_phases = request.random_stream.fork("wireless.tdl.los_phases").uniform(
            antenna_shape, 0.0, 2.0 * pi, dtype=request.execution.float_dtype
        )
        los_time = backend.reshape(time, (request.num_time_samples,))
        path_values = []
        for index, (power, kind) in enumerate(
            zip(powers, self.profile.component_types, strict=True)
        ):
            if kind == "los":
                value = sqrt(power) * backend.exp(
                    1j
                    * (
                        los_phases[..., None]
                        + 2.0 * pi * 0.7 * self.maximum_doppler_hz * los_time
                    )
                )
            else:
                value = sqrt(power) * diffuse[..., index, :]
            path_values.append(value)
        coefficients = backend.stack(tuple(path_values), axis=5)
        coefficients = self._apply_spatial_correlation(
            coefficients, backend, request.execution
        )
        delay_vector = backend.array(
            delays_s, dtype=request.execution.float_dtype, device=request.execution.device
        )
        delays = backend.broadcast_to(
            backend.reshape(delay_vector, (1, 1, 1, paths)),
            (request.batch_size, config.num_rx, config.num_tx, paths),
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
            resolved_configuration=self._resolved_config,
            software_version=_software_version(),
            request_id=request.request_id,
        )
        return CIRBatch(coefficients, delays, metadata)


def _build_tdl(config: object) -> TappedDelayLine:
    if not isinstance(config, TDLConfig):
        raise TypeError("TDL builder requires TDLConfig")
    return TappedDelayLine(config)


def register_tdl_model(registry: ModelRegistry) -> None:
    """Register the ``wireless/tdl`` typed parser and builder.

    Parameters
    ----------
    registry : ModelRegistry
        Registry receiving the TDL model key.
    """

    registry.register("wireless", "tdl", parse_tdl_config, _build_tdl)
