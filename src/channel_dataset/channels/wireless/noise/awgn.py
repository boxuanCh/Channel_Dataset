"""Independent additive white Gaussian noise generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from channel_dataset.backends import RandomStream, backend_name_for_array, get_backend


@dataclass(frozen=True, slots=True)
class AWGNConfig:
    r"""Configuration for circular complex additive white Gaussian noise.

    Parameters
    ----------
    noise_variance_linear : float
        Nonnegative variance per complex sample,
        :math:`E[|n|^2]`. Independent real and imaginary components each have
        variance ``noise_variance_linear / 2``.
    """

    noise_variance_linear: float

    def __post_init__(self) -> None:
        value = self.noise_variance_linear
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("noise_variance_linear must be numeric")
        if not isfinite(float(value)) or value < 0.0:
            raise ValueError("noise_variance_linear must be finite and nonnegative")


def parse_awgn_config(parameters: Mapping[str, Any]) -> AWGNConfig:
    """Parse a strict AWGN parameter mapping."""

    fields = {"noise_variance_linear"}
    unknown = set(parameters) - fields
    if unknown:
        raise ValueError(f"unknown AWGN parameters: {sorted(unknown)}")
    missing = fields - set(parameters)
    if missing:
        raise ValueError(f"missing AWGN parameters: {sorted(missing)}")
    return AWGNConfig(parameters["noise_variance_linear"])


class AdditiveWhiteGaussianNoise:
    """Sample or add independent circular complex Gaussian noise.

    Noise generation is separate from every channel model and signal operator.
    A dedicated child stream makes its draws independent of fading draw order.
    """

    def __init__(self, config: AWGNConfig) -> None:
        self.config = config

    def sample(
        self,
        shape: tuple[int, ...],
        random_stream: RandomStream[Any],
        *,
        dtype: Any,
    ) -> Any:
        """Sample complex AWGN with the configured variance.

        Parameters
        ----------
        shape : tuple of int
            Positive output dimensions.
        random_stream : RandomStream
            Injected backend-native root stream.
        dtype : backend dtype or str
            Complex output dtype.
        """

        if not isinstance(shape, tuple) or not shape:
            raise ValueError("shape must be a non-empty tuple")
        if any(isinstance(size, bool) or not isinstance(size, int) for size in shape):
            raise TypeError("shape dimensions must be integers")
        if any(size <= 0 for size in shape):
            raise ValueError("shape dimensions must be positive")
        stream = random_stream.fork("wireless.awgn.samples")
        return stream.complex_normal(
            shape,
            variance=float(self.config.noise_variance_linear),
            dtype=dtype,
        )

    def apply(self, signal: Any, random_stream: RandomStream[Any]) -> Any:
        """Add a sampled noise tensor to a complex signal.

        The signal backend, device, dtype, and shape are preserved.
        """

        backend_name = backend_name_for_array(signal)
        backend = get_backend(backend_name)
        if random_stream.backend_name != backend_name:
            raise ValueError("random stream and signal use different backends")
        if random_stream.device != backend.device(signal):
            raise ValueError("random stream and signal use different devices")
        if not backend.is_complex(signal):
            raise TypeError("AWGN can only be added to a complex signal")
        if not backend.all_finite(signal):
            raise ValueError("signal must contain only finite values")
        noise = self.sample(
            backend.shape(signal), random_stream, dtype=backend.dtype(signal)
        )
        return signal + noise


__all__ = ["AWGNConfig", "AdditiveWhiteGaussianNoise", "parse_awgn_config"]
