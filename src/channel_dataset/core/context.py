"""Simulation inputs which are independent of a channel family."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from channel_dataset.backends import ArrayBackend, RandomStream


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Numerical execution controls that do not alter channel physics.

    Parameters
    ----------
    backend : {"torch", "numpy"}, default="torch"
        Numerical backend. PyTorch is selected by default.
    precision : {"single", "double"}, default="single"
        Maps to ``float32/complex64`` or ``float64/complex128``.
    device : str, default="cpu"
        Backend-native device. NumPy supports CPU only; ``"cuda"`` is
        canonicalized to ``"cuda:0"`` for PyTorch.

    Raises
    ------
    TypeError
        If backend, precision, or device is not a string.
    ValueError
        If a backend, precision, or device combination is unsupported.

    Examples
    --------
    >>> context = ExecutionContext("numpy", "double", "cpu")
    >>> context.float_dtype
    'float64'
    """

    backend: str = "torch"
    precision: str = "single"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str):
            raise TypeError("backend must be a string")
        if not isinstance(self.precision, str):
            raise TypeError("precision must be a string")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        normalized_backend = self.backend.lower()
        normalized_precision = self.precision.lower()
        if normalized_backend not in {"numpy", "torch"}:
            raise ValueError("backend must be 'numpy' or 'torch'")
        if normalized_precision not in {"single", "double"}:
            raise ValueError("precision must be 'single' or 'double'")
        if normalized_backend == "numpy" and self.device != "cpu":
            raise ValueError("the NumPy backend supports only device='cpu'")
        normalized_device = (
            "cuda:0"
            if normalized_backend == "torch" and self.device == "cuda"
            else self.device
        )
        object.__setattr__(self, "backend", normalized_backend)
        object.__setattr__(self, "precision", normalized_precision)
        object.__setattr__(self, "device", normalized_device)

    @property
    def float_dtype(self) -> str:
        """Return the backend-neutral real dtype name.

        Returns
        -------
        {"float32", "float64"}
            Dtype corresponding to :attr:`precision`.
        """

        return "float32" if self.precision == "single" else "float64"

    @property
    def complex_dtype(self) -> str:
        """Return the backend-neutral complex dtype name.

        Returns
        -------
        {"complex64", "complex128"}
            Dtype corresponding to :attr:`precision`.
        """

        return "complex64" if self.precision == "single" else "complex128"

    @property
    def array_backend(self) -> ArrayBackend[Any]:
        """Resolve the configured backend lazily.

        Returns
        -------
        ArrayBackend
            Cached NumPy or PyTorch backend instance.

        Raises
        ------
        ImportError
            If PyTorch was selected but is not installed.
        """

        from channel_dataset.backends import get_backend

        return get_backend(self.backend)

    def random_stream(self, seed: int) -> RandomStream[Any]:
        """Create a private random stream for this execution context.

        Parameters
        ----------
        seed : int
            Nonnegative root seed.

        Returns
        -------
        RandomStream
            Stream using this context's backend and device.
        """

        return self.array_backend.random_stream(seed, device=self.device)


@dataclass(frozen=True, slots=True)
class Vector3:
    """Cartesian vector expressed in a documented coordinate frame.

    Parameters
    ----------
    x, y, z : float
        Components in the units defined by the containing field.
    """

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One endpoint of a propagation link, expressed in SI units.

    Parameters
    ----------
    position_m : Vector3
        Cartesian endpoint position in metres.
    velocity_mps : Vector3, optional
        Cartesian velocity in metres per second.
    metadata : mapping, optional
        Read-only application metadata.
    """

    position_m: Vector3
    velocity_mps: Vector3 = Vector3(0.0, 0.0, 0.0)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SimulationContext:
    """Scalar link conditions for models that require explicit geometry.

    Parameters
    ----------
    transmitter, receiver : Endpoint
        Link endpoints.
    observation_time_s : float
        Nonnegative observation time in seconds.
    sample_rate_hz : float
        Positive sample rate in hertz.
    carrier_frequency_hz : float, optional
        Positive carrier frequency in hertz.
    environment : mapping, optional
        Read-only medium or scenario metadata.
    """

    transmitter: Endpoint
    receiver: Endpoint
    observation_time_s: float
    sample_rate_hz: float
    carrier_frequency_hz: float | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observation_time_s < 0:
            raise ValueError("observation_time_s cannot be negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.carrier_frequency_hz is not None and self.carrier_frequency_hz <= 0:
            raise ValueError("carrier_frequency_hz must be positive when provided")
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
