"""Native PyTorch array backend and random streams.

This module is imported lazily so NumPy-only users never import PyTorch.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

try:
    import torch
except ImportError as error:  # pragma: no cover - exercised in an isolated import test
    raise ImportError(
        "PyTorch is required for the 'torch' backend; install channel-dataset[wireless-torch] "
        "or select backend='numpy'"
    ) from error

from channel_dataset.backends.base import (
    ArrayBackend,
    RandomStream,
    _inherit_protocol_docs,
    derive_child_seed,
)


def _normalize_device(device: str) -> str:
    resolved = str(torch.device(device))
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {device}")
    return resolved


_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
    "int64": torch.int64,
    "bool": torch.bool,
}


@_inherit_protocol_docs(RandomStream)
class TorchRandomStream:
    """Random stream backed by a private ``torch.Generator``.

    Parameters
    ----------
    root_seed : int
        Nonnegative experiment seed.
    lineage : tuple of str, optional
        Component labels used to derive this stream from the root.
    device : str, default="cpu"
        PyTorch device on which the generator and samples are created.

    Notes
    -----
    See :class:`channel_dataset.backends.RandomStream` for distribution
    semantics and method documentation.
    """

    def __init__(
        self, root_seed: int, *, lineage: tuple[str, ...] = (), device: str = "cpu",
    ) -> None:
        if root_seed < 0:
            raise ValueError("seed must be nonnegative")
        self._root_seed = int(root_seed)
        self._lineage = lineage
        self._device = _normalize_device(device)
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(derive_child_seed(root_seed, lineage))

    @property
    def backend_name(self) -> str:
        return "torch"

    @property
    def device(self) -> str:
        return self._device

    @property
    def root_seed(self) -> int:
        return self._root_seed

    @property
    def stream_id(self) -> str:
        suffix = "/".join(self._lineage) or "root"
        return f"torch:{self._root_seed}:{suffix}"

    def _dtype(self, dtype: Any, default: torch.dtype) -> torch.dtype:
        if dtype is None:
            return default
        if isinstance(dtype, str):
            try:
                return _DTYPES[dtype]
            except KeyError as error:
                raise TypeError(f"unsupported PyTorch dtype: {dtype}") from error
        return dtype

    def uniform(
        self, shape: tuple[int, ...], low: float = 0.0, high: float = 1.0,
        *, dtype: Any = None,
    ) -> torch.Tensor:
        if not high > low:
            raise ValueError("high must be greater than low")
        value = torch.rand(
            shape, generator=self._generator, dtype=self._dtype(dtype, torch.float64),
            device=self._device,
        )
        return value * (high - low) + low

    def normal(
        self, shape: tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
        *, dtype: Any = None,
    ) -> torch.Tensor:
        if stddev < 0:
            raise ValueError("stddev must be nonnegative")
        value = torch.randn(
            shape, generator=self._generator, dtype=self._dtype(dtype, torch.float64),
            device=self._device,
        )
        return value * stddev + mean

    def complex_normal(
        self, shape: tuple[int, ...], variance: float = 1.0, *, dtype: Any = None,
    ) -> torch.Tensor:
        if variance < 0:
            raise ValueError("variance must be nonnegative")
        complex_dtype = self._dtype(dtype, torch.complex128)
        if complex_dtype not in {torch.complex64, torch.complex128}:
            raise TypeError("complex_normal requires complex64 or complex128 dtype")
        real_dtype = torch.float32 if complex_dtype == torch.complex64 else torch.float64
        scale = sqrt(variance / 2.0)
        real = self.normal(shape, stddev=scale, dtype=real_dtype)
        imag = self.normal(shape, stddev=scale, dtype=real_dtype)
        return torch.complex(real, imag)

    def integers(self, shape: tuple[int, ...], low: int, high: int) -> torch.Tensor:
        if high <= low:
            raise ValueError("high must be greater than low")
        return torch.randint(
            low, high, shape, generator=self._generator, dtype=torch.int64, device=self._device
        )

    def bernoulli(self, shape: tuple[int, ...], probability: float) -> torch.Tensor:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        return torch.rand(shape, generator=self._generator, device=self._device) < probability

    def permutation(self, n: int) -> torch.Tensor:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return torch.randperm(n, generator=self._generator, device=self._device)

    def fork(self, label: str) -> TorchRandomStream:
        if not label:
            raise ValueError("stream label must be non-empty")
        return TorchRandomStream(
            self._root_seed, lineage=self._lineage + (label,), device=self._device
        )


@_inherit_protocol_docs(ArrayBackend)
class TorchBackend:
    """Implement the shared array contract using PyTorch tensors.

    Operations preserve tensor devices and autograd graphs. The module is
    imported lazily so NumPy-only applications do not require PyTorch.

    See Also
    --------
    channel_dataset.backends.ArrayBackend
        Complete operation semantics shared by both backends.
    """

    name = "torch"

    def resolve_dtype(self, dtype: Any) -> torch.dtype:
        if isinstance(dtype, str):
            try:
                return _DTYPES[dtype]
            except KeyError as error:
                raise TypeError(f"unsupported PyTorch dtype: {dtype}") from error
        if isinstance(dtype, torch.dtype):
            return dtype
        raise TypeError(f"unsupported PyTorch dtype: {dtype!r}")

    def array(self, value: Any, *, dtype: Any = None, device: str = "cpu") -> torch.Tensor:
        resolved_dtype = None if dtype is None else self.resolve_dtype(dtype)
        return torch.as_tensor(value, dtype=resolved_dtype, device=_normalize_device(device))

    def zeros(self, shape: tuple[int, ...], *, dtype: Any, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(shape, dtype=self.resolve_dtype(dtype), device=_normalize_device(device))

    def ones(self, shape: tuple[int, ...], *, dtype: Any, device: str = "cpu") -> torch.Tensor:
        return torch.ones(shape, dtype=self.resolve_dtype(dtype), device=_normalize_device(device))

    def full(
        self, shape: tuple[int, ...], value: Any, *, dtype: Any, device: str = "cpu",
    ) -> torch.Tensor:
        return torch.full(
            shape, value, dtype=self.resolve_dtype(dtype), device=_normalize_device(device)
        )

    def arange(self, stop: int, *, dtype: Any, device: str = "cpu") -> torch.Tensor:
        return torch.arange(stop, dtype=self.resolve_dtype(dtype), device=_normalize_device(device))

    def astype(self, value: torch.Tensor, dtype: Any) -> torch.Tensor:
        return value.to(dtype=self.resolve_dtype(dtype))

    def reshape(self, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.reshape(value, shape)

    def transpose(self, value: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        return value.permute(axes)

    def broadcast_to(self, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(value, shape)

    def stack(self, values: tuple[torch.Tensor, ...], axis: int = 0) -> torch.Tensor:
        return torch.stack(values, dim=axis)

    def concatenate(self, values: tuple[torch.Tensor, ...], axis: int = 0) -> torch.Tensor:
        return torch.cat(values, dim=axis)

    def where(
        self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(condition, x, y)

    def clip(self, value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
        return torch.clamp(value, min=minimum, max=maximum)

    def cumsum(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.cumsum(value, dim=axis)

    def sort(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.sort(value, dim=axis).values

    def argsort(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.argsort(value, dim=axis)

    def take_along_axis(
        self, value: torch.Tensor, indices: torch.Tensor, axis: int
    ) -> torch.Tensor:
        return torch.take_along_dim(value, indices, dim=axis)

    def sum(self, value: torch.Tensor, axis: Any = None) -> torch.Tensor:
        return torch.sum(value) if axis is None else torch.sum(value, dim=axis)

    def mean(self, value: torch.Tensor, axis: Any = None) -> torch.Tensor:
        return torch.mean(value) if axis is None else torch.mean(value, dim=axis)

    def min(self, value: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        return torch.min(value) if axis is None else torch.min(value, dim=axis).values

    def max(self, value: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        return torch.max(value) if axis is None else torch.max(value, dim=axis).values

    def abs(self, value: torch.Tensor) -> torch.Tensor:
        return torch.abs(value)

    def exp(self, value: torch.Tensor) -> torch.Tensor:
        return torch.exp(value)

    def sqrt(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value)

    def log(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(value)

    def log10(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log10(value)

    def sin(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sin(value)

    def sinc(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sinc(value)

    def cos(self, value: torch.Tensor) -> torch.Tensor:
        return torch.cos(value)

    def atan2(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(y, x)

    def conj(self, value: torch.Tensor) -> torch.Tensor:
        return torch.conj(value)

    def matmul(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.matmul(left, right)

    def einsum(self, equation: str, *values: torch.Tensor) -> torch.Tensor:
        return torch.einsum(equation, *values)

    def fft(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.fft.fft(value, dim=axis)

    def ifft(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.fft.ifft(value, dim=axis)

    def cholesky(self, value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.cholesky(value)

    def eigh(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigh(value)

    def svdvals(self, value: torch.Tensor) -> torch.Tensor:
        return torch.linalg.svdvals(value)

    def ndim(self, value: torch.Tensor) -> int:
        return value.ndim

    def shape(self, value: torch.Tensor) -> tuple[int, ...]:
        return tuple(value.shape)

    def dtype(self, value: torch.Tensor) -> torch.dtype:
        return value.dtype

    def device(self, value: torch.Tensor) -> str:
        return str(value.device)

    def is_array(self, value: Any) -> bool:
        return isinstance(value, torch.Tensor)

    def is_complex(self, value: torch.Tensor) -> bool:
        return value.is_complex()

    def is_floating(self, value: torch.Tensor) -> bool:
        return value.is_floating_point()

    def is_integer(self, value: torch.Tensor) -> bool:
        return value.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }

    def is_bool(self, value: torch.Tensor) -> bool:
        return value.dtype == torch.bool

    def all_finite(self, value: torch.Tensor) -> bool:
        return bool(torch.all(torch.isfinite(value)).item())

    def all_nonnegative(self, value: torch.Tensor) -> bool:
        return bool(torch.all(value >= 0).item())

    def to_scalar(self, value: Any) -> Any:
        return value.item()

    def random_stream(self, seed: int, *, device: str = "cpu") -> TorchRandomStream:
        return TorchRandomStream(seed, device=device)
