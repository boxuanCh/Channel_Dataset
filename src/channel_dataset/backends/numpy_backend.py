"""Native NumPy array backend and random streams."""

from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np

from channel_dataset.backends.base import (
    ArrayBackend,
    RandomStream,
    _inherit_protocol_docs,
    derive_child_seed,
)


def _validate_device(device: str) -> None:
    if device != "cpu":
        raise ValueError("the NumPy backend supports only device='cpu'")


@_inherit_protocol_docs(RandomStream)
class NumpyRandomStream:
    """Random stream backed by an independent ``numpy.random.Generator``.

    Parameters
    ----------
    root_seed : int
        Nonnegative experiment seed.
    lineage : tuple of str, optional
        Component labels used to derive this stream from the root.
    device : {"cpu"}, default="cpu"
        NumPy execution device.

    Notes
    -----
    See :class:`channel_dataset.backends.RandomStream` for distribution
    semantics and method documentation.
    """

    def __init__(
        self, root_seed: int, *, lineage: tuple[str, ...] = (), device: str = "cpu",
    ) -> None:
        _validate_device(device)
        if root_seed < 0:
            raise ValueError("seed must be nonnegative")
        self._root_seed = int(root_seed)
        self._lineage = lineage
        self._generator = np.random.default_rng(derive_child_seed(root_seed, lineage))

    @property
    def backend_name(self) -> str:
        return "numpy"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def root_seed(self) -> int:
        return self._root_seed

    @property
    def stream_id(self) -> str:
        suffix = "/".join(self._lineage) or "root"
        return f"numpy:{self._root_seed}:{suffix}"

    def uniform(
        self, shape: tuple[int, ...], low: float = 0.0, high: float = 1.0,
        *, dtype: Any = None,
    ) -> np.ndarray:
        if not high > low:
            raise ValueError("high must be greater than low")
        resolved = np.dtype(dtype or np.float64)
        return self._generator.uniform(low, high, size=shape).astype(resolved, copy=False)

    def normal(
        self, shape: tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
        *, dtype: Any = None,
    ) -> np.ndarray:
        if stddev < 0:
            raise ValueError("stddev must be nonnegative")
        resolved = np.dtype(dtype or np.float64)
        return self._generator.normal(mean, stddev, size=shape).astype(resolved, copy=False)

    def complex_normal(
        self, shape: tuple[int, ...], variance: float = 1.0, *, dtype: Any = None,
    ) -> np.ndarray:
        if variance < 0:
            raise ValueError("variance must be nonnegative")
        complex_dtype = np.dtype(dtype or np.complex128)
        if complex_dtype not in {np.dtype(np.complex64), np.dtype(np.complex128)}:
            raise TypeError("complex_normal requires complex64 or complex128 dtype")
        real_dtype = np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
        scale = sqrt(variance / 2.0)
        real = self.normal(shape, stddev=scale, dtype=real_dtype)
        imag = self.normal(shape, stddev=scale, dtype=real_dtype)
        return (real + 1j * imag).astype(complex_dtype, copy=False)

    def integers(self, shape: tuple[int, ...], low: int, high: int) -> np.ndarray:
        if high <= low:
            raise ValueError("high must be greater than low")
        return self._generator.integers(low, high, size=shape, dtype=np.int64)

    def bernoulli(self, shape: tuple[int, ...], probability: float) -> np.ndarray:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        return self._generator.random(shape) < probability

    def permutation(self, n: int) -> np.ndarray:
        if n < 0:
            raise ValueError("n must be nonnegative")
        return self._generator.permutation(n)

    def fork(self, label: str) -> NumpyRandomStream:
        if not label:
            raise ValueError("stream label must be non-empty")
        return NumpyRandomStream(self._root_seed, lineage=self._lineage + (label,))


@_inherit_protocol_docs(ArrayBackend)
class NumpyBackend:
    """Implement the shared array contract using NumPy only.

    Arrays remain :class:`numpy.ndarray` objects and random operations use
    :class:`NumpyRandomStream`. PyTorch is neither imported nor used as a
    conversion layer. Only the CPU device is supported.

    See Also
    --------
    channel_dataset.backends.ArrayBackend
        Complete operation semantics shared by both backends.
    """

    name = "numpy"

    def resolve_dtype(self, dtype: Any) -> np.dtype[Any]:
        return np.dtype(dtype)

    def array(self, value: Any, *, dtype: Any = None, device: str = "cpu") -> np.ndarray:
        _validate_device(device)
        return np.asarray(value, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], *, dtype: Any, device: str = "cpu") -> np.ndarray:
        _validate_device(device)
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], *, dtype: Any, device: str = "cpu") -> np.ndarray:
        _validate_device(device)
        return np.ones(shape, dtype=dtype)

    def full(
        self, shape: tuple[int, ...], value: Any, *, dtype: Any, device: str = "cpu",
    ) -> np.ndarray:
        _validate_device(device)
        return np.full(shape, value, dtype=dtype)

    def arange(self, stop: int, *, dtype: Any, device: str = "cpu") -> np.ndarray:
        _validate_device(device)
        return np.arange(stop, dtype=dtype)

    def astype(self, value: np.ndarray, dtype: Any) -> np.ndarray:
        return value.astype(dtype, copy=False)

    def reshape(self, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.reshape(value, shape)

    def transpose(self, value: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
        return np.transpose(value, axes)

    def broadcast_to(self, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        return np.broadcast_to(value, shape)

    def stack(self, values: tuple[np.ndarray, ...], axis: int = 0) -> np.ndarray:
        return np.stack(values, axis=axis)

    def concatenate(self, values: tuple[np.ndarray, ...], axis: int = 0) -> np.ndarray:
        return np.concatenate(values, axis=axis)

    def where(
        self, condition: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        return np.where(condition, x, y)

    def clip(self, value: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
        return np.clip(value, minimum, maximum)

    def cumsum(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.cumsum(value, axis=axis)

    def sort(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.sort(value, axis=axis)

    def argsort(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.argsort(value, axis=axis)

    def take_along_axis(
        self, value: np.ndarray, indices: np.ndarray, axis: int
    ) -> np.ndarray:
        return np.take_along_axis(value, indices, axis=axis)

    def sum(self, value: np.ndarray, axis: Any = None) -> np.ndarray:
        return np.sum(value, axis=axis)

    def mean(self, value: np.ndarray, axis: Any = None) -> np.ndarray:
        return np.mean(value, axis=axis)

    def min(self, value: np.ndarray, axis: int | None = None) -> np.ndarray:
        return np.min(value, axis=axis)

    def max(self, value: np.ndarray, axis: int | None = None) -> np.ndarray:
        return np.max(value, axis=axis)

    def abs(self, value: np.ndarray) -> np.ndarray:
        return np.abs(value)

    def exp(self, value: np.ndarray) -> np.ndarray:
        return np.exp(value)

    def sqrt(self, value: np.ndarray) -> np.ndarray:
        return np.sqrt(value)

    def log(self, value: np.ndarray) -> np.ndarray:
        return np.log(value)

    def log10(self, value: np.ndarray) -> np.ndarray:
        return np.log10(value)

    def sin(self, value: np.ndarray) -> np.ndarray:
        return np.sin(value)

    def sinc(self, value: np.ndarray) -> np.ndarray:
        return np.sinc(value)

    def cos(self, value: np.ndarray) -> np.ndarray:
        return np.cos(value)

    def atan2(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.arctan2(y, x)

    def conj(self, value: np.ndarray) -> np.ndarray:
        return np.conj(value)

    def matmul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.matmul(left, right)

    def einsum(self, equation: str, *values: np.ndarray) -> np.ndarray:
        return np.einsum(equation, *values)

    def fft(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.fft.fft(value, axis=axis)

    def ifft(self, value: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.fft.ifft(value, axis=axis)

    def cholesky(self, value: np.ndarray) -> np.ndarray:
        return np.linalg.cholesky(value)

    def eigh(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(value)

    def svdvals(self, value: np.ndarray) -> np.ndarray:
        return np.linalg.svd(value, compute_uv=False)

    def ndim(self, value: np.ndarray) -> int:
        return value.ndim

    def shape(self, value: np.ndarray) -> tuple[int, ...]:
        return value.shape

    def dtype(self, value: np.ndarray) -> np.dtype[Any]:
        return value.dtype

    def device(self, value: np.ndarray) -> str:
        return "cpu"

    def is_array(self, value: Any) -> bool:
        return isinstance(value, np.ndarray)

    def is_complex(self, value: np.ndarray) -> bool:
        return bool(np.issubdtype(value.dtype, np.complexfloating))

    def is_floating(self, value: np.ndarray) -> bool:
        return bool(np.issubdtype(value.dtype, np.floating))

    def is_integer(self, value: np.ndarray) -> bool:
        return bool(np.issubdtype(value.dtype, np.integer))

    def is_bool(self, value: np.ndarray) -> bool:
        return value.dtype == np.bool_

    def all_finite(self, value: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(value)))

    def all_nonnegative(self, value: np.ndarray) -> bool:
        return bool(np.all(value >= 0))

    def to_scalar(self, value: Any) -> Any:
        return value.item()

    def random_stream(self, seed: int, *, device: str = "cpu") -> NumpyRandomStream:
        return NumpyRandomStream(seed, device=device)
