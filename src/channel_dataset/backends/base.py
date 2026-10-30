"""Backend-neutral array and random-stream contracts."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Protocol, TypeVar, runtime_checkable

ArrayT = TypeVar("ArrayT")


def _inherit_protocol_docs(protocol: type[Any]):
    """Copy missing method/property documentation from a protocol."""

    def decorate(concrete: type[Any]) -> type[Any]:
        for name, source in vars(protocol).items():
            if name.startswith("_") or not hasattr(concrete, name):
                continue
            target = vars(concrete).get(name)
            if isinstance(source, property) and isinstance(target, property):
                if target.__doc__ is None:
                    setattr(
                        concrete,
                        name,
                        property(target.fget, target.fset, target.fdel, source.__doc__),
                    )
            elif callable(source) and callable(target) and target.__doc__ is None:
                target.__doc__ = source.__doc__
        return concrete

    return decorate


def derive_child_seed(root_seed: int, lineage: tuple[str, ...]) -> int:
    """Derive a stable nonnegative seed for a component stream.

    Parameters
    ----------
    root_seed : int
        Nonnegative seed identifying the root experiment stream.
    lineage : tuple of str
        Ordered component labels from the root stream to the child.

    Returns
    -------
    int
        A deterministic 63-bit seed suitable for NumPy and PyTorch generators.

    Raises
    ------
    ValueError
        If ``root_seed`` is negative or any lineage label is empty.

    Notes
    -----
    The derivation uses SHA-256 rather than Python's process-randomized
    :func:`hash`, so results are stable across processes and platforms.
    """

    if root_seed < 0:
        raise ValueError("root_seed must be nonnegative")
    digest = sha256()
    digest.update(str(root_seed).encode("ascii"))
    for label in lineage:
        if not label:
            raise ValueError("stream labels must be non-empty")
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)


@runtime_checkable
class RandomStream(Protocol[ArrayT]):
    r"""Backend-native random source with deterministic component streams.

    Random streams own private generator state and never modify process-global
    NumPy or PyTorch state. Calling :meth:`fork` derives a child from the root
    seed and label lineage; it does not consume samples from the parent.

    Notes
    -----
    ``complex_normal`` follows the communications convention
    :math:`E[|z|^2] = \sigma^2`, where ``variance`` is :math:`\sigma^2` and
    each real component has variance :math:`\sigma^2 / 2`.
    """

    @property
    def backend_name(self) -> str:
        """str: Name of the array backend that receives generated samples."""

        ...

    @property
    def device(self) -> str:
        """str: Canonical device on which samples are generated."""

        ...

    @property
    def root_seed(self) -> int:
        """int: Nonnegative seed shared by this stream's lineage."""

        ...

    @property
    def stream_id(self) -> str:
        """str: Stable provenance identifier containing backend, seed, and lineage."""

        ...

    def uniform(
        self, shape: tuple[int, ...], low: float = 0.0, high: float = 1.0,
        *, dtype: Any = None,
    ) -> ArrayT:
        """Draw samples from a continuous uniform distribution.

        Parameters
        ----------
        shape : tuple of int
            Output shape.
        low, high : float
            Half-open sampling interval ``[low, high)``.
        dtype : optional
            Backend dtype. Defaults to double precision.

        Returns
        -------
        backend array
            Independent uniform samples on this stream's device.

        Raises
        ------
        ValueError
            If ``high`` is not greater than ``low``.
        """

        ...

    def normal(
        self, shape: tuple[int, ...], mean: float = 0.0, stddev: float = 1.0,
        *, dtype: Any = None,
    ) -> ArrayT:
        """Draw real Gaussian samples.

        Parameters
        ----------
        shape : tuple of int
            Output shape.
        mean : float
            Distribution mean.
        stddev : float
            Nonnegative standard deviation.
        dtype : optional
            Real backend dtype. Defaults to double precision.

        Returns
        -------
        backend array
            Gaussian samples on this stream's device.
        """

        ...

    def complex_normal(
        self, shape: tuple[int, ...], variance: float = 1.0, *, dtype: Any = None,
    ) -> ArrayT:
        """Draw circularly symmetric complex Gaussian samples.

        Parameters
        ----------
        shape : tuple of int
            Output shape.
        variance : float
            Nonnegative complex variance :math:`E[|z|^2]`.
        dtype : optional
            ``complex64`` or ``complex128`` backend dtype.

        Returns
        -------
        backend array
            Complex Gaussian samples.
        """

        ...

    def integers(self, shape: tuple[int, ...], low: int, high: int) -> ArrayT:
        """Draw integers uniformly from the half-open interval ``[low, high)``."""

        ...

    def bernoulli(self, shape: tuple[int, ...], probability: float) -> ArrayT:
        """Draw Boolean Bernoulli samples with the given success probability."""

        ...

    def permutation(self, n: int) -> ArrayT:
        """Return a random permutation of integers in ``[0, n)``."""

        ...

    def fork(self, label: str) -> RandomStream[ArrayT]:
        """Derive a repeatable, draw-order-independent child stream.

        Parameters
        ----------
        label : str
            Nonempty component label appended to the current lineage.

        Returns
        -------
        RandomStream
            New stream using the same backend, device, and root seed.
        """

        ...


@runtime_checkable
class ArrayBackend(Protocol[ArrayT]):
    """Array operations required by backend-neutral channel algorithms.

    Implementations preserve their native array type, dtype, device, and
    automatic-differentiation behavior. Methods intentionally mirror familiar
    NumPy and PyTorch operations; they do not convert arrays through another
    backend.

    Notes
    -----
    Shape arguments are tuples and reduction axes use NumPy-style integer or
    tuple notation. Validation helpers return Python booleans so container
    constructors can issue deterministic exceptions.
    """

    name: str

    def resolve_dtype(self, dtype: Any) -> Any:
        """Resolve a neutral dtype name to the backend's native dtype object."""

        ...

    def array(self, value: Any, *, dtype: Any = None, device: str = "cpu") -> ArrayT:
        """Convert array-like input to a native array on ``device``."""

        ...

    def zeros(self, shape: tuple[int, ...], *, dtype: Any, device: str = "cpu") -> ArrayT:
        """Create a native array filled with zeros."""

        ...

    def ones(self, shape: tuple[int, ...], *, dtype: Any, device: str = "cpu") -> ArrayT:
        """Create a native array filled with ones."""

        ...

    def full(
        self, shape: tuple[int, ...], value: Any, *, dtype: Any, device: str = "cpu",
    ) -> ArrayT:
        """Create a native array filled with ``value``."""

        ...

    def arange(self, stop: int, *, dtype: Any, device: str = "cpu") -> ArrayT:
        """Create values from zero up to, but excluding, ``stop``."""

        ...

    def astype(self, value: ArrayT, dtype: Any) -> ArrayT:
        """Cast an array to ``dtype`` without changing its backend."""

        ...

    def reshape(self, value: ArrayT, shape: tuple[int, ...]) -> ArrayT:
        """Return ``value`` with a new shape when storage permits."""

        ...

    def transpose(self, value: ArrayT, axes: tuple[int, ...]) -> ArrayT:
        """Permute array axes according to ``axes``."""

        ...

    def broadcast_to(self, value: ArrayT, shape: tuple[int, ...]) -> ArrayT:
        """Broadcast an array to ``shape``, returning a view when supported."""

        ...

    def stack(self, values: tuple[ArrayT, ...], axis: int = 0) -> ArrayT:
        """Join equal-shaped arrays along a new axis."""

        ...

    def concatenate(self, values: tuple[ArrayT, ...], axis: int = 0) -> ArrayT:
        """Join arrays along an existing axis."""

        ...

    def where(self, condition: ArrayT, x: ArrayT, y: ArrayT) -> ArrayT:
        """Select values from ``x`` or ``y`` according to ``condition``."""

        ...

    def clip(self, value: ArrayT, minimum: float, maximum: float) -> ArrayT:
        """Clamp every value to the closed interval ``[minimum, maximum]``."""

        ...

    def cumsum(self, value: ArrayT, axis: int = -1) -> ArrayT:
        """Return cumulative sums along ``axis``."""

        ...

    def sort(self, value: ArrayT, axis: int = -1) -> ArrayT:
        """Sort values along an axis in ascending order."""

        ...

    def argsort(self, value: ArrayT, axis: int = -1) -> ArrayT:
        """Return indices that sort values along an axis."""

        ...

    def take_along_axis(self, value: ArrayT, indices: ArrayT, axis: int) -> ArrayT:
        """Select values using same-rank indices along one axis."""

        ...

    def sum(self, value: ArrayT, axis: int | tuple[int, ...] | None = None) -> ArrayT:
        """Sum array values over ``axis`` or over all axes."""

        ...

    def mean(self, value: ArrayT, axis: int | tuple[int, ...] | None = None) -> ArrayT:
        """Compute the arithmetic mean over ``axis`` or over all axes."""

        ...

    def min(self, value: ArrayT, axis: int | None = None) -> ArrayT:
        """Return minimum values over ``axis`` or over the entire array."""

        ...

    def max(self, value: ArrayT, axis: int | None = None) -> ArrayT:
        """Return maximum values over ``axis`` or over the entire array."""

        ...

    def abs(self, value: ArrayT) -> ArrayT:
        """Compute elementwise absolute values."""

        ...

    def exp(self, value: ArrayT) -> ArrayT:
        """Compute the elementwise exponential."""

        ...

    def sqrt(self, value: ArrayT) -> ArrayT:
        """Compute the elementwise square root."""

        ...

    def log(self, value: ArrayT) -> ArrayT:
        """Compute the elementwise natural logarithm."""

        ...

    def log10(self, value: ArrayT) -> ArrayT:
        """Compute the elementwise base-10 logarithm."""

        ...

    def sin(self, value: ArrayT) -> ArrayT:
        """Compute the elementwise sine in radians."""

        ...

    def sinc(self, value: ArrayT) -> ArrayT:
        r"""Compute the normalized sinc function elementwise.

        The normalized convention is :math:`\operatorname{sinc}(x) =
        \sin(\pi x)/(\pi x)`, with the value at zero defined as one.
        """

        ...

    def cos(self, value: ArrayT) -> ArrayT:
        """Compute the elementwise cosine in radians."""

        ...

    def atan2(self, y: ArrayT, x: ArrayT) -> ArrayT:
        """Compute the elementwise quadrant-aware arctangent in radians."""

        ...

    def conj(self, value: ArrayT) -> ArrayT:
        """Return the elementwise complex conjugate."""

        ...

    def matmul(self, left: ArrayT, right: ArrayT) -> ArrayT:
        """Perform matrix multiplication with native broadcasting rules."""

        ...

    def einsum(self, equation: str, *values: ArrayT) -> ArrayT:
        """Evaluate an Einstein summation expression."""

        ...

    def fft(self, value: ArrayT, axis: int = -1) -> ArrayT:
        """Compute the one-dimensional discrete Fourier transform along ``axis``."""

        ...

    def ifft(self, value: ArrayT, axis: int = -1) -> ArrayT:
        """Compute the inverse one-dimensional discrete Fourier transform."""

        ...

    def cholesky(self, value: ArrayT) -> ArrayT:
        """Compute a lower-triangular Cholesky factor."""

        ...

    def eigh(self, value: ArrayT) -> tuple[ArrayT, ArrayT]:
        """Return eigenvalues and eigenvectors of a Hermitian matrix."""

        ...

    def svdvals(self, value: ArrayT) -> ArrayT:
        """Return singular values in descending order."""

        ...

    def ndim(self, value: ArrayT) -> int:
        """Return the number of array dimensions."""

        ...

    def shape(self, value: ArrayT) -> tuple[int, ...]:
        """Return the array shape as a tuple of Python integers."""

        ...

    def dtype(self, value: ArrayT) -> Any:
        """Return the native dtype object for ``value``."""

        ...

    def device(self, value: ArrayT) -> str:
        """Return the canonical device string for ``value``."""

        ...

    def is_array(self, value: Any) -> bool:
        """Return whether ``value`` is a native backend array."""

        ...

    def is_complex(self, value: ArrayT) -> bool:
        """Return whether ``value`` has a complex dtype."""

        ...

    def is_floating(self, value: ArrayT) -> bool:
        """Return whether ``value`` has a real floating dtype."""

        ...

    def is_integer(self, value: ArrayT) -> bool:
        """Return whether ``value`` has an integer dtype."""

        ...

    def is_bool(self, value: ArrayT) -> bool:
        """Return whether ``value`` has a Boolean dtype."""

        ...

    def all_finite(self, value: ArrayT) -> bool:
        """Return whether every value is finite."""

        ...

    def all_nonnegative(self, value: ArrayT) -> bool:
        """Return whether every value is greater than or equal to zero."""

        ...

    def to_scalar(self, value: Any) -> Any:
        """Convert a one-element native value to a Python scalar."""

        ...

    def random_stream(self, seed: int, *, device: str = "cpu") -> RandomStream[ArrayT]:
        """Create a private native random stream from ``seed`` on ``device``."""

        ...
