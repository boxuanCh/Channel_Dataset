"""Lazy backend discovery and default selection."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from channel_dataset.backends.base import ArrayBackend, RandomStream

_BACKEND_TYPES = {
    "numpy": ("channel_dataset.backends.numpy_backend", "NumpyBackend"),
    "torch": ("channel_dataset.backends.torch_backend", "TorchBackend"),
}
_CACHE: dict[str, ArrayBackend[Any]] = {}


def get_backend(name: str | None = None) -> ArrayBackend[Any]:
    """Resolve a lazily imported numerical backend.

    Parameters
    ----------
    name : {"torch", "numpy"}, optional
        Backend name. ``None`` selects PyTorch, the project default.

    Returns
    -------
    ArrayBackend
        Cached backend instance implementing the shared array protocol.

    Raises
    ------
    ValueError
        If ``name`` is not a registered backend.
    ImportError
        If PyTorch is requested but the optional dependency is unavailable.

    Notes
    -----
    NumPy resolution never imports PyTorch. The registry does not silently
    fall back from PyTorch to NumPy because doing so could change execution
    and differentiation behavior.
    """

    resolved = "torch" if name is None else name.lower()
    if resolved not in _BACKEND_TYPES:
        raise ValueError(f"unknown backend {name!r}; expected one of {sorted(_BACKEND_TYPES)}")
    if resolved not in _CACHE:
        module_name, class_name = _BACKEND_TYPES[resolved]
        try:
            module = import_module(module_name)
        except ImportError as error:
            if resolved == "torch":
                raise ImportError(
                    "PyTorch is the default backend but is unavailable; install "
                    "channel-dataset[wireless-torch] or select backend='numpy'"
                ) from error
            raise
        _CACHE[resolved] = getattr(module, class_name)()
    return _CACHE[resolved]


def backend_name_for_array(value: Any) -> str:
    """Identify the backend that owns an array.

    Parameters
    ----------
    value : object
        Candidate NumPy array or PyTorch tensor.

    Returns
    -------
    {"numpy", "torch"}
        Name of the owning backend.

    Raises
    ------
    TypeError
        If ``value`` is not a supported native array.
    """

    module_root = type(value).__module__.partition(".")[0]
    if module_root == "numpy":
        return "numpy"
    if module_root == "torch":
        return "torch"
    raise TypeError(f"unsupported array type: {type(value).__name__}")


def create_random_stream(
    seed: int, *, backend: str | None = None, device: str = "cpu"
) -> RandomStream[Any]:
    """Create a private backend-native random stream.

    Parameters
    ----------
    seed : int
        Nonnegative root seed.
    backend : {"torch", "numpy"}, optional
        Backend name. ``None`` selects PyTorch.
    device : str, default="cpu"
        Native device string. NumPy accepts only ``"cpu"``.

    Returns
    -------
    RandomStream
        Random stream backed by ``torch.Generator`` or
        ``numpy.random.Generator``.
    """

    return get_backend(backend).random_stream(seed, device=device)
