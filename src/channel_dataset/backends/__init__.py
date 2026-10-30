"""Numerical backends and injected random streams."""

from channel_dataset.backends.base import ArrayBackend, RandomStream
from channel_dataset.backends.registry import (
    backend_name_for_array,
    create_random_stream,
    get_backend,
)

__all__ = [
    "ArrayBackend",
    "RandomStream",
    "backend_name_for_array",
    "create_random_stream",
    "get_backend",
]
