"""Private validation helpers for backend-neutral wireless mathematics."""

from __future__ import annotations

from typing import Any

from channel_dataset.backends import backend_name_for_array
from channel_dataset.core import ExecutionContext


def as_real_array(value: Any, execution: ExecutionContext, name: str) -> Any:
    backend = execution.array_backend
    try:
        owner = backend_name_for_array(value)
    except TypeError:
        owner = None
    if owner is not None and owner != execution.backend:
        raise ValueError(f"{name} backend does not match execution context")
    array = backend.array(value, dtype=execution.float_dtype, device=execution.device)
    if not backend.is_floating(array):
        raise TypeError(f"{name} must have a real floating dtype")
    if not backend.all_finite(array):
        raise ValueError(f"{name} must contain only finite values")
    return array
