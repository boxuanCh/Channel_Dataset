"""Polarization operations specific to Earth-space propagation."""

from __future__ import annotations

from typing import Any

from channel_dataset.core import ExecutionContext


def faraday_rotation_matrix(angle_rad: Any, execution: ExecutionContext) -> Any:
    """Return the power-preserving linear-polarization rotation matrix."""

    backend = execution.array_backend
    angle = backend.array(angle_rad, dtype=execution.float_dtype, device=execution.device)
    cosine = backend.cos(angle)
    sine = backend.sin(angle)
    return backend.stack(
        (
            backend.stack((cosine, -sine), axis=-1),
            backend.stack((sine, cosine), axis=-1),
        ),
        axis=-2,
    )


__all__ = ["faraday_rotation_matrix"]
