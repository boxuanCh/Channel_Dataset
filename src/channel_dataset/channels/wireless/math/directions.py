"""Conversions between spherical directions and Cartesian vectors."""

from __future__ import annotations

from math import pi
from typing import Any

from channel_dataset.backends import backend_name_for_array, get_backend
from channel_dataset.core import ExecutionContext

from ._utils import as_real_array
from .angles import wrap_angle_radians


def spherical_to_cartesian(
    azimuth_rad: Any,
    elevation_rad: Any,
    execution: ExecutionContext,
    *,
    radius: Any = 1.0,
) -> Any:
    """Convert azimuth/elevation coordinates to Cartesian vectors.

    Parameters
    ----------
    azimuth_rad, elevation_rad : scalar or array-like
        Azimuth about the positive z-axis and elevation from the xy-plane.
    execution : ExecutionContext
        Backend, precision, and device for the returned array.
    radius : scalar or array-like, default=1
        Nonnegative vector magnitude. Inputs follow normal broadcasting rules.

    Returns
    -------
    backend array
        Cartesian vectors with final axis ``[x, y, z]``.

    Raises
    ------
    ValueError
        If an input is non-finite or a radius is negative.
    """

    backend = execution.array_backend
    azimuth = as_real_array(azimuth_rad, execution, "azimuth_rad")
    elevation = as_real_array(elevation_rad, execution, "elevation_rad")
    radius_array = as_real_array(radius, execution, "radius")
    if backend.to_scalar(backend.min(radius_array)) < 0.0:
        raise ValueError("radius must be nonnegative")
    zero = azimuth * 0.0 + elevation * 0.0 + radius_array * 0.0
    horizontal = radius_array * backend.cos(elevation)
    x = horizontal * backend.cos(azimuth) + zero
    y = horizontal * backend.sin(azimuth) + zero
    z = radius_array * backend.sin(elevation) + zero
    return backend.stack((x, y, z), axis=-1)


def cartesian_to_spherical(vectors: Any) -> tuple[Any, Any, Any]:
    """Convert Cartesian vectors to azimuth, elevation, and radius.

    Parameters
    ----------
    vectors : backend array
        Real array whose final axis is ``[x, y, z]``.

    Returns
    -------
    azimuth_rad, elevation_rad, radius : tuple of backend arrays
        Azimuth is wrapped to ``[-pi, pi)`` and elevation lies in
        ``[-pi/2, pi/2]``.

    Raises
    ------
    TypeError
        If ``vectors`` is not a real NumPy or PyTorch array.
    ValueError
        If its final dimension is not three, values are non-finite, or any
        vector has zero length.
    """

    backend = get_backend(backend_name_for_array(vectors))
    if not backend.is_floating(vectors):
        raise TypeError("vectors must have a real floating dtype")
    shape = backend.shape(vectors)
    if not shape or shape[-1] != 3:
        raise ValueError("vectors must have final dimension 3")
    if not backend.all_finite(vectors):
        raise ValueError("vectors must contain only finite values")
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    horizontal = backend.sqrt(x * x + y * y)
    radius = backend.sqrt(horizontal * horizontal + z * z)
    if backend.to_scalar(backend.min(radius)) <= 0.0:
        raise ValueError("zero-length vectors do not define a direction")
    azimuth = wrap_angle_radians(backend.atan2(y, x), lower_bound=-pi)
    elevation = backend.atan2(z, horizontal)
    return azimuth, elevation, radius


__all__ = ["cartesian_to_spherical", "spherical_to_cartesian"]
