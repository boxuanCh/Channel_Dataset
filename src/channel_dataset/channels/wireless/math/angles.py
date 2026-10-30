"""Angle conversion and canonical wrapping helpers."""

from __future__ import annotations

from math import pi
from typing import Any


def degrees_to_radians(angle_degrees: Any) -> Any:
    """Convert degrees to radians while preserving an array backend."""

    return angle_degrees * (pi / 180.0)


def radians_to_degrees(angle_radians: Any) -> Any:
    """Convert radians to degrees while preserving an array backend."""

    return angle_radians * (180.0 / pi)


def wrap_angle_radians(angle_radians: Any, *, lower_bound: float = -pi) -> Any:
    r"""Wrap angles into a half-open interval of width :math:`2\pi`.

    Parameters
    ----------
    angle_radians : float or backend array
        Input angles in radians.
    lower_bound : float, default=-pi
        Inclusive lower bound. The upper bound is exclusive.

    Returns
    -------
    float or backend array
        Angles in ``[lower_bound, lower_bound + 2*pi)``.
    """

    return (angle_radians - lower_bound) % (2.0 * pi) + lower_bound


def wrap_angle_degrees(angle_degrees: Any, *, lower_bound: float = -180.0) -> Any:
    """Wrap degree angles into a half-open interval of width 360 degrees."""

    return (angle_degrees - lower_bound) % 360.0 + lower_bound


__all__ = [
    "degrees_to_radians",
    "radians_to_degrees",
    "wrap_angle_degrees",
    "wrap_angle_radians",
]
