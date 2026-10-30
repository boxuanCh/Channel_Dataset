"""Optional plotting helpers for antenna elements."""

from __future__ import annotations

from math import pi
from typing import Any

from channel_dataset.core import ExecutionContext

from .coordinates import DEFAULT_ORIENTATION, Orientation
from .elements import AntennaElement


def plot_horizontal_power_pattern(
    element: AntennaElement,
    *,
    num_points: int = 361,
    orientation: Orientation = DEFAULT_ORIENTATION,
) -> Any:
    """Plot an element's horizontal-cut power pattern with Matplotlib.

    Parameters
    ----------
    element : AntennaElement
        Element pattern to evaluate.
    num_points : int, default=361
        Number of azimuth samples over ``[-pi, pi]``.
    orientation : Orientation, optional
        Element orientation in the plotted global coordinate system.

    Returns
    -------
    matplotlib.axes.Axes
        Polar axes containing the plotted linear power gain.

    Raises
    ------
    ImportError
        If the optional Matplotlib dependency is unavailable.
    """

    if num_points < 2:
        raise ValueError("num_points must be at least two")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("plotting antenna patterns requires matplotlib") from error
    azimuth = np.linspace(-pi, pi, num_points)
    field = element.field_pattern(
        np.full_like(azimuth, pi / 2.0),
        azimuth,
        ExecutionContext("numpy", "double"),
        orientation=orientation,
    )
    power = np.abs(field.theta) ** 2 + np.abs(field.phi) ** 2
    axes = plt.subplot(projection="polar")
    axes.plot(azimuth, power)
    return axes


__all__ = ["plot_horizontal_power_pattern"]
