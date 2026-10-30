"""Medium-independent multipath value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class Direction:
    """Azimuth/elevation direction in a model coordinate frame.

    Parameters
    ----------
    azimuth_rad : float
        Azimuth angle in radians.
    elevation_rad : float
        Elevation angle in radians, measured from the horizontal plane.
    """

    azimuth_rad: float
    elevation_rad: float


@dataclass(frozen=True, slots=True)
class PropagationPath:
    """One resolved scalar contribution to a channel realization.

    Parameters
    ----------
    path_id : str
        Identifier unique within its containing cluster.
    complex_gain : complex
        Complex scalar gain, including magnitude and phase.
    delay_s : float
        Nonnegative propagation delay in seconds.
    doppler_hz : float, default=0
        Doppler shift in hertz.
    departure, arrival : Direction, optional
        Departure and arrival directions.
    metadata : mapping, optional
        Read-only path annotations.

    Raises
    ------
    ValueError
        If ``delay_s`` is negative.
    """

    path_id: str
    complex_gain: complex
    delay_s: float
    doppler_hz: float = 0.0
    departure: Direction | None = None
    arrival: Direction | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.delay_s < 0:
            raise ValueError("delay_s cannot be negative")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class PathCluster:
    """Physically or statistically related scalar propagation paths.

    Parameters
    ----------
    cluster_id : str
        Stable cluster identifier.
    paths : tuple of PropagationPath
        Scalar paths belonging to the cluster.
    metadata : mapping, optional
        Read-only cluster annotations.

    Raises
    ------
    ValueError
        If path identifiers are not unique within the cluster.
    """

    cluster_id: str
    paths: tuple[PropagationPath, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", tuple(self.paths))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        path_ids = [path.path_id for path in self.paths]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("path_id values must be unique within a cluster")
