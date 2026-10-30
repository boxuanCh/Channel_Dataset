"""3GPP scenario geometry constraints and common scenario interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.topology import LinkGeometry, compute_link_geometry
from channel_dataset.channels.wireless.types import TopologyBatch
from channel_dataset.config import load_resource_document

_GEOMETRY_RESOURCE = "standards/wireless/tr_38_901/v16_1/scenario_geometry.yaml"
_SCENARIO_NAMES = frozenset({"RMa", "UMa", "UMi", "InH", "InF"})


@runtime_checkable
class Scenario(Protocol):
    """Interface separating scenario rules from topology generation."""

    @property
    def name(self) -> str:
        """str: Stable scenario identifier."""

        ...

    def validate_topology(self, topology: TopologyBatch) -> LinkGeometry:
        """Validate applicability and return derived link geometry."""

        ...

    def parameter(self, name: str) -> Any:
        """Return a named, versioned scenario parameter."""

        ...

    def los_probability(self, geometry: LinkGeometry) -> Any:
        """Return per-link LoS probability when implemented by the phase."""

        ...

    def path_loss_db(self, topology: TopologyBatch, los_mask: Any) -> Any:
        """Return per-link path loss when implemented by the phase."""

        ...


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Validated geometry inputs shared by 3GPP scenarios.

    Parameters
    ----------
    name : {"RMa", "UMa", "UMi", "InH", "InF"}
        Scenario family.
    carrier_frequency_hz : float
        Carrier frequency checked against the versioned resource.
    room_width_m, room_length_m, ceiling_height_m : float, optional
        Required for indoor scenarios.
    clutter_density : float, optional
        Required for InF and constrained to ``[0, 1]``.
    effective_clutter_height_m : float, optional
        Required for InF and below the ceiling height.
    """

    name: str
    carrier_frequency_hz: float
    room_width_m: float | None = None
    room_length_m: float | None = None
    ceiling_height_m: float | None = None
    clutter_density: float | None = None
    effective_clutter_height_m: float | None = None

    def __post_init__(self) -> None:
        if self.name not in _SCENARIO_NAMES:
            raise ValueError(f"unsupported scenario: {self.name!r}")
        if not isfinite(float(self.carrier_frequency_hz)) or self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be finite and positive")
        indoor_values = (self.room_width_m, self.room_length_m, self.ceiling_height_m)
        if self.name in {"InH", "InF"}:
            if any(value is None or not isfinite(float(value)) or value <= 0.0 for value in indoor_values):
                raise ValueError("indoor scenarios require positive room dimensions and ceiling height")
        elif any(value is not None for value in indoor_values):
            raise ValueError("room geometry is supported only for indoor scenarios")
        if self.name == "InF":
            area = float(self.room_width_m) * float(self.room_length_m)
            if not 20.0 <= area <= 160000.0:
                raise ValueError("InF room area must lie in [20, 160000] square metres")
            if not 5.0 <= float(self.ceiling_height_m) <= 25.0:
                raise ValueError("InF ceiling_height_m must lie in [5, 25]")
            if self.clutter_density is None or not 0.0 <= self.clutter_density <= 1.0:
                raise ValueError("InF clutter_density must lie in [0, 1]")
            if (
                self.effective_clutter_height_m is None
                or self.effective_clutter_height_m < 0.0
                or self.effective_clutter_height_m >= float(self.ceiling_height_m)
            ):
                raise ValueError("InF effective clutter height must be below the ceiling")
        elif self.clutter_density is not None or self.effective_clutter_height_m is not None:
            raise ValueError("clutter parameters are supported only for InF")


def load_scenario_geometry(name: str) -> Mapping[str, Any]:
    """Load versioned geometry limits for a 3GPP scenario.

    Parameters
    ----------
    name : str
        One of ``RMa``, ``UMa``, ``UMi``, ``InH``, or ``InF``.

    Returns
    -------
    mapping
        Read-only geometry and applicability parameters.
    """

    if name not in _SCENARIO_NAMES:
        raise ValueError(f"unsupported scenario: {name!r}")
    document = load_resource_document(_GEOMETRY_RESOURCE)
    expected = {
        "schema_version", "standard", "standard_version", "tables", "scenarios",
        "source_url", "transcription_date", "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1:
        raise ValueError("invalid scenario geometry resource")
    scenarios = document["scenarios"]
    if not isinstance(scenarios, dict) or set(scenarios) != _SCENARIO_NAMES:
        raise ValueError("scenario geometry resource has incomplete scenarios")
    return MappingProxyType(dict(scenarios[name]))


class GeometryScenario:
    """Scenario implementation limited to Phase 10 geometry applicability.

    Parameters
    ----------
    config : ScenarioConfig
        Scenario name, frequency, and optional room/factory geometry.

    Notes
    -----
    Probability and path-loss equations are intentionally introduced in Phase
    11. Their interface methods raise :class:`NotImplementedError` here so
    topology generation cannot accidentally sample large-scale effects.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self._parameters = load_scenario_geometry(config.name)
        low = float(self._parameters["minimum_frequency_hz"])
        high = float(self._parameters["maximum_frequency_hz"])
        if not low <= config.carrier_frequency_hz <= high:
            raise ValueError("carrier frequency is outside scenario applicability")

    @property
    def name(self) -> str:
        """str: Stable scenario identifier."""

        return self.config.name

    def parameter(self, name: str) -> Any:
        """Return a named geometry parameter.

        Parameters
        ----------
        name : str
            Exact resource field name.

        Returns
        -------
        object
            Immutable parameter value.

        Raises
        ------
        KeyError
            If the parameter is not defined.
        """

        return self._parameters[name]

    def validate_topology(self, topology: TopologyBatch) -> LinkGeometry:
        """Validate distances, heights, and indoor state.

        Parameters
        ----------
        topology : TopologyBatch
            Candidate endpoints in metres.

        Returns
        -------
        LinkGeometry
            Derived geometry when every endpoint/link is applicable.
        """

        backend = get_backend(topology.metadata.backend)
        geometry = compute_link_geometry(topology)
        distance = geometry.distance_2d_m
        minimum = float(self._parameters["minimum_distance_2d_m"])
        maximum = float(self._parameters["maximum_distance_2d_m"])
        tolerance = 1.0e-9
        if (
            backend.to_scalar(backend.min(distance)) < minimum - tolerance
            or backend.to_scalar(backend.max(distance)) > maximum + tolerance
        ):
            raise ValueError("2D link distance is outside scenario applicability")
        tx_height = topology.tx_positions_m[..., 2]
        rx_height = topology.rx_positions_m[..., 2]
        for values, endpoint in ((tx_height, "bs"), (rx_height, "ut")):
            low = float(self._parameters[f"minimum_{endpoint}_height_m"])
            high = float(self._parameters[f"maximum_{endpoint}_height_m"])
            if (
                backend.to_scalar(backend.min(values)) < low - tolerance
                or backend.to_scalar(backend.max(values)) > high + tolerance
            ):
                raise ValueError(f"{endpoint.upper()} height is outside scenario applicability")
        if self._parameters["indoor_requirement"] == "required":
            indoor_count = backend.to_scalar(
                backend.sum(backend.astype(topology.rx_indoor, "int64"))
            )
            expected = topology.shape[0] * topology.shape[1]
            if indoor_count != expected:
                raise ValueError("indoor scenario requires every receiver to be indoor")
        return geometry

    def los_probability(self, geometry: LinkGeometry) -> Any:
        """Raise until Phase 11 introduces scenario probability equations."""

        raise NotImplementedError("LoS probability is implemented in Phase 11")

    def path_loss_db(self, geometry: LinkGeometry, los_mask: Any) -> Any:
        """Raise until Phase 11 introduces scenario path-loss equations."""

        raise NotImplementedError("path loss is implemented in Phase 11")


def create_scenario(config: ScenarioConfig) -> GeometryScenario:
    """Construct a standard scenario from typed configuration.

    Parameters
    ----------
    config : ScenarioConfig
        Validated scenario inputs.

    Returns
    -------
    GeometryScenario
        Scenario geometry, probability, path-loss, and LSP implementation.
    """

    from channel_dataset.channels.wireless.large_scale import LargeScaleScenario

    return LargeScaleScenario(config)


__all__ = [
    "GeometryScenario", "Scenario", "ScenarioConfig", "create_scenario",
    "load_scenario_geometry",
]
