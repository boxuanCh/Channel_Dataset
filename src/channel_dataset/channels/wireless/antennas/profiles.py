"""Strict external profiles for named antenna layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import pi
from types import MappingProxyType
from typing import Any

from channel_dataset.config import load_resource_document

from .arrays import AntennaPanel, MultiPanelArray
from .elements import IsotropicAntennaElement, ThreeGPPAntennaElement


@dataclass(frozen=True, slots=True)
class AntennaLayout:
    """Typed named antenna layout loaded from an external resource.

    Parameters
    ----------
    name : str
        Stable profile name.
    array : MultiPanelArray
        Constructed geometry and element pattern.
    metadata : mapping
        Read-only source and review metadata.
    """

    name: str
    array: MultiPanelArray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def _strict_mapping(value: Any, name: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")
    return value


def parse_antenna_layout(document: Mapping[str, Any]) -> AntennaLayout:
    """Validate a version-1 antenna layout document.

    Parameters
    ----------
    document : mapping
        Decoded YAML or JSON document. Unknown fields are rejected at every
        level.

    Returns
    -------
    AntennaLayout
        Typed element model and multi-panel geometry.
    """

    root = _strict_mapping(
        dict(document),
        "layout",
        {"schema_version", "name", "element", "panel", "array", "metadata"},
        {"schema_version", "name", "element", "panel", "array", "metadata"},
    )
    if isinstance(root["schema_version"], bool) or root["schema_version"] != 1:
        raise ValueError("unsupported antenna layout schema_version")
    if not isinstance(root["name"], str) or not root["name"]:
        raise ValueError("layout.name must be a non-empty string")
    metadata = _strict_mapping(root["metadata"], "metadata", {"source", "review_status"}, {"source", "review_status"})
    if any(not isinstance(metadata[key], str) or not metadata[key] for key in metadata):
        raise ValueError("antenna metadata values must be non-empty strings")
    element_config = _strict_mapping(
        root["element"],
        "element",
        {
            "kind",
            "maximum_gain_dbi",
            "vertical_beamwidth_degrees",
            "horizontal_beamwidth_degrees",
            "vertical_sidelobe_attenuation_db",
            "front_back_attenuation_db",
        },
        {"kind"},
    )
    kind = element_config["kind"]
    if kind == "isotropic":
        unknown_for_kind = set(element_config) - {"kind", "maximum_gain_dbi"}
        if unknown_for_kind:
            raise ValueError(f"unsupported isotropic element fields: {sorted(unknown_for_kind)}")
        element = IsotropicAntennaElement(float(element_config.get("maximum_gain_dbi", 0.0)))
    elif kind == "3gpp":
        element = ThreeGPPAntennaElement(
            maximum_gain_dbi=float(element_config.get("maximum_gain_dbi", 8.0)),
            vertical_beamwidth_rad=float(element_config.get("vertical_beamwidth_degrees", 65.0)) * pi / 180.0,
            horizontal_beamwidth_rad=float(element_config.get("horizontal_beamwidth_degrees", 65.0)) * pi / 180.0,
            vertical_sidelobe_attenuation_db=float(element_config.get("vertical_sidelobe_attenuation_db", 30.0)),
            front_back_attenuation_db=float(element_config.get("front_back_attenuation_db", 30.0)),
        )
    else:
        raise ValueError("element.kind must be 'isotropic' or '3gpp'")
    panel_config = _strict_mapping(
        root["panel"],
        "panel",
        {
            "rows",
            "columns",
            "vertical_spacing_wavelengths",
            "horizontal_spacing_wavelengths",
            "polarization_slant_angles_degrees",
        },
        {
            "rows",
            "columns",
            "vertical_spacing_wavelengths",
            "horizontal_spacing_wavelengths",
            "polarization_slant_angles_degrees",
        },
    )
    slants = panel_config["polarization_slant_angles_degrees"]
    if not isinstance(slants, list):
        raise TypeError("polarization_slant_angles_degrees must be a list")
    panel = AntennaPanel(
        rows=panel_config["rows"],
        columns=panel_config["columns"],
        vertical_spacing_wavelengths=float(panel_config["vertical_spacing_wavelengths"]),
        horizontal_spacing_wavelengths=float(panel_config["horizontal_spacing_wavelengths"]),
        polarization_slant_angles_rad=tuple(float(value) * pi / 180.0 for value in slants),
        element=element,
    )
    array_config = _strict_mapping(
        root["array"],
        "array",
        {
            "panel_rows",
            "panel_columns",
            "vertical_panel_spacing_wavelengths",
            "horizontal_panel_spacing_wavelengths",
        },
        {"panel_rows", "panel_columns"},
    )
    return AntennaLayout(
        name=root["name"],
        array=MultiPanelArray(
            panel=panel,
            panel_rows=array_config["panel_rows"],
            panel_columns=array_config["panel_columns"],
            vertical_panel_spacing_wavelengths=array_config.get("vertical_panel_spacing_wavelengths"),
            horizontal_panel_spacing_wavelengths=array_config.get("horizontal_panel_spacing_wavelengths"),
        ),
        metadata=metadata,
    )


def load_antenna_layout(reference: str) -> AntennaLayout:
    """Load and validate a named antenna layout resource.

    Parameters
    ----------
    reference : str
        Package resource reference accepted by
        :func:`channel_dataset.config.load_resource_document`.

    Returns
    -------
    AntennaLayout
        Strictly validated layout.
    """

    return parse_antenna_layout(load_resource_document(reference))


__all__ = ["AntennaLayout", "load_antenna_layout", "parse_antenna_layout"]
