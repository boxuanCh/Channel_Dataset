"""Validated loaders for versioned TR 38.811 resources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from channel_dataset.backends import get_backend
from channel_dataset.channels.wireless.tr38811.config import TR38811DeploymentConfig
from channel_dataset.channels.wireless.tr38811.geometry import EARTH_RADIUS_M
from channel_dataset.channels.wireless.tr38811.types import NTNTopologyBatch
from channel_dataset.config import load_resource_document

_ROOT = "standards/wireless/tr_38_811/v15_4"


@dataclass(frozen=True, slots=True)
class NTNDelayProfile:
    """TDL/CDL path data and standard provenance."""

    name: str
    table: str
    delays: tuple[float, ...]
    powers_db: tuple[float, ...]
    component_types: tuple[str, ...]
    default_k_factor_db: float | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.delays)
        if count == 0 or len(self.powers_db) != count or len(self.component_types) != count:
            raise ValueError("profile columns must have equal nonzero length")
        if self.delays[0] != 0.0 or any(value < 0.0 for value in self.delays):
            raise ValueError("profile delays must be nonnegative and begin at zero")
        if any(not isfinite(value) for value in self.delays + self.powers_db):
            raise ValueError("profile values must be finite")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class NTNCDLProfile:
    """Cluster means, spreads, powers, and provenance for an NTN CDL."""

    name: str
    table: str
    delays: tuple[float, ...]
    powers_db: tuple[float, ...]
    aod_degrees: tuple[float, ...]
    aoa_degrees: tuple[float, ...]
    zod_degrees: tuple[float, ...]
    zoa_degrees: tuple[float, ...]
    asd_degrees: float
    asa_degrees: float
    zsd_degrees: float
    zsa_degrees: float
    xpr_db: float
    specular_power_db: float | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.delays)
        columns = (
            self.powers_db, self.aod_degrees, self.aoa_degrees,
            self.zod_degrees, self.zoa_degrees,
        )
        if count == 0 or any(len(column) != count for column in columns):
            raise ValueError("CDL resource columns must have equal nonzero length")
        if self.delays[0] != 0.0 or any(value < 0.0 for value in self.delays):
            raise ValueError("CDL delays must be nonnegative and begin at zero")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def has_los(self) -> bool:
        """Return whether the profile contains a specular LOS component."""

        return self.specular_power_db is not None


def load_tdl_profile(name: str) -> NTNDelayProfile:
    """Load NTN-TDL-A through NTN-TDL-D from packaged resources."""

    if name not in {"NTN-TDL-A", "NTN-TDL-B", "NTN-TDL-C", "NTN-TDL-D"}:
        raise ValueError(f"unsupported NTN TDL profile: {name!r}")
    document = load_resource_document(f"{_ROOT}/tdl_{name[-1].lower()}.yaml")
    expected = {
        "schema_version", "name", "standard", "standard_version", "table",
        "reference_elevation_degrees", "delays", "powers_db", "component_types",
        "default_k_factor_db", "source_url", "transcription_date", "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1 or document["name"] != name:
        raise ValueError("invalid NTN TDL resource")
    return NTNDelayProfile(
        name=name,
        table=str(document["table"]),
        delays=tuple(float(value) for value in document["delays"]),
        powers_db=tuple(float(value) for value in document["powers_db"]),
        component_types=tuple(str(value) for value in document["component_types"]),
        default_k_factor_db=(
            None if document["default_k_factor_db"] is None
            else float(document["default_k_factor_db"])
        ),
        metadata={
            key: document[key]
            for key in (
                "standard", "standard_version", "reference_elevation_degrees",
                "source_url", "transcription_date", "review_status",
            )
        },
    )


def load_cdl_profile(name: str) -> NTNCDLProfile:
    """Load NTN-CDL-A through NTN-CDL-D from packaged resources."""

    if name not in {"NTN-CDL-A", "NTN-CDL-B", "NTN-CDL-C", "NTN-CDL-D"}:
        raise ValueError(f"unsupported NTN CDL profile: {name!r}")
    document = load_resource_document(f"{_ROOT}/cdl_{name[-1].lower()}.yaml")
    expected = {
        "schema_version", "name", "standard", "standard_version", "table",
        "reference_elevation_degrees", "delays", "powers_db", "aod_degrees",
        "aoa_degrees", "zod_degrees", "zoa_degrees", "cluster_spreads_degrees",
        "xpr_db", "specular_power_db", "source_url", "transcription_date",
        "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1 or document["name"] != name:
        raise ValueError("invalid NTN CDL resource")
    spreads = document["cluster_spreads_degrees"]
    if not isinstance(spreads, dict) or set(spreads) != {"asd", "asa", "zsd", "zsa"}:
        raise ValueError("invalid NTN CDL cluster spreads")
    vector_names = ("delays", "powers_db", "aod_degrees", "aoa_degrees", "zod_degrees", "zoa_degrees")
    vectors = {key: tuple(float(value) for value in document[key]) for key in vector_names}
    return NTNCDLProfile(
        name=name,
        table=str(document["table"]),
        delays=vectors["delays"],
        powers_db=vectors["powers_db"],
        aod_degrees=vectors["aod_degrees"],
        aoa_degrees=vectors["aoa_degrees"],
        zod_degrees=vectors["zod_degrees"],
        zoa_degrees=vectors["zoa_degrees"],
        asd_degrees=float(spreads["asd"]),
        asa_degrees=float(spreads["asa"]),
        zsd_degrees=float(spreads["zsd"]),
        zsa_degrees=float(spreads["zsa"]),
        xpr_db=float(document["xpr_db"]),
        specular_power_db=(
            None if document["specular_power_db"] is None
            else float(document["specular_power_db"])
        ),
        metadata={
            key: document[key]
            for key in (
                "standard", "standard_version", "reference_elevation_degrees",
                "source_url", "transcription_date", "review_status",
            )
        },
    )


def load_resource(name: str) -> Mapping[str, Any]:
    """Load one general TR 38.811 resource document."""

    if name not in {"manifest", "deployments", "los_probability", "large_scale"}:
        raise ValueError(f"unsupported TR 38.811 resource: {name!r}")
    return MappingProxyType(dict(load_resource_document(f"{_ROOT}/{name}.yaml")))


def validate_deployment_applicability(
    config: TR38811DeploymentConfig,
    *,
    model_kind: str,
) -> Mapping[str, Any]:
    """Validate a model against the packaged D1--D5 applicability table.

    Parameters
    ----------
    config:
        Common deployment, carrier-frequency, and bandwidth settings.
    model_kind:
        One of ``"flat"`` or ``"frequency_selective"``. This identifies the
        requested formulation for diagnostics; the D1--D5 table's
        ``link_model`` entry is a calibration classification, not an
        exclusivity constraint.

    Returns
    -------
    Mapping[str, Any]
        The immutable resource row used for validation.

    Raises
    ------
    ValueError
        If the model kind is invalid or the carrier frequency or bandwidth is
        outside the deployment's applicability range.
    """

    if model_kind not in {"flat", "frequency_selective"}:
        raise ValueError("model_kind must be 'flat' or 'frequency_selective'")
    document = load_resource("deployments")
    deployments = document.get("deployments")
    if not isinstance(deployments, dict):
        raise TypeError("invalid TR 38.811 deployment resource")
    row = deployments.get(config.deployment)
    if not isinstance(row, dict):
        raise TypeError(f"invalid deployment row for {config.deployment!r}")
    frequency = config.carrier_frequency_hz
    if not float(row["frequency_min_hz"]) <= frequency <= float(row["frequency_max_hz"]):
        raise ValueError(
            f"carrier_frequency_hz is outside the {config.deployment} applicability range"
        )
    if config.bandwidth_hz > float(row["maximum_bandwidth_hz"]):
        raise ValueError(
            f"bandwidth_hz exceeds the {config.deployment} maximum bandwidth"
        )
    return MappingProxyType(dict(row))


def validate_topology_applicability(
    topology: NTNTopologyBatch,
    deployment: str,
) -> None:
    """Validate platform kind and altitude against one D1--D5 profile.

    The caller-supplied state vector remains authoritative; this function only
    rejects states outside the selected deployment's declared applicability.
    """

    document = load_resource("deployments")
    deployments = document.get("deployments")
    if not isinstance(deployments, dict):
        raise TypeError("invalid TR 38.811 deployment resource")
    row = deployments.get(deployment)
    if not isinstance(row, dict):
        raise TypeError(f"invalid deployment row for {deployment!r}")
    tx_is_platform = topology.tx_endpoint_kind != "ground"
    actual_kind = topology.tx_endpoint_kind if tx_is_platform else topology.rx_endpoint_kind
    expected_kind = "haps" if row["platform"] == "haps" else "satellite"
    if actual_kind != expected_kind:
        raise ValueError(
            f"deployment {deployment} requires a {expected_kind} platform endpoint"
        )

    positions = (
        topology.tx_positions_ecef_m if tx_is_platform else topology.rx_positions_ecef_m
    )
    backend = get_backend(topology.metadata.backend)
    radii = backend.sqrt(backend.sum(positions * positions, axis=-1))
    flat = backend.reshape(radii, (-1,))
    altitudes = [float(flat[index]) - EARTH_RADIUS_M for index in range(backend.shape(flat)[0])]
    lower = float(row["altitude_min_m"])
    upper = float(row["altitude_max_m"])
    tolerance = 1.0 if lower != upper else 100_000.0
    if any(value < lower - tolerance or value > upper + tolerance for value in altitudes):
        raise ValueError(
            f"platform altitude is outside the {deployment} applicability range"
        )


__all__ = [
    "NTNCDLProfile", "NTNDelayProfile", "load_cdl_profile", "load_resource",
    "load_tdl_profile", "validate_deployment_applicability",
    "validate_topology_applicability",
]
