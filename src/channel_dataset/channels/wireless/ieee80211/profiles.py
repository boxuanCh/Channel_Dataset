"""Typed IEEE 802.11 profile resources and strict resource loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, pi
from types import MappingProxyType
from typing import Any

from channel_dataset.config import load_resource_document

_TGN_ROOT = "standards/wireless/ieee80211/03_0940r4"
_TGN_RESOURCES = {name: f"{_TGN_ROOT}/model_{name.lower()}.yaml" for name in "ABCDEF"}


def _freeze_resource(value: Any) -> Any:
    """Recursively freeze mappings and sequences loaded from a resource."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_resource(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_resource(item) for item in value)
    return value


def _physical_checksum(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class IEEE80211ClusterProfile:
    """One angular cluster contributing power to one or more delay taps.

    Parameters
    ----------
    cluster_id : int
        Positive cluster identifier from the source table.
    tap_indices : tuple of int
        One-based indices of taps receiving this cluster contribution.
    powers_db : tuple of float
        Contribution powers in decibels, aligned with ``tap_indices``.
    mean_aoa_rad, mean_aod_rad : float
        Mean azimuth angles of arrival and departure in radians.
    rx_angular_spread_rad, tx_angular_spread_rad : float
        RMS azimuth angular spreads in radians.
    """

    cluster_id: int
    tap_indices: tuple[int, ...]
    powers_db: tuple[float, ...]
    mean_aoa_rad: float
    mean_aod_rad: float
    rx_angular_spread_rad: float
    tx_angular_spread_rad: float

    def __post_init__(self) -> None:
        if self.cluster_id <= 0:
            raise ValueError("cluster_id must be positive")
        if not self.tap_indices or len(self.tap_indices) != len(self.powers_db):
            raise ValueError("cluster tap indices and powers must be nonempty and aligned")
        if len(set(self.tap_indices)) != len(self.tap_indices) or min(self.tap_indices) <= 0:
            raise ValueError("cluster tap indices must be unique positive integers")
        values = (
            *self.powers_db,
            self.mean_aoa_rad,
            self.mean_aod_rad,
            self.rx_angular_spread_rad,
            self.tx_angular_spread_rad,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("cluster values must be finite")
        if self.rx_angular_spread_rad <= 0 or self.tx_angular_spread_rad <= 0:
            raise ValueError("cluster angular spreads must be positive")


@dataclass(frozen=True, slots=True)
class IEEE80211TapProfile:
    """Resolved delay tap with its normalized power and contributing clusters."""

    tap_index: int
    delay_s: float
    power_linear: float
    cluster_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IEEE80211Profile:
    """Validated IEEE 802.11 TGn A-F profile with source provenance.

    Notes
    -----
    Overlapping cluster powers are summed in linear units and the resulting tap
    powers are normalized over the full delay profile. Source values remain in
    :attr:`clusters` for diagnostics and independent review.
    """

    name: str
    delays_s: tuple[float, ...]
    clusters: tuple[IEEE80211ClusterProfile, ...]
    rms_delay_spread_s: float
    breakpoint_distance_m: float
    shadow_std_los_db: float
    shadow_std_nlos_db: float
    los_k_factor_db: float
    source_document: str
    source_revision: str
    source_table: str
    source_url: str
    review_status: str
    checksum: str

    def __post_init__(self) -> None:
        if self.name not in set("ABCDEF"):
            raise ValueError("profile name must be A through F")
        if not self.delays_s or self.delays_s[0] != 0.0:
            raise ValueError("profile delays must begin at zero")
        if any(not isfinite(value) or value < 0 for value in self.delays_s):
            raise ValueError("profile delays must be finite and nonnegative")
        if tuple(sorted(self.delays_s)) != self.delays_s:
            raise ValueError("profile delays must be sorted")
        if len({cluster.cluster_id for cluster in self.clusters}) != len(self.clusters):
            raise ValueError("cluster identifiers must be unique")
        covered: set[int] = set()
        for cluster in self.clusters:
            if max(cluster.tap_indices) > len(self.delays_s):
                raise ValueError("cluster membership references a nonexistent tap")
            covered.update(cluster.tap_indices)
        if covered != set(range(1, len(self.delays_s) + 1)):
            raise ValueError("every delay tap must belong to at least one cluster")
        if self.review_status not in {"pending_independent_review", "independently_reviewed"}:
            raise ValueError("invalid resource review status")
        if len(self.checksum) != 64:
            raise ValueError("resource checksum must be SHA-256 hexadecimal text")

    @property
    def raw_tap_powers_linear(self) -> tuple[float, ...]:
        """Return aggregate cluster powers before profile normalization."""
        raw = [0.0] * len(self.delays_s)
        for cluster in self.clusters:
            for index, power_db in zip(cluster.tap_indices, cluster.powers_db, strict=True):
                raw[index - 1] += 10.0 ** (power_db / 10.0)
        return tuple(raw)

    @property
    def taps(self) -> tuple[IEEE80211TapProfile, ...]:
        """Return aggregate, unit-total-power taps in delay order."""

        raw = self.raw_tap_powers_linear
        memberships: list[list[int]] = [[] for _ in self.delays_s]
        for cluster in self.clusters:
            for index in cluster.tap_indices:
                memberships[index - 1].append(cluster.cluster_id)
        total = sum(raw)
        return tuple(
            IEEE80211TapProfile(index + 1, delay, power / total, tuple(memberships[index]))
            for index, (delay, power) in enumerate(zip(self.delays_s, raw, strict=True))
        )


@dataclass(frozen=True, slots=True)
class TGacBandwidthRule:
    """TGac bandwidth and PDP-spacing mapping from 09/0308r12."""

    bandwidth_hz: tuple[float, ...]
    delay_reduction_factors: tuple[int, ...]
    review_status: str

    def factor(self, bandwidth_hz: float) -> int:
        """Return the delay reduction factor for an exact supported bandwidth."""
        try:
            return self.delay_reduction_factors[self.bandwidth_hz.index(float(bandwidth_hz))]
        except ValueError as error:
            raise ValueError("unsupported TGac bandwidth") from error


@dataclass(frozen=True, slots=True)
class TGacMultiUserRule:
    """TGac multiuser antenna, direction, and independence rules."""

    maximum_ap_antennas: int
    maximum_station_antennas: int
    directions: tuple[str, ...]
    shared_ap_side_covariance: bool
    independent_user_fading: bool
    review_status: str


@dataclass(frozen=True, slots=True)
class TGahBandwidthRule:
    """TGah bandwidth and PDP-spacing mapping from 11/0968r4."""

    bandwidth_hz: tuple[float, ...]
    delay_reduction_factors: tuple[int, ...]
    review_status: str

    def factor(self, bandwidth_hz: float) -> int:
        """Return the delay reduction factor for an exact supported bandwidth."""
        try:
            return self.delay_reduction_factors[self.bandwidth_hz.index(float(bandwidth_hz))]
        except ValueError as error:
            raise ValueError("unsupported TGah bandwidth") from error


@dataclass(frozen=True, slots=True)
class TGahPathLossProfile:
    """TGah deployment path-loss and shadow-fading parameters."""

    path_loss_exponents: Mapping[str, float]
    shadow_std_db: Mapping[str, float]
    station_to_station_additional_loss_db: float
    review_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_loss_exponents", MappingProxyType(dict(self.path_loss_exponents)))
        object.__setattr__(self, "shadow_std_db", MappingProxyType(dict(self.shadow_std_db)))


@dataclass(frozen=True, slots=True)
class TGaxIndoorProfile:
    """TGax indoor breakpoint and penetration parameters."""

    breakpoint_distance_m: Mapping[str, float]
    wall_loss_db: Mapping[str, float]
    floor_loss_coefficient_db: float
    floor_loss_exponent_offset: float
    review_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "breakpoint_distance_m", MappingProxyType(dict(self.breakpoint_distance_m)))
        object.__setattr__(self, "wall_loss_db", MappingProxyType(dict(self.wall_loss_db)))


@dataclass(frozen=True, slots=True)
class TGaxOutdoorProfile:
    """TGax/ITU outdoor formula coefficients and fixed calibration defaults."""

    scenarios: Mapping[str, Any]
    default_street_width_m: float
    default_average_building_height_m: float
    review_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenarios", _freeze_resource(self.scenarios))


def _load_rule_document(reference: str, name: str) -> dict[str, Any]:
    document = load_resource_document(reference)
    if document.get("schema_version") != 1 or document.get("name") != name:
        raise ValueError(f"invalid {name} resource identity")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("units_reviewed") is not True:
        raise ValueError(f"invalid {name} resource metadata")
    physical = {key: value for key, value in document.items() if key != "metadata"}
    if metadata.get("checksum") != _physical_checksum(physical):
        raise ValueError(f"{name} physical-table checksum mismatch")
    return document


def load_tgac_bandwidth_rule() -> TGacBandwidthRule:
    """Load the packaged TGac bandwidth transformation rule."""
    document = _load_rule_document(
        "standards/wireless/ieee80211/09_0308r12/bandwidth_rules.yaml",
        "tgac_bandwidth_rules",
    )
    if set(document) != {"schema_version", "name", "bandwidth_hz", "delay_reduction_factors", "metadata"}:
        raise ValueError("invalid TGac bandwidth resource fields")
    if len(document["bandwidth_hz"]) != len(document["delay_reduction_factors"]):
        raise ValueError("TGac bandwidth and factor tables must align")
    return TGacBandwidthRule(
        tuple(float(value) for value in document["bandwidth_hz"]),
        tuple(int(value) for value in document["delay_reduction_factors"]),
        document["metadata"]["review_status"],
    )


def load_tgac_multiuser_rule() -> TGacMultiUserRule:
    """Load the packaged TGac multiuser behavior rule."""
    document = _load_rule_document(
        "standards/wireless/ieee80211/09_0308r12/multiuser_rules.yaml",
        "tgac_multiuser_rules",
    )
    expected = {
        "schema_version", "name", "maximum_ap_antennas", "maximum_station_antennas",
        "directions", "shared_ap_side_covariance", "independent_user_fading", "metadata",
    }
    if set(document) != expected:
        raise ValueError("invalid TGac multiuser resource fields")
    return TGacMultiUserRule(
        int(document["maximum_ap_antennas"]), int(document["maximum_station_antennas"]),
        tuple(document["directions"]), bool(document["shared_ap_side_covariance"]),
        bool(document["independent_user_fading"]), document["metadata"]["review_status"],
    )


def load_tgah_bandwidth_rule() -> TGahBandwidthRule:
    """Load the packaged TGah bandwidth transformation rule."""
    document = _load_rule_document(
        "standards/wireless/ieee80211/11_0968r4/bandwidth_rules.yaml",
        "tgah_bandwidth_rules",
    )
    if set(document) != {"schema_version", "name", "bandwidth_hz", "delay_reduction_factors", "metadata"}:
        raise ValueError("invalid TGah bandwidth resource fields")
    return TGahBandwidthRule(
        tuple(float(value) for value in document["bandwidth_hz"]),
        tuple(int(value) for value in document["delay_reduction_factors"]),
        document["metadata"]["review_status"],
    )


def load_tgah_path_loss_profile() -> TGahPathLossProfile:
    """Load TGah deployment path-loss and shadowing parameters."""
    document = _load_rule_document(
        "standards/wireless/ieee80211/11_0968r4/pathloss_scenarios.yaml",
        "tgah_pathloss_scenarios",
    )
    if set(document) != {"schema_version", "name", "scenarios", "station_to_station_additional_loss_db", "metadata"}:
        raise ValueError("invalid TGah path-loss resource fields")
    scenarios = document["scenarios"]
    return TGahPathLossProfile(
        {key: float(value["path_loss_exponent"]) for key, value in scenarios.items()},
        {key: float(value["shadow_std_db"]) for key, value in scenarios.items()},
        float(document["station_to_station_additional_loss_db"]),
        document["metadata"]["review_status"],
    )


def load_tgax_indoor_profile() -> TGaxIndoorProfile:
    """Load TGax indoor breakpoint and penetration parameters."""
    document = _load_rule_document(
        "standards/wireless/ieee80211/14_0882r4/indoor.yaml", "tgax_indoor"
    )
    expected = {
        "schema_version", "name", "scenarios", "floor_loss_coefficient_db",
        "floor_loss_exponent_offset", "metadata",
    }
    if set(document) != expected:
        raise ValueError("invalid TGax indoor resource fields")
    return TGaxIndoorProfile(
        {key: float(value["breakpoint_distance_m"]) for key, value in document["scenarios"].items()},
        {key: float(value["wall_loss_db"]) for key, value in document["scenarios"].items()},
        float(document["floor_loss_coefficient_db"]),
        float(document["floor_loss_exponent_offset"]),
        document["metadata"]["review_status"],
    )


def load_tgax_outdoor_profile() -> TGaxOutdoorProfile:
    """Load TGax outdoor UMi/UMa formula resources."""
    document = _load_rule_document(
        "standards/wireless/ieee80211/14_0882r4/outdoor.yaml", "tgax_outdoor"
    )
    expected = {
        "schema_version", "name", "scenarios", "default_street_width_m",
        "default_average_building_height_m", "metadata",
    }
    if set(document) != expected:
        raise ValueError("invalid TGax outdoor resource fields")
    return TGaxOutdoorProfile(
        document["scenarios"], float(document["default_street_width_m"]),
        float(document["default_average_building_height_m"]),
        document["metadata"]["review_status"],
    )


def load_ieee80211_profile(name: str) -> IEEE80211Profile:
    """Load and strictly validate a packaged TGn A-F profile.

    Parameters
    ----------
    name : {"A", "B", "C", "D", "E", "F"}
        IEEE 802.11-03/940r4 model letter.
    """

    try:
        document = load_resource_document(_TGN_RESOURCES[name.upper()])
    except (AttributeError, KeyError) as error:
        raise ValueError(f"unsupported IEEE 802.11 profile: {name!r}") from error
    expected = {
        "schema_version", "name", "delays_ns", "clusters", "rms_delay_spread_ns",
        "breakpoint_distance_m", "shadow_std_los_db", "shadow_std_nlos_db",
        "los_k_factor_db", "metadata",
    }
    if set(document) != expected:
        raise ValueError(
            f"invalid IEEE 802.11 profile fields; missing={sorted(expected - set(document))}, "
            f"unknown={sorted(set(document) - expected)}"
        )
    metadata = document["metadata"]
    metadata_fields = {
        "source_document", "source_revision", "source_table", "source_url",
        "units_reviewed", "review_status", "checksum",
    }
    if not isinstance(metadata, dict) or set(metadata) != metadata_fields:
        raise ValueError("invalid IEEE 802.11 profile metadata fields")
    if document["schema_version"] != 1 or document["name"] != name.upper():
        raise ValueError("IEEE 802.11 profile schema version or name is invalid")
    cluster_fields = {"id", "tap_indices", "powers_db", "aoa_deg", "aod_deg", "rx_as_deg", "tx_as_deg"}
    clusters = []
    for item in document["clusters"]:
        if not isinstance(item, dict) or set(item) != cluster_fields:
            raise ValueError("invalid IEEE 802.11 cluster fields")
        clusters.append(
            IEEE80211ClusterProfile(
                int(item["id"]), tuple(item["tap_indices"]), tuple(float(x) for x in item["powers_db"]),
                float(item["aoa_deg"]) * pi / 180.0,
                float(item["aod_deg"]) * pi / 180.0,
                float(item["rx_as_deg"]) * pi / 180.0,
                float(item["tx_as_deg"]) * pi / 180.0,
            )
        )
    physical = {key: document[key] for key in sorted(expected - {"metadata"})}
    if metadata["checksum"] != _physical_checksum(physical):
        raise ValueError("IEEE 802.11 profile physical-table checksum mismatch")
    if metadata["units_reviewed"] is not True:
        raise ValueError("IEEE 802.11 profile units must be reviewed")
    return IEEE80211Profile(
        name=document["name"],
        delays_s=tuple(float(value) * 1e-9 for value in document["delays_ns"]),
        clusters=tuple(clusters),
        rms_delay_spread_s=float(document["rms_delay_spread_ns"]) * 1e-9,
        breakpoint_distance_m=float(document["breakpoint_distance_m"]),
        shadow_std_los_db=float(document["shadow_std_los_db"]),
        shadow_std_nlos_db=float(document["shadow_std_nlos_db"]),
        los_k_factor_db=float(document["los_k_factor_db"]),
        source_document=metadata["source_document"],
        source_revision=metadata["source_revision"],
        source_table=metadata["source_table"],
        source_url=metadata["source_url"],
        review_status=metadata["review_status"],
        checksum=metadata["checksum"],
    )


__all__ = [
    "IEEE80211ClusterProfile", "IEEE80211Profile", "IEEE80211TapProfile",
    "TGacBandwidthRule", "TGacMultiUserRule", "TGahBandwidthRule",
    "TGahPathLossProfile", "TGaxIndoorProfile", "TGaxOutdoorProfile",
    "load_ieee80211_profile", "load_tgac_bandwidth_rule",
    "load_tgac_multiuser_rule", "load_tgah_bandwidth_rule", "load_tgah_path_loss_profile",
    "load_tgax_indoor_profile", "load_tgax_outdoor_profile",
]
