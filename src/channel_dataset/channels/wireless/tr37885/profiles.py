"""Versioned TR 37.885 profile loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from channel_dataset.config import load_resource_document

_ROOT = "standards/wireless/tr_37_885/v15_3"


@dataclass(frozen=True, slots=True)
class V2XCDLProfile:
    """One validated fixed urban V2X CDL profile.

    Parameters
    ----------
    name : str
        Stable profile name (urban LOS, NLOSv, or NLOS).
    delays_s, powers_db : tuple of float
        Excess delays in seconds and path powers in decibels.
    aod_degrees, aoa_degrees, zod_degrees, zoa_degrees : tuple of float
        Mean departure and arrival angles in degrees.
    xpr_db : float
        Per-cluster cross-polarization power ratio in decibels.
    standard_version, review_status : str
        Source revision and independent-review state.
    source, units : mapping
        Immutable provenance and unit declarations.
    """

    name: str
    delays_s: tuple[float, ...]
    powers_db: tuple[float, ...]
    aod_degrees: tuple[float, ...]
    aoa_degrees: tuple[float, ...]
    zod_degrees: tuple[float, ...]
    zoa_degrees: tuple[float, ...]
    xpr_db: float
    standard_version: str
    review_status: str
    source: Mapping[str, Any]
    units: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.delays_s)
        columns = (
            self.powers_db, self.aod_degrees, self.aoa_degrees,
            self.zod_degrees, self.zoa_degrees,
        )
        if count == 0 or any(len(column) != count for column in columns):
            raise ValueError("V2X CDL columns must have one value per tap")
        if any(not isfinite(value) or value < 0.0 for value in self.delays_s):
            raise ValueError("V2X CDL delays must be finite and nonnegative")
        if any(not isfinite(value) for column in columns for value in column):
            raise ValueError("V2X CDL powers and angles must be finite")
        if not isfinite(self.xpr_db):
            raise ValueError("V2X CDL xpr_db must be finite")
        if self.standard_version != "15.3.0":
            raise ValueError("V2X CDL standard_version must be '15.3.0'")
        if self.review_status not in {"pending_independent_review", "independently_reviewed"}:
            raise ValueError("invalid V2X CDL review status")
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))


def load_urban_parameters() -> dict:
    """Load the strict urban scenario resource."""

    document = load_resource_document(f"{_ROOT}/urban.yaml")
    expected = {
        "schema_version", "standard", "standard_version", "scenario", "source", "units",
        "grid", "vehicle_types", "mobility", "link_state", "path_loss", "fast_fading",
        "vehicle_antenna",
        "transcription_date", "review_status",
    }
    if set(document) != expected or document["schema_version"] != 1:
        raise ValueError("invalid TR 37.885 urban resource")
    return document


def load_cdl_profile(condition: str) -> V2XCDLProfile:
    """Load an urban LOS, NLOSv, or NLOS fixed CDL profile."""

    if condition not in {"los", "nlosv", "nlos"}:
        raise ValueError("condition must be los, nlosv, or nlos")
    document = load_resource_document(f"{_ROOT}/cdl_urban_{condition}.yaml")
    expected = {
        "schema_version", "standard", "standard_version", "profile", "source", "units",
        "per_cluster", "taps", "transcription_date", "review_status",
    }
    if set(document) != expected or document["profile"] != f"urban_{condition}":
        raise ValueError("invalid TR 37.885 CDL resource")
    taps = document["taps"]
    if not isinstance(taps, list) or not taps or any(len(row) != 6 for row in taps):
        raise ValueError("invalid TR 37.885 CDL taps")
    columns = tuple(tuple(float(row[index]) for row in taps) for index in range(6))
    cluster = document["per_cluster"]
    if not isinstance(cluster, dict) or set(cluster) != {
        "asd_deg", "asa_deg", "zsd_deg", "zsa_deg", "xpr_db",
    }:
        raise ValueError("invalid TR 37.885 per-cluster metadata")
    return V2XCDLProfile(
        name=document["profile"],
        delays_s=tuple(value * 1.0e-9 for value in columns[0]),
        powers_db=columns[1],
        aod_degrees=columns[2],
        aoa_degrees=columns[3],
        zod_degrees=columns[4],
        zoa_degrees=columns[5],
        xpr_db=float(cluster["xpr_db"]),
        standard_version=document["standard_version"],
        review_status=document["review_status"],
        source=document["source"],
        units=document["units"],
    )


__all__ = ["V2XCDLProfile", "load_cdl_profile", "load_urban_parameters"]
