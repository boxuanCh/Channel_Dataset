from __future__ import annotations

import pytest

from channel_dataset.channels.wireless.tr37885 import (
    load_cdl_profile,
    load_urban_parameters,
)
from channel_dataset.config import load_resource_document, resource_checksum

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885, pytest.mark.reference]

_ROOT = "standards/wireless/tr_37_885/v15_3"
_CHECKSUMS = {
    "manifest.yaml": "8f92815a9e32ce879a36c59dcc4f0fedab4e79c244560140578df476834c88cd",
    "urban.yaml": "17ff60758573ff93bef359974e94b5cfce93769dfa7924c7460596414dde723c",
    "cdl_urban_los.yaml": "3ca3ed0c300fffc73985816fa3764acf05c0acfbe0518396e0eb64690260dc59",
    "cdl_urban_nlos.yaml": "867c8d5f946747c0bca2c4e05c73902e04a83dff25aa8d5aba674bcee05b533a",
    "cdl_urban_nlosv.yaml": "af054c4b4ebf639f5884d4be05ab5c8e4b213ba603d0bfa1f3c40ea18ea2aa7d",
}


@pytest.mark.parametrize(("filename", "checksum"), _CHECKSUMS.items())
def test_resource_checksum_and_review_status(filename: str, checksum: str) -> None:
    reference = f"{_ROOT}/{filename}"
    document = load_resource_document(reference)
    assert resource_checksum(reference) == checksum
    assert document["review_status"] == "pending_independent_review"


def test_manifest_pins_official_source_archive() -> None:
    manifest = load_resource_document(f"{_ROOT}/manifest.yaml")
    assert manifest["source_sha256"] == (
        "9ba942558275626524015d8c444d57ee36a46092c3e674a85480c9872b4fe5dc"
    )


def test_urban_resource_contains_reference_values():
    resource = load_urban_parameters()
    assert resource["review_status"] == "pending_independent_review"
    assert resource["grid"]["intersection_spacing_m"] == [433.0, 250.0]
    assert resource["path_loss"]["los_nlosv"]["constant_db"] == 38.77
    assert resource["fast_fading"]["nlos"]["clusters"] == 19


@pytest.mark.parametrize(
    ("condition", "count", "last_delay_s"),
    [("los", 17, 326.7825e-9), ("nlos", 23, 380.2389e-9), ("nlosv", 24, 471.3768e-9)],
)
def test_exact_urban_cdl_profile_lengths(condition, count, last_delay_s):
    profile = load_cdl_profile(condition)
    assert len(profile.delays_s) == count
    assert profile.delays_s[-1] == pytest.approx(last_delay_s)
    assert len(profile.powers_db) == count
