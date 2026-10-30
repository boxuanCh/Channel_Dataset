import pytest

from channel_dataset.channels.wireless.tr38901 import load_cdl_profile, load_cdl_ray_offsets
from channel_dataset.config import resource_checksum

RESOURCE_CHECKSUMS = {
    "cdl_a.yaml": "023f7b6d54aafaa22456a96c5e62685dbe193fa5f129316585f9a7c3d2dbc6e6",
    "cdl_b.yaml": "fb97fda5e1d1e5a0a4ea310d648cee2d8a8e54459dfcd64f62d0e338f132809d",
    "cdl_c.yaml": "3071fa6867a699d02ad9eab10ea3285797b5ee2d126d324ae3711a866c6c8e1c",
    "cdl_d.yaml": "86205a498d5e9bf41ec05472ae0988ac14c81a61f7eaa40adfbe89ab3b62919f",
    "cdl_e.yaml": "f297a8d18f58023595846d64060f03d6f7ae78ebd2bc2f8edf21f6856584e34c",
}


@pytest.mark.parametrize(
    ("name", "clusters", "table", "los"),
    [
        ("CDL-A", 23, "7.7.1-1", False),
        ("CDL-B", 23, "7.7.1-2", False),
        ("CDL-C", 24, "7.7.1-3", False),
        ("CDL-D", 13, "7.7.1-4", True),
        ("CDL-E", 14, "7.7.1-5", True),
    ],
)
def test_cdl_resource_identity_and_counts(name, clusters, table, los) -> None:
    profile = load_cdl_profile(name)
    assert profile.num_clusters == clusters
    assert profile.table == table
    assert profile.has_los is los
    assert profile.metadata["review_status"] == "pending_independent_review"


def test_known_cdl_table_values() -> None:
    profile = load_cdl_profile("CDL-C")
    assert profile.delays[23] == 8.6523
    assert profile.powers_db[0] == -4.4
    assert profile.aod_degrees[5] == 0.3
    assert profile.aoa_degrees[23] == 33.6
    assert profile.xpr_db == 7.0
    assert load_cdl_ray_offsets()[0:4] == (0.0447, -0.0447, 0.1413, -0.1413)


@pytest.mark.parametrize("filename", RESOURCE_CHECKSUMS)
def test_cdl_resource_checksum(filename) -> None:
    reference = f"standards/wireless/tr_38_901/v16_1/{filename}"
    assert resource_checksum(reference) == RESOURCE_CHECKSUMS[filename]
