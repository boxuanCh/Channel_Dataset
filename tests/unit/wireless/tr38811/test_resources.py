from __future__ import annotations

from channel_dataset.channels.wireless.tr38811 import load_resource, load_tdl_profile


def test_manifest_pins_official_release_15_archive() -> None:
    manifest = load_resource("manifest")
    assert manifest["standard_version"] == "15.4.0"
    assert manifest["source_archive"] == "38811-f40.zip"
    assert manifest["source_sha256"] == (
        "fff99141ab5f93fcfbca8321de8c4eb48b98347d01c5be53baf5c782308f37b3"
    )
    assert manifest["review_status"] == "pending_independent_review"


def test_los_probability_table_values_are_exact() -> None:
    document = load_resource("los_probability")
    assert document["elevation_degrees"] == [10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert document["dense_urban"] == [
        0.282, 0.331, 0.398, 0.468, 0.537, 0.612, 0.738, 0.820, 0.981,
    ]


def test_tdl_profiles_match_clause_6_9_2() -> None:
    profile = load_tdl_profile("NTN-TDL-D")
    assert profile.table == "6.9.2-4"
    assert profile.delays == (0.0, 0.0, 0.5596, 7.334)
    assert profile.powers_db == (-0.284, -11.991, -9.887, -16.771)
    assert profile.default_k_factor_db == 11.707
