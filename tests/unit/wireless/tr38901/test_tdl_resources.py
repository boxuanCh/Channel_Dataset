from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.validators import validator_for

from channel_dataset.channels.wireless.tr38901 import load_tdl_profile
from channel_dataset.config import load_resource_document, resource_checksum

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("name", "count", "table"),
    [
        ("TDL-A", 23, "7.7.2-1"),
        ("TDL-B", 23, "7.7.2-2"),
        ("TDL-C", 24, "7.7.2-3"),
        ("TDL-D", 14, "7.7.2-4"),
        ("TDL-E", 15, "7.7.2-5"),
        ("TDLA30", 12, "B.2.1.1-2 / B.2.1.2-2"),
        ("TDLB100", 12, "B.2.1.1-3"),
        ("TDLC300", 12, "B.2.1.1-4"),
        ("TDLC60", 12, "B.2.1.2-3"),
    ],
)
def test_standard_resource_identity_and_row_count(name, count, table) -> None:
    profile = load_tdl_profile(name)
    assert profile.num_paths == count
    assert profile.table == table
    assert profile.standard_version == "16.1.0"
    assert profile.review_status == "pending_independent_review"


def test_los_resources_have_one_specular_component() -> None:
    for name in ("TDL-D", "TDL-E"):
        profile = load_tdl_profile(name)
        assert profile.component_types.count("los") == 1
        assert profile.delays[profile.component_types.index("los")] == 0.0


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("standards/wireless/tr_38_901/v16_1/tdl_a.yaml", "4d417e4ed6765b4991cc3aedb6ded4d973e9945af1738a60324ac2422ce45a2e"),
        ("standards/wireless/tr_38_901/v16_1/tdl_b.yaml", "46323f316003ef909a2c0ee4f77fdf6ca8af22762b5eb0c1b47ecf9483a2fa70"),
        ("standards/wireless/tr_38_901/v16_1/tdl_c.yaml", "b23a0a664da7c3128fd15c5c47d79c5e0038c967e2235859769831f4787e73cd"),
        ("standards/wireless/tr_38_901/v16_1/tdl_d.yaml", "f2328bfaee07aa7701e9fd44071f894a2c2ae185a9b6b60cc2aaeef57beaeea3"),
        ("standards/wireless/tr_38_901/v16_1/tdl_e.yaml", "46b3698289a717285cadf62514405793d8b1021bba770362af32f15e525ed414"),
        ("standards/wireless/ts_38_101_4/v16_1/tdla30.yaml", "a950d7183421c49f2fea4df2917c1a7aee95b4d3cdc78c78c9a0e2735f14bd7b"),
        ("standards/wireless/ts_38_101_4/v16_1/tdlb100.yaml", "e8206860e99a16231e069b6c726923237f415872971019d7463c8067cda5baeb"),
        ("standards/wireless/ts_38_101_4/v16_1/tdlc300.yaml", "6f9288f8fb895d305e1990a6549090b814b3badca95099f9f6d502d4b7255488"),
        ("standards/wireless/ts_38_101_4/v16_1/tdlc60.yaml", "d5cc2cfb42314f855a97e9bfcfadfbdbb996ac5188675920fd745ea0ea931d08"),
    ],
)
def test_standard_resource_checksums_are_stable(reference, expected) -> None:
    assert resource_checksum(reference) == expected


def test_tdl_schema_is_valid_and_packaged_copy_matches() -> None:
    packaged = load_resource_document("schema/tdl-v1.schema.json")
    validator_for(packaged).check_schema(packaged)
    repository = json.loads((ROOT / "configs" / "schema" / "tdl-v1.schema.json").read_text())
    assert packaged == repository
