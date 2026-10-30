from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from channel_dataset.channels.wireless.ieee80211 import register_ieee80211_models
from channel_dataset.config import load_config, load_resource_document
from channel_dataset.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.ieee80211
@pytest.mark.parametrize(
    "schema",
    [
        "ieee80211-profile-v1.schema.json",
        "ieee80211-tgn-v1.schema.json",
        "ieee80211-tgac-v1.schema.json",
        "ieee80211-tgah-v1.schema.json",
        "ieee80211-tgax-v1.schema.json",
    ],
)
def test_public_and_packaged_schemas_are_semantically_identical(schema: str) -> None:
    public = json.loads((ROOT / "configs" / "schema" / schema).read_text(encoding="utf-8"))
    packaged = load_resource_document(f"schema/{schema}")
    assert public == packaged


@pytest.mark.ieee80211
@pytest.mark.parametrize(
    "filename,schema",
    [
        ("tgn_model_b.yaml", "ieee80211-tgn-v1.schema.json"),
        ("tgac_multiuser.yaml", "ieee80211-tgac-v1.schema.json"),
        ("tgah_indoor.yaml", "ieee80211-tgah-v1.schema.json"),
        ("tgah_station_to_station.yaml", "ieee80211-tgah-v1.schema.json"),
        ("tgax_indoor.yaml", "ieee80211-tgax-v1.schema.json"),
        ("tgax_outdoor_umi.yaml", "ieee80211-tgax-v1.schema.json"),
    ],
)
def test_strict_examples_validate_and_construct(filename: str, schema: str) -> None:
    envelope = load_config(ROOT / "configs" / "wireless" / "ieee80211" / filename)
    jsonschema.validate(
        dict(envelope.parameters), load_resource_document(f"schema/{schema}")
    )
    registry = ModelRegistry()
    register_ieee80211_models(registry)
    model = registry.create(envelope)
    assert model.name == envelope.name


@pytest.mark.ieee80211
@pytest.mark.parametrize("profile", "abcdef")
def test_tgn_profile_resources_validate_against_public_schema(profile: str) -> None:
    document = load_resource_document(
        f"standards/wireless/ieee80211/03_0940r4/model_{profile}.yaml"
    )
    schema = load_resource_document("schema/ieee80211-profile-v1.schema.json")
    jsonschema.validate(document, schema)
