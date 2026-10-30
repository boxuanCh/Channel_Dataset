from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from channel_dataset.channels.wireless import register_tr37885_models
from channel_dataset.config import load_config, load_resource_document
from channel_dataset.registry import ModelRegistry

pytestmark = [pytest.mark.v2x, pytest.mark.tr37885]

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "schema_name",
    ["tr37885-urban-system-v1.schema.json", "tr37885-urban-cdl-v1.schema.json"],
)
def test_public_and_packaged_schemas_match(schema_name: str) -> None:
    public = json.loads((ROOT / "configs/schema" / schema_name).read_text(encoding="utf-8"))
    packaged = load_resource_document(f"schema/{schema_name}")
    assert public == packaged
    jsonschema.validators.validator_for(public).check_schema(public)


@pytest.mark.parametrize(
    "filename,schema_name",
    [
        ("urban_option_a.yaml", "tr37885-urban-system-v1.schema.json"),
        ("urban_option_b.yaml", "tr37885-urban-cdl-v1.schema.json"),
        ("urban_controlled_variation.yaml", "tr37885-urban-system-v1.schema.json"),
    ],
)
def test_examples_validate_and_construct(filename: str, schema_name: str) -> None:
    envelope = load_config(ROOT / "configs/wireless/tr37885" / filename)
    jsonschema.validate(dict(envelope.parameters), load_resource_document(f"schema/{schema_name}"))
    registry = ModelRegistry()
    register_tr37885_models(registry)
    assert registry.create(envelope).name == envelope.name
