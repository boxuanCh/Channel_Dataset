from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema.validators import validator_for

from channel_dataset.config import (
    load_resource_document,
    parse_config_document,
    resource_checksum,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "config"


def test_package_resource_lookup_does_not_depend_on_working_directory(monkeypatch) -> None:
    monkeypatch.chdir(ROOT / "tests")
    profile = load_resource_document("package://profiles/wireless/rayleigh_siso.yaml")
    assert profile["family"] == "wireless"
    assert profile["parameters"]["average_power_linear"] == 1.0


def test_bare_resource_reference_is_package_relative() -> None:
    explicit = load_resource_document("package://profiles/wireless/rayleigh_siso.yaml")
    bare = load_resource_document("profiles/wireless/rayleigh_siso.yaml")
    assert bare == explicit


def test_relative_file_resource_uses_explicit_base_directory() -> None:
    profile = load_resource_document("file:profile_base.yaml", base_directory=FIXTURES)
    assert profile["name"] == "nested_model"


def test_package_traversal_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot escape"):
        load_resource_document("package://../pyproject.toml")


def test_resource_checksum_is_stable() -> None:
    reference = "package://profiles/wireless/rayleigh_siso.yaml"
    checksum = resource_checksum(reference)
    assert len(checksum) == 64
    assert checksum == resource_checksum(reference)


@pytest.mark.parametrize(
    "schema_name",
    [
        "antenna-layout-v1.schema.json",
        "awgn-v1.schema.json",
        "cdl-v1.schema.json",
        "flat-mimo-fading-v1.schema.json",
        "model-config-v1.schema.json",
        "model-profile-v1.schema.json",
        "rayleigh-block-fading-v1.schema.json",
        "tdl-v1.schema.json",
    ],
)
def test_bundled_json_schemas_are_valid(schema_name: str) -> None:
    schema = load_resource_document(f"package://schema/{schema_name}")
    validator_for(schema).check_schema(schema)


def test_common_and_rayleigh_schemas_validate_resolved_configuration() -> None:
    envelope = parse_config_document(
        {
            "schema_version": 1,
            "model": {
                "family": "wireless",
                "name": "rayleigh_block_fading",
                "profile": "package://profiles/wireless/rayleigh_siso.yaml",
            },
        }
    )
    common_schema = load_resource_document("package://schema/model-config-v1.schema.json")
    model_schema = load_resource_document(
        "package://schema/rayleigh-block-fading-v1.schema.json"
    )
    validator_for(common_schema)(common_schema).validate(envelope.to_document())
    validator_for(model_schema)(model_schema).validate(dict(envelope.parameters))


def test_repository_and_packaged_schemas_match() -> None:
    for name in (
        "antenna-layout-v1.schema.json",
        "awgn-v1.schema.json",
        "cdl-v1.schema.json",
        "flat-mimo-fading-v1.schema.json",
        "model-config-v1.schema.json",
        "model-profile-v1.schema.json",
        "rayleigh-block-fading-v1.schema.json",
        "tdl-v1.schema.json",
    ):
        repository_schema = json.loads((ROOT / "configs" / "schema" / name).read_text("utf-8"))
        packaged_schema = load_resource_document(f"package://schema/{name}")
        assert repository_schema == packaged_schema


def test_yaml_profile_satisfies_profile_schema() -> None:
    schema = load_resource_document("package://schema/model-profile-v1.schema.json")
    source = ROOT / "src" / "channel_dataset" / "resources" / "profiles" / "wireless"
    document = yaml.safe_load((source / "rayleigh_2x2.yaml").read_text("utf-8"))
    validator_for(schema)(schema).validate(document)


@pytest.mark.parametrize(
    ("profile", "schema_name"),
    [
        ("profiles/wireless/awgn_unit_variance.yaml", "awgn-v1.schema.json"),
        (
            "profiles/wireless/flat_mimo_uncorrelated.yaml",
            "flat-mimo-fading-v1.schema.json",
        ),
        (
            "profiles/wireless/flat_mimo_kronecker.yaml",
            "flat-mimo-fading-v1.schema.json",
        ),
        (
            "profiles/wireless/flat_mimo_per_column.yaml",
            "flat-mimo-fading-v1.schema.json",
        ),
    ],
)
def test_phase5_profiles_resolve_against_model_schemas(
    profile: str, schema_name: str
) -> None:
    profile_document = load_resource_document(profile)
    envelope = parse_config_document(
        {
            "schema_version": 1,
            "model": {
                "family": profile_document["family"],
                "name": profile_document["name"],
                "profile": profile,
            },
        }
    )
    schema = load_resource_document(f"schema/{schema_name}")
    resolved_parameters = envelope.to_document()["model"]["parameters"]
    validator_for(schema)(schema).validate(resolved_parameters)
