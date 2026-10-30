from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from channel_dataset.channels.wireless import (
    GeometricCDLConfig,
    load_geometric_resource,
    parse_geometric_cdl_config,
)
from channel_dataset.config import load_config


def test_strict_config_and_mode_defaults():
    scenario = parse_geometric_cdl_config({
        "standard_version": "16.1.0", "carrier_frequencies_hz": [3.5e9]
    })
    assert scenario.apply_path_loss is True
    fixed = GeometricCDLConfig(
        "19.2.0", (3.5e9,), initialization_mode="fixed_profile", fixed_profile="CDL-A"
    )
    assert fixed.apply_path_loss is False
    with pytest.raises(ValueError, match="unknown"):
        parse_geometric_cdl_config({
            "standard_version": "16.1.0", "carrier_frequencies_hz": [3.5e9], "extra": 1
        })


@pytest.mark.parametrize("version", ["16.1.0", "19.2.0"])
def test_versioned_resources(version):
    evolution = load_geometric_resource(version, "geometric_evolution.yaml")
    assert evolution["standard_version"] == version
    assert load_geometric_resource(version, "cdl_a.yaml")["name"] == "CDL-A"


def test_example_matches_schema():
    root = Path(__file__).resolve().parents[4]
    envelope = load_config(root / "configs/wireless/geometric_cdl_multiband.yaml")
    schema = json.loads(
        (root / "configs/schema/geometric-cdl-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    parameters = json.loads(json.dumps(dict(envelope.parameters)))
    jsonschema.validate(parameters, schema)
