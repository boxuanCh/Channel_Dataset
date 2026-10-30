from __future__ import annotations

from pathlib import Path

import pytest

from channel_dataset.channels.wireless.fading import register_rayleigh_model
from channel_dataset.config import load_config, parse_config_document, resolve_model_profile
from channel_dataset.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures" / "config"


def test_packaged_profile_inheritance_and_envelope_override() -> None:
    envelope = load_config(ROOT / "configs" / "wireless" / "rayleigh_block_fading_profile.yaml")

    assert envelope.profile == "package://profiles/wireless/rayleigh_2x2.yaml"
    assert dict(envelope.parameters) == {
        "num_rx": 1,
        "num_rx_antennas": 2,
        "num_tx": 1,
        "num_tx_antennas": 2,
        "average_power_linear": 0.5,
    }

    registry = ModelRegistry()
    register_rayleigh_model(registry)
    assert registry.create(envelope).config.average_power_linear == 0.5


def test_file_profile_recursively_merges_mappings_and_replaces_sequences() -> None:
    resolved = resolve_model_profile(
        "file:profile_child.json",
        family="fixture",
        name="nested_model",
        overrides={"nested": {"right": "envelope", "added": True}},
        base_directory=FIXTURES,
    )

    assert resolved == {
        "scalar": 1,
        "nested": {"left": "inherited", "right": "envelope", "added": True},
        "sequence": [3],
    }


def test_profile_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="inheritance cycle"):
        resolve_model_profile(
            "file:profile_cycle_a.yaml",
            family="fixture",
            name="cyclic_model",
            base_directory=FIXTURES,
        )


def test_profile_identity_must_match_envelope() -> None:
    with pytest.raises(ValueError, match="does not match envelope"):
        resolve_model_profile(
            "package://profiles/wireless/rayleigh_siso.yaml",
            family="wireless",
            name="different_model",
        )


def test_profile_only_envelope_is_valid() -> None:
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
    assert envelope.parameters["average_power_linear"] == 1.0


def test_ambiguous_rayleigh_power_unit_alias_is_rejected() -> None:
    document = {
        "schema_version": 1,
        "model": {
            "family": "wireless",
            "name": "rayleigh_block_fading",
            "profile": "package://profiles/wireless/rayleigh_siso.yaml",
            "parameters": {"average_power_db": 0.0},
        },
    }
    registry = ModelRegistry()
    register_rayleigh_model(registry)
    with pytest.raises(ValueError, match="unknown Rayleigh parameters"):
        registry.create(parse_config_document(document))
