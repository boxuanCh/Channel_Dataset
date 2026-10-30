from __future__ import annotations

import json

import pytest
import yaml

from channel_dataset.config import ConfigEnvelope, parse_config_document
from channel_dataset.core import RealizationMetadata


def _direct_document() -> dict:
    return {
        "schema_version": 1,
        "model": {
            "family": "wireless",
            "name": "rayleigh_block_fading",
            "parameters": {
                "num_rx": 1,
                "num_rx_antennas": 1,
                "num_tx": 1,
                "num_tx_antennas": 1,
                "average_power_linear": 1.0,
            },
        },
    }


def test_yaml_and_json_documents_have_the_same_digest() -> None:
    document = _direct_document()
    yaml_config = parse_config_document(yaml.safe_load(yaml.safe_dump(document)))
    json_config = parse_config_document(json.loads(json.dumps(document, sort_keys=True)))
    assert yaml_config.model_digest == json_config.model_digest


def test_key_order_does_not_change_digest() -> None:
    first = parse_config_document(_direct_document())
    reordered = _direct_document()
    reordered["model"]["parameters"] = dict(
        reversed(tuple(reordered["model"]["parameters"].items()))
    )
    assert first.model_digest == parse_config_document(reordered).model_digest


def test_profile_and_direct_parameters_have_the_same_digest() -> None:
    profile = parse_config_document(
        {
            "schema_version": 1,
            "model": {
                "family": "wireless",
                "name": "rayleigh_block_fading",
                "profile": "package://profiles/wireless/rayleigh_siso.yaml",
            },
            "execution": {
                "backend": "numpy",
                "precision": "single",
                "device": "cpu",
                "seed": 999,
            },
            "metadata": {"label": "ignored by physical digest"},
        }
    )
    assert profile.model_digest == parse_config_document(_direct_document()).model_digest


def test_physical_parameter_change_changes_digest() -> None:
    changed = _direct_document()
    changed["model"]["parameters"]["average_power_linear"] = 2.0
    assert (
        parse_config_document(changed).model_digest
        != parse_config_document(_direct_document()).model_digest
    )


def test_resolved_document_is_serializable_and_round_trips() -> None:
    envelope = parse_config_document(_direct_document())
    serialized = json.dumps(envelope.to_document())
    replayed = parse_config_document(json.loads(serialized))
    assert replayed.model_digest == envelope.model_digest


def test_parameters_are_deeply_immutable() -> None:
    document = _direct_document()
    document["model"]["parameters"]["nested"] = {"values": [1, 2]}
    envelope = parse_config_document(document)
    with pytest.raises(TypeError):
        envelope.parameters["nested"]["new"] = 3
    assert envelope.parameters["nested"]["values"] == (1, 2)


def test_realization_metadata_reconstructs_configuration() -> None:
    envelope = parse_config_document(_direct_document())
    metadata = RealizationMetadata(
        model_family=envelope.family,
        model_name=envelope.name,
        schema_version=envelope.schema_version,
        backend="numpy",
        precision="double",
        device="cpu",
        stream_id="numpy:7:root",
        config_digest=envelope.model_digest,
        resolved_configuration=envelope.parameters,
        software_version="test",
        request_id="phase-3",
    )

    reconstructed = ConfigEnvelope.from_realization_metadata(metadata)
    assert reconstructed.model_digest == envelope.model_digest
    json.dumps(metadata.to_document())


def test_non_finite_configuration_is_rejected() -> None:
    document = _direct_document()
    document["model"]["parameters"]["average_power_linear"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        parse_config_document(document)
