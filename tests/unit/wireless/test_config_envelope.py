from __future__ import annotations

import pytest

from channel_dataset.config import parse_config_document


def _document() -> dict:
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
        "execution": {"backend": "numpy", "precision": "double", "device": "cpu", "seed": 7},
    }


def test_execution_config_and_hash_are_loaded() -> None:
    config = parse_config_document(_document())
    assert config.execution is not None
    assert config.execution.seed == 7
    assert len(config.model_digest) == 64


def test_execution_does_not_change_model_hash() -> None:
    first = _document()
    second = _document()
    second["execution"]["seed"] = 999
    assert parse_config_document(first).model_digest == parse_config_document(second).model_digest


def test_unknown_execution_field_is_rejected() -> None:
    document = _document()
    document["execution"]["extra"] = True
    with pytest.raises(ValueError, match="unknown execution"):
        parse_config_document(document)


def test_unknown_model_field_is_rejected() -> None:
    document = _document()
    document["model"]["alias"] = "not-supported"
    with pytest.raises(ValueError, match="unknown model"):
        parse_config_document(document)


def test_boolean_schema_version_is_rejected() -> None:
    document = _document()
    document["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        parse_config_document(document)
