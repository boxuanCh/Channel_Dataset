import pytest

from channel_dataset.config import load_config, parse_config_document


def test_load_yaml_config() -> None:
    config = load_config("configs/wireless/example_tdl.yaml")

    assert config.family == "wireless"
    assert config.name == "tdl"
    assert config.parameters["profile"] == "TDL-C"
    assert config.parameters["delay_spread_s"] == 300e-9


def test_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        parse_config_document(
            {
                "schema_version": 99,
                "model": {"family": "x", "name": "y", "parameters": {}},
            }
        )
