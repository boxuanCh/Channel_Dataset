import pytest

from channel_dataset.channels.wireless.tr38901 import parse_cdl_config, register_cdl_model
from channel_dataset.config import load_config
from channel_dataset.registry import ModelRegistry


def test_example_cdl_configuration_loads() -> None:
    envelope = load_config("configs/wireless/example_cdl.yaml")
    registry = ModelRegistry()
    register_cdl_model(registry)
    model = registry.create(envelope)
    assert model.config.profile == "CDL-C"
    assert model.transmitter_layout.array.num_elements == 1


def test_cdl_config_rejects_unknown_fields(cdl_config_factory) -> None:
    values = {
        name: getattr(cdl_config_factory(), name)
        for name in cdl_config_factory().__dataclass_fields__
    }
    values["unknown"] = 1
    with pytest.raises(ValueError, match="unknown CDL"):
        parse_cdl_config(values)


def test_k_factor_is_restricted_to_los_profiles(cdl_config_factory) -> None:
    with pytest.raises(ValueError, match="CDL-D/E"):
        cdl_config_factory("CDL-A", k_factor_db=10.0)
