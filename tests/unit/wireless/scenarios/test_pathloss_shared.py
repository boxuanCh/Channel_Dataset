import numpy as np
import pytest

from channel_dataset.channels.wireless import ScenarioConfig, create_scenario
from channel_dataset.channels.wireless.math import SPEED_OF_LIGHT_MPS


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_umi_breakpoint_is_continuous(backend, realization_metadata, topology_factory) -> None:
    scenario = create_scenario(ScenarioConfig("UMi", 3.5e9))
    breakpoint = 4.0 * 9.0 * 0.5 * 3.5e9 / SPEED_OF_LIGHT_MPS
    below = topology_factory(
        backend, realization_metadata, tx_height=10.0, rx_height=1.5,
        distance=breakpoint * (1.0 - 1.0e-7),
    )
    above = topology_factory(
        backend, realization_metadata, tx_height=10.0, rx_height=1.5,
        distance=breakpoint * (1.0 + 1.0e-7),
    )
    before = _numpy(scenario.path_loss_components_db(below)[0])
    after = _numpy(scenario.path_loss_components_db(above)[0])
    assert np.allclose(before, after, atol=1.0e-4)


def test_umi_o2i_component_is_explicit(backend, realization_metadata, topology_factory) -> None:
    scenario = create_scenario(ScenarioConfig("UMi", 3.5e9))
    outdoor = topology_factory(
        backend, realization_metadata, tx_height=10.0, rx_height=1.5,
        distance=100.0, indoor=False,
    )
    indoor = topology_factory(
        backend, realization_metadata, tx_height=10.0, rx_height=1.5,
        distance=100.0, indoor=True,
    )
    assert np.allclose(_numpy(scenario.path_loss_components_db(outdoor)[2]), 0.0)
    assert np.allclose(_numpy(scenario.path_loss_components_db(indoor)[2]), 20.0)


@pytest.mark.torch
def test_deterministic_pathloss_backend_parity(topology_factory) -> None:
    from channel_dataset.backends import get_backend
    from channel_dataset.core import RealizationMetadata

    outputs = []
    for name in ("numpy", "torch"):
        backend = get_backend(name)
        metadata = RealizationMetadata(
            "wireless", "parity", 1, name, "double", "cpu",
            f"{name}:1:root", "0" * 64, {}, "test",
        )
        topology = topology_factory(
            backend, metadata, tx_height=25.0, rx_height=1.5, distance=100.0
        )
        scenario = create_scenario(ScenarioConfig("UMa", 3.5e9))
        outputs.append(_numpy(scenario.path_loss_components_db(topology)[0]))
    assert np.allclose(outputs[0], outputs[1], atol=1.0e-12)
