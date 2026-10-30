import numpy as np

from channel_dataset.channels.wireless import compute_link_geometry, create_scenario


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_inf_los_probability(backend, realization_metadata, scenario_config_factory, valid_scenario_topology) -> None:
    scenario = create_scenario(scenario_config_factory("InF"))
    topology = valid_scenario_topology("InF", backend, realization_metadata)
    probability = _numpy(scenario.los_probability(compute_link_geometry(topology)))[0, 0, 0]
    assert 0.0 <= probability <= 1.0
