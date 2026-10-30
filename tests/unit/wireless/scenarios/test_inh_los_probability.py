import numpy as np

from channel_dataset.channels.wireless import compute_link_geometry, create_scenario


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_inh_los_probability(backend, realization_metadata, scenario_config_factory, valid_scenario_topology) -> None:
    scenario = create_scenario(scenario_config_factory("InH"))
    topology = valid_scenario_topology("InH", backend, realization_metadata)
    probability = _numpy(scenario.los_probability(compute_link_geometry(topology)))[0, 0, 0]
    assert 0.0 <= probability <= 1.0
    np.testing.assert_allclose(probability, 0.32 * np.exp(-(20.0 - 6.5) / 32.6), rtol=1e-12)
