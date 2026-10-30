import numpy as np

from channel_dataset.channels.wireless import compute_link_geometry, create_scenario


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_umi_los_probability(backend, realization_metadata, scenario_config_factory, valid_scenario_topology) -> None:
    scenario = create_scenario(scenario_config_factory("UMi"))
    topology = valid_scenario_topology("UMi", backend, realization_metadata)
    probability = _numpy(scenario.los_probability(compute_link_geometry(topology)))[0, 0, 0]
    assert 0.0 <= probability <= 1.0
    np.testing.assert_allclose(probability, np.minimum(18.0 / 100.0, 1.0) * (1.0 - np.exp(-100.0 / 36.0)) + np.exp(-100.0 / 36.0), rtol=1e-12)
