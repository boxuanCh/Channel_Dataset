import numpy as np

from channel_dataset.channels.wireless import create_scenario


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_umi_pathloss_los_nlos_ordering(backend, realization_metadata, scenario_config_factory, valid_scenario_topology) -> None:
    scenario = create_scenario(scenario_config_factory("UMi"))
    topology = valid_scenario_topology("UMi", backend, realization_metadata)
    los, nlos, o2i = scenario.path_loss_components_db(topology)
    los, nlos, o2i = _numpy(los), _numpy(nlos), _numpy(o2i)
    assert np.all(nlos >= los)
    assert np.all(np.isfinite(los))
    assert np.all(o2i >= 0.0)


def test_umi_pathloss_selects_los_state(backend, realization_metadata, scenario_config_factory, valid_scenario_topology) -> None:
    scenario = create_scenario(scenario_config_factory("UMi"))
    topology = valid_scenario_topology("UMi", backend, realization_metadata)
    los_mask = backend.full((1, 1, 1), True, dtype="bool")
    nlos_mask = backend.full((1, 1, 1), False, dtype="bool")
    los = _numpy(scenario.path_loss_db(topology, los_mask))
    nlos = _numpy(scenario.path_loss_db(topology, nlos_mask))
    assert np.all(nlos >= los)
