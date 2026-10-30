import numpy as np

from channel_dataset.channels.wireless import create_scenario


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_large_scale_invalid_link_mask_zeroes_physics(
    backend,
    realization_metadata,
    scenario_config_factory,
    valid_scenario_topology,
) -> None:
    scenario = create_scenario(scenario_config_factory("UMi"))
    topology = valid_scenario_topology("UMi", backend, realization_metadata)
    topology = type(topology)(
        topology.tx_positions_m,
        topology.rx_positions_m,
        topology.tx_velocities_mps,
        topology.rx_velocities_mps,
        topology.tx_orientations_rad,
        topology.rx_orientations_rad,
        topology.rx_indoor,
        topology.tx_site_ids,
        topology.metadata,
        topology.los_mask,
        backend.zeros((1, 1, 1), dtype="bool"),
    )
    result = scenario.sample_large_scale(topology, backend.random_stream(7))
    assert np.allclose(_numpy(result.delay_spread_s), 0.0)
    assert np.allclose(_numpy(result.path_loss_db), 0.0)
