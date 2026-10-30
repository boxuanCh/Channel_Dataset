import numpy as np

from channel_dataset.channels.wireless import create_scenario


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_uma_large_scale_replay_and_switch_invariance(backend, realization_metadata, scenario_config_factory, valid_scenario_topology) -> None:
    scenario = create_scenario(scenario_config_factory("UMa"))
    topology = valid_scenario_topology("UMa", backend, realization_metadata)
    enabled = scenario.sample_large_scale(topology, backend.random_stream(44))
    disabled = scenario.sample_large_scale(
        topology,
        backend.random_stream(44),
        path_loss_enabled=False,
        shadow_fading_enabled=False,
    )
    replay = scenario.sample_large_scale(topology, backend.random_stream(44))
    assert enabled.shape == (1, 1, 1)
    assert np.array_equal(_numpy(enabled.delay_spread_s), _numpy(replay.delay_spread_s))
    assert np.array_equal(_numpy(enabled.k_factor_db), _numpy(disabled.k_factor_db))
    assert np.allclose(_numpy(disabled.path_loss_db), 0.0)
    assert np.allclose(_numpy(disabled.shadow_fading_db), 0.0)
