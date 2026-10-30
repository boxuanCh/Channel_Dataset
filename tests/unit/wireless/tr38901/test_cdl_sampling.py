import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import ClusteredDelayLine


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_cdl_seeded_replay(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory("CDL-C"))
    def realize():
        request = ChannelSamplingRequest(2, 3, 10_000.0, execution_context, execution_context.random_stream(91))
        return _numpy(model.realize(request).coefficients)
    assert np.array_equal(realize(), realize())


def test_cdl_topology_matches_request(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory("CDL-D", link_distance_m=42.0))
    request = ChannelSamplingRequest(3, 1, 1.0, execution_context, execution_context.random_stream(2))
    topology = model.topology(request)
    distances = np.linalg.norm(_numpy(topology.rx_positions_m - topology.tx_positions_m), axis=-1)
    assert topology.shape == (3, 1, 1)
    assert np.allclose(distances, 42.0)
    assert np.all(_numpy(topology.los_mask))
