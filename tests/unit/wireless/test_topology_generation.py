import numpy as np

from channel_dataset.channels.wireless import (
    ChannelSamplingRequest,
    RectangularTopologyConfig,
    RectangularTopologyGenerator,
    reverse_ray_batch,
)
from channel_dataset.channels.wireless.tr38901 import ClusteredDelayLine


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_topology_generation_is_reproducible(execution_context) -> None:
    generator = RectangularTopologyGenerator(
        RectangularTopologyConfig(2, 3, 100.0, 50.0, 10.0, 1.5, 5.0, 0.25)
    )
    def generate(seed):
        request = ChannelSamplingRequest(4, 1, 1.0, execution_context, execution_context.random_stream(seed))
        return generator.generate(request)
    first, second, other = generate(11), generate(11), generate(12)
    assert np.array_equal(_numpy(first.tx_positions_m), _numpy(second.tx_positions_m))
    assert np.array_equal(_numpy(first.rx_positions_m), _numpy(second.rx_positions_m))
    assert not np.array_equal(_numpy(first.rx_positions_m), _numpy(other.rx_positions_m))
    speed = np.linalg.norm(_numpy(first.rx_velocities_mps), axis=-1)
    assert np.allclose(speed, 5.0)


def test_ray_reciprocal_view_does_not_mutate_source(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory())
    request = ChannelSamplingRequest(1, 1, 1.0, execution_context, execution_context.random_stream(8))
    rays = model.sample_rays(request)
    original = _numpy(rays.azimuth_arrival_rad).copy()
    reverse = reverse_ray_batch(rays)
    assert reverse.shape == (1, 1, 1, 23, 20)
    assert np.array_equal(_numpy(reverse.azimuth_departure_rad), original)
    assert np.array_equal(_numpy(rays.azimuth_arrival_rad), original)
