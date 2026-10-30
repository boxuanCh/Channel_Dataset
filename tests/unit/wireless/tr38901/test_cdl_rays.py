import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import ClusteredDelayLine


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_cdl_ray_shape_power_and_offsets(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory("CDL-A"))
    request = ChannelSamplingRequest(2, 1, 1.0, execution_context, execution_context.random_stream(8))
    rays = model.sample_rays(request)
    assert rays.shape == (2, 1, 1, 23, 20)
    assert np.allclose(_numpy(rays.powers_linear).sum(axis=-1), 1.0)
    offsets = np.sort(
        np.rad2deg(_numpy(rays.azimuth_arrival_rad)[0, 0, 0, 0])
        - model.profile.aoa_degrees[0]
    )
    expected = np.sort(np.asarray([
        0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492,
        0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797,
        0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195,
        2.1551, -2.1551,
    ]) * 11.0)
    assert np.allclose(offsets, expected, atol=1.0e-10)


def test_random_angle_coupling_is_seeded(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory())
    def sample(seed):
        request = ChannelSamplingRequest(1, 1, 1.0, execution_context, execution_context.random_stream(seed))
        return _numpy(model.sample_rays(request).azimuth_departure_rad)
    assert np.array_equal(sample(17), sample(17))
    assert not np.array_equal(sample(17), sample(18))
