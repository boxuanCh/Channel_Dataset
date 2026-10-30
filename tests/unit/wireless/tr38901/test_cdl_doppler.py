import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import ClusteredDelayLine


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_static_cdl_is_constant_over_time(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory())
    request = ChannelSamplingRequest(2, 4, 10_000.0, execution_context, execution_context.random_stream(10))
    coefficients = _numpy(model.realize(request).coefficients)
    assert np.allclose(coefficients, coefficients[..., :1], atol=1.0e-12)


def test_motion_evolves_cdl_coefficients(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory(terminal_velocity_mps=(30.0, 0.0, 0.0)))
    request = ChannelSamplingRequest(2, 4, 10_000.0, execution_context, execution_context.random_stream(10))
    coefficients = _numpy(model.realize(request).coefficients)
    assert not np.allclose(coefficients[..., 0], coefficients[..., -1])
