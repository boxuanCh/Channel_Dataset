import numpy as np

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import ClusteredDelayLine


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_cdl_coefficients_match_array_dimensions(execution_context, cdl_config_factory) -> None:
    config = cdl_config_factory(
        base_station_array_profile="package://profiles/wireless/antenna_3gpp_8x8_dual.yaml"
    )
    model = ClusteredDelayLine(config)
    request = ChannelSamplingRequest(1, 2, 1000.0, execution_context, execution_context.random_stream(3))
    cir = model.realize(request)
    assert cir.shape == (1, 1, 1, 1, 128, 23, 2)
    assert np.all(np.isfinite(_numpy(cir.coefficients)))


def test_uplink_reverses_endpoint_array_assignment(execution_context, cdl_config_factory) -> None:
    config = cdl_config_factory(
        link_direction="uplink",
        base_station_array_profile="package://profiles/wireless/antenna_3gpp_8x8_dual.yaml",
    )
    model = ClusteredDelayLine(config)
    request = ChannelSamplingRequest(1, 1, 1.0, execution_context, execution_context.random_stream(4))
    assert model.realize(request).shape == (1, 1, 128, 1, 1, 23, 1)


def test_los_profile_adds_finite_specular_component(execution_context, cdl_config_factory) -> None:
    model = ClusteredDelayLine(cdl_config_factory("CDL-D"))
    request = ChannelSamplingRequest(2, 2, 1000.0, execution_context, execution_context.random_stream(5))
    cir, rays = model.realize_with_rays(request)
    assert cir.shape[-2] == 13
    assert rays.shape[-1] == 20
    assert np.all(np.isfinite(_numpy(cir.coefficients)))
