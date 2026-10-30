import numpy as np
import pytest

from channel_dataset.channels.wireless import ChannelSamplingRequest
from channel_dataset.channels.wireless.tr38901 import CDLConfig, ClusteredDelayLine
from channel_dataset.core import ExecutionContext


@pytest.mark.statistical
@pytest.mark.parametrize(
    "backend_name", ["numpy", pytest.param("torch", marks=pytest.mark.torch)]
)
def test_cdl_a_average_siso_power_is_near_unity(backend_name) -> None:
    execution = ExecutionContext(backend_name, "double")
    model = ClusteredDelayLine(CDLConfig(
        profile="CDL-A",
        standard_version="16.1.0",
        delay_spread_s=1.0e-7,
        carrier_frequency_hz=3.5e9,
        bandwidth_hz=1.0e8,
        link_direction="downlink",
        base_station_array_profile="package://profiles/wireless/antenna_isotropic_siso.yaml",
        terminal_array_profile="package://profiles/wireless/antenna_isotropic_siso.yaml",
        base_station_orientation_rad=(0.0, 0.0, 0.0),
        terminal_orientation_rad=(0.0, 0.0, 0.0),
        base_station_velocity_mps=(0.0, 0.0, 0.0),
        terminal_velocity_mps=(0.0, 0.0, 0.0),
        link_distance_m=100.0,
        k_factor_db=None,
    ))
    request = ChannelSamplingRequest(2000, 1, 1.0, execution, execution.random_stream(123))
    coefficients = model.realize(request).coefficients
    if backend_name == "torch":
        coefficients = coefficients.detach().cpu().numpy()
    power = np.sum(np.abs(coefficients[:, 0, 0, 0, 0, :, 0]) ** 2, axis=-1)
    assert np.mean(power) == pytest.approx(1.0, abs=0.04)
