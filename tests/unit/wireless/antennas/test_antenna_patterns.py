from math import pi

import numpy as np
import pytest

from channel_dataset.channels.wireless.antennas import ThreeGPPAntennaElement


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_three_gpp_horizontal_reference_points(execution_context) -> None:
    element = ThreeGPPAntennaElement()
    azimuth = np.deg2rad([0.0, 32.5, 65.0, 180.0]).tolist()
    gain = _numpy(element.power_gain_db(pi / 2.0, azimuth, execution_context))
    assert np.allclose(gain, [8.0, 5.0, -4.0, -22.0], atol=1.0e-10)


def test_three_gpp_pattern_is_symmetric(execution_context) -> None:
    element = ThreeGPPAntennaElement()
    angles = np.deg2rad([10.0, 40.0, 90.0, 160.0]).tolist()
    positive = _numpy(element.power_gain_db(pi / 2.0, angles, execution_context))
    negative = _numpy(
        element.power_gain_db(pi / 2.0, [-angle for angle in angles], execution_context)
    )
    assert np.allclose(positive, negative, atol=1.0e-12)


def test_vertical_and_combined_attenuation_are_capped(execution_context) -> None:
    element = ThreeGPPAntennaElement()
    gain = _numpy(element.power_gain_db([0.0, 0.0], [0.0, pi], execution_context))
    expected_vertical = 8.0 - 12.0 * (90.0 / 65.0) ** 2
    assert np.allclose(gain, [expected_vertical, -22.0], atol=1.0e-10)


@pytest.mark.torch
def test_numpy_torch_pattern_parity() -> None:
    from channel_dataset.core import ExecutionContext

    element = ThreeGPPAntennaElement()
    angles = [0.1, 0.8, 1.4]
    numpy_gain = element.power_gain_db(
        [0.5, 1.0, 2.0], angles, ExecutionContext("numpy", "double")
    )
    torch_gain = element.power_gain_db(
        [0.5, 1.0, 2.0], angles, ExecutionContext("torch", "double")
    )
    assert np.allclose(numpy_gain, torch_gain.detach().cpu().numpy(), atol=1.0e-12)
