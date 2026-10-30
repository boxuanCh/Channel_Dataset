from math import pi

import numpy as np

from channel_dataset.channels.wireless.antennas import (
    IsotropicAntennaElement,
    Orientation,
    ThreeGPPAntennaElement,
)


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _power(field):
    return np.abs(_numpy(field.theta)) ** 2 + np.abs(_numpy(field.phi)) ** 2


def test_isotropic_power_is_angle_independent(execution_context) -> None:
    field = IsotropicAntennaElement().field_pattern(
        [0.2, pi / 2.0, 2.9], [-2.0, 0.0, 1.0], execution_context
    )
    assert np.allclose(_power(field), 1.0, atol=1.0e-12)


def test_three_gpp_boresight_gain_is_eight_dbi(execution_context) -> None:
    field = ThreeGPPAntennaElement().field_pattern(pi / 2.0, 0.0, execution_context)
    assert np.allclose(10.0 * np.log10(_power(field)), 8.0, atol=1.0e-10)


def test_element_orientation_rotates_boresight(execution_context) -> None:
    element = ThreeGPPAntennaElement()
    field = element.field_pattern(
        pi / 2.0,
        pi / 2.0,
        execution_context,
        orientation=Orientation(bearing_rad=pi / 2.0),
    )
    assert np.allclose(10.0 * np.log10(_power(field)), 8.0, atol=1.0e-10)
