from math import pi

import numpy as np

from channel_dataset.channels.wireless.antennas import (
    AntennaPanel,
    IsotropicAntennaElement,
    Orientation,
)


def _numpy(value):
    if type(value).__module__.partition(".")[0] == "torch":
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _vector(field):
    return np.stack((_numpy(field.theta), _numpy(field.phi)), axis=-1)


def test_cross_polarized_fields_are_orthogonal(execution_context) -> None:
    panel = AntennaPanel(
        1,
        1,
        polarization_slant_angles_rad=(-pi / 4.0, pi / 4.0),
        element=IsotropicAntennaElement(),
    )
    negative = _vector(panel.field_pattern(0, 1.1, 0.4, execution_context))
    positive = _vector(panel.field_pattern(1, 1.1, 0.4, execution_context))
    assert np.allclose(np.sum(negative * positive, axis=-1), 0.0, atol=1.0e-12)
    assert np.allclose(np.sum(negative**2, axis=-1), 1.0, atol=1.0e-12)
    assert np.allclose(np.sum(positive**2, axis=-1), 1.0, atol=1.0e-12)


def test_mechanical_slant_preserves_field_power(execution_context) -> None:
    element = IsotropicAntennaElement(slant_angle_rad=0.3)
    base = element.field_pattern(1.0, 0.7, execution_context)
    rotated = element.field_pattern(
        1.0,
        0.7,
        execution_context,
        orientation=Orientation(0.2, -0.4, 0.5),
    )
    assert np.allclose(np.sum(_vector(base) ** 2), 1.0, atol=1.0e-12)
    assert np.allclose(np.sum(_vector(rotated) ** 2), 1.0, atol=1.0e-12)
