from math import pi

from channel_dataset.channels.wireless import evaluate_array_field
from channel_dataset.channels.wireless.antennas import load_antenna_layout


def test_isotropic_field_response_has_antenna_axis(execution_context, backend):
    layout = load_antenna_layout("profiles/wireless/antenna_isotropic_siso.yaml")
    theta = backend.array([pi / 2, pi / 3], dtype=execution_context.float_dtype)
    phi = backend.array([0.0, 0.2], dtype=execution_context.float_dtype)
    field_theta, field_phi = evaluate_array_field(
        layout, theta, phi, execution_context
    )
    assert backend.shape(field_theta) == (2, 1)
    power = field_theta * field_theta + field_phi * field_phi
    assert backend.to_scalar(backend.max(backend.abs(power - 1.0))) < 1e-12
