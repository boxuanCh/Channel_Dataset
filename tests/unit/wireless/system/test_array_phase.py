from channel_dataset.channels.wireless import array_phase_offsets
from channel_dataset.channels.wireless.antennas import load_antenna_layout


def test_centered_siso_array_has_zero_phase(backend, execution_context):
    layout = load_antenna_layout("profiles/wireless/antenna_isotropic_siso.yaml")
    direction = backend.array([[1.0, 0.0, 0.0]], dtype=execution_context.float_dtype)
    response = array_phase_offsets(
        direction, direction, layout, layout, execution_context
    )
    assert backend.shape(response) == (1, 1, 1)
    assert backend.to_scalar(backend.max(backend.abs(response - 1.0))) < 1e-12
