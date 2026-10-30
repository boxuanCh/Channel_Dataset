from channel_dataset.channels.wireless import polarization_coupling


def test_copolar_unit_fields_have_unit_response(backend, execution_context):
    one = backend.ones((2, 1), dtype=execution_context.float_dtype)
    zero = backend.zeros((2, 1), dtype=execution_context.float_dtype)
    xpr = backend.ones((2,), dtype=execution_context.float_dtype)
    phases = backend.zeros((2, 4), dtype=execution_context.float_dtype)
    response = polarization_coupling(
        (one, zero), (one, zero), xpr, phases, execution_context
    )
    assert backend.shape(response) == (2, 1, 1)
    assert backend.to_scalar(backend.max(backend.abs(response - 1.0))) < 1e-12
