from channel_dataset.channels.wireless import CIRBatch, LargeScaleParameters, apply_large_scale_loss


def test_twenty_db_path_loss_scales_voltage_by_one_tenth(backend, realization_metadata):
    coefficients = backend.ones((1, 1, 1, 1, 1, 1, 1), dtype="complex128")
    zero = backend.zeros((1, 1, 1), dtype="float64")
    cir = CIRBatch(coefficients, backend.zeros((1, 1, 1, 1), dtype="float64"), realization_metadata)
    lsp = LargeScaleParameters(
        zero, zero, zero, zero, zero, zero, zero,
        backend.full((1, 1, 1), 20.0, dtype="float64"),
        realization_metadata,
    )
    attenuated = apply_large_scale_loss(cir, lsp)
    error = backend.abs(attenuated.coefficients - 0.1)
    assert backend.to_scalar(backend.max(error)) < 1e-12
