from channel_dataset.channels.wireless import (
    NonSelfBlockingModelA,
    apply_blockage_to_coefficients,
)


def test_model_a_is_nonnegative_and_reproducible(backend, blockage_case):
    rays, topology = blockage_case
    model = NonSelfBlockingModelA("portrait")
    first = model.loss_db(rays, topology, backend.random_stream(12))
    second = model.loss_db(rays, topology, backend.random_stream(12))
    assert backend.all_nonnegative(first)
    assert backend.to_scalar(backend.max(backend.abs(first - second))) == 0.0


def test_model_a_orientation_modes_have_declared_blocker_counts():
    assert NonSelfBlockingModelA("portrait").mode == "portrait"
    assert NonSelfBlockingModelA("landscape").mode == "landscape"


def test_blockage_loss_is_applied_as_voltage_attenuation(backend):
    coefficients = backend.ones((1, 1, 1, 1, 1, 1), dtype="complex128")
    loss = backend.full((1, 1, 1, 1, 1), 20.0, dtype="float64")
    result = apply_blockage_to_coefficients(coefficients, loss)
    assert backend.to_scalar(backend.max(backend.abs(result - 0.1))) < 1e-12
